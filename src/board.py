"""Terminal rendering.

Plain ANSI, no curses and no third-party TUI library -- during a live draft the
thing that matters is that it starts instantly and never crashes.
"""

from __future__ import annotations

import os
import shutil
import sys

from .lineup import (
    bye_conflicts,
    optimal_lineup,
    position_counts,
    remaining_starter_needs,
)

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

POS_COLOR = {
    "QB": "\033[95m",
    "RB": "\033[92m",
    "WR": "\033[96m",
    "TE": "\033[93m",
    "K": "\033[90m",
    "DEF": "\033[94m",
}

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
GREY = "\033[90m"

_ENABLED = True

# Time value above this is worth flagging: the player's uncertainty alone is
# carrying real weight, which is the late-round profile you want.
UPSIDE_HIGHLIGHT = 10.0


def enable_colors(flag: bool) -> None:
    """Turn on VT processing on Windows consoles that need it."""
    global _ENABLED
    _ENABLED = flag and sys.stdout.isatty()
    if not _ENABLED:
        return
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if _ENABLED else text


def clear() -> None:
    if _ENABLED:
        sys.stdout.write("\033[H\033[J")
    else:
        sys.stdout.write("\n" * 3)


def rule(width: int, char: str = "-") -> str:
    return c(char * width, GREY)


def _pos(p: str) -> str:
    return c(f"{p:<3}", POS_COLOR.get(p, ""))


def _pct_color(v: float) -> str:
    if v >= 0.75:
        return GREEN
    if v >= 0.4:
        return YELLOW
    return RED


def render(st, recs, outlook, league, diagnostics, meta: dict) -> str:
    width = min(shutil.get_terminal_size((110, 40)).columns, 118)
    lines: list[str] = []

    # ---- header -------------------------------------------------------
    status = meta.get("status", "?")
    title = f" DRAFT ASSISTANT  |  pick {st.current_pick}/{league.total_picks}  |  {status} "
    lines.append(c(BOLD + title + RESET, CYAN))

    if st.my_slot:
        rnd = (st.current_pick - 1) // league.teams + 1
        if st.on_the_clock:
            lines.append(c(BOLD + ">>> YOU ARE ON THE CLOCK <<<" + RESET, GREEN))
        elif st.my_future_picks:
            nxt = st.my_future_picks[0]
            lines.append(
                f"slot {st.my_slot}  |  round {rnd}  |  "
                + c(f"{st.picks_until_mine} picks until yours (#{nxt})", YELLOW)
            )
        else:
            lines.append(c("your draft is complete", GREY))
    else:
        lines.append(c("draft slot unknown -- pass --slot N for personalised advice", YELLOW))

    lines.append(rule(width))

    # ---- recommendations ----------------------------------------------
    if recs:
        head = (
            f"{'#':<3}{'PLAYER':<20}{'POS':<5}{'TM':<4}{'BYE':<4}"
            f"{'PROJ':>7}{'VOR':>6}{'VAL':>6}{'UP':>5}{'BACK':>6}{'SIM':>8}   NOTES"
        )
        lines.append(c(BOLD + head + RESET, ""))

        for i, r in enumerate(recs[: meta.get("top_n", 12)], 1):
            p = r.player
            back = f"{r.survive_next * 100:>4.0f}%"
            sim = f"{r.sim_score:>8.1f}" if r.sim_score is not None else f"{'--':>8}"
            marker = c("*", GREEN) if i == 1 else " "
            # Highlight genuine boom candidates -- a big UP on a small VAL is
            # exactly the late-round shape worth gambling on.
            up_txt = f"{p.upside:>5.0f}"
            up = c(up_txt, GREEN) if p.upside >= UPSIDE_HIGHLIGHT else up_txt
            row = (
                f"{marker}{i:<2}"
                f"{p.name[:19]:<20}"
                f"{_pos(p.position)}  "
                f"{(p.team or '--'):<4}"
                f"{(str(p.bye) if p.bye else '--'):<4}"
                f"{p.projection:>7.1f}"
                f"{p.vor:>6.0f}"
                f"{p.val:>6.0f}"
                f"{up}"
                f"{c(back, _pct_color(r.survive_next)):>6}"
                f"{sim}   "
                f"{c(r.note, GREY)}"
            )
            lines.append(row)

        if recs[0].sim_score is not None and len(recs) > 1:
            gap = recs[0].sim_score - recs[1].sim_score
            verdict = (
                f"take {recs[0].player.name}"
                if gap > 1.5
                else f"{recs[0].player.name} or {recs[1].player.name} -- near tie"
            )
            lines.append("")
            lines.append(c(BOLD + f"  VERDICT: {verdict}  (+{gap:.1f} pts)" + RESET, GREEN))
    else:
        lines.append(c("no candidates -- roster full or draft over", GREY))

    lines.append(rule(width))

    # ---- cost of waiting ----------------------------------------------
    if outlook:
        lines.append(c(BOLD + "COST OF WAITING (value lost at each position by your next pick)" + RESET, ""))
        for o in outlook[:6]:
            bar_len = max(0, min(28, int(o.cost_of_waiting / 2)))
            colour = RED if o.cost_of_waiting >= 15 else (YELLOW if o.cost_of_waiting >= 6 else GREEN)
            bar = c("#" * bar_len, colour)
            best = o.best_player.name[:20] if o.best_player else "--"
            lines.append(
                f"  {_pos(o.position)} {o.cost_of_waiting:>6.1f}  {bar:<40} "
                + c(f"best: {best} | ~{o.expected_gone:.1f} gone", GREY)
            )
        lines.append(rule(width))

    # ---- my roster ----------------------------------------------------
    if st.my_slot:
        starters, total = optimal_lineup(st.my_roster, league)
        needs = remaining_starter_needs(st.my_roster, league)
        conflicts = bye_conflicts(starters, 2)

        lines.append(
            c(BOLD + f"YOUR ROSTER ({len(st.my_roster)}/{league.rounds})" + RESET, "")
            + c(f"   projected starters: {total:.1f} pts", GREY)
        )
        if st.my_roster:
            for pos in ("QB", "RB", "WR", "TE", "FLEX", "K", "DEF"):
                group = starters.get(pos, [])
                if not group:
                    continue
                names = ", ".join(f"{q.name} ({q.bye or '?'})" for q in group)
                lines.append(f"  {_pos(pos)} {names}")
            started = {q.pid for g in starters.values() for q in g}
            bench = [q for q in st.my_roster if q.pid not in started]
            if bench:
                lines.append(
                    c(f"  BN  {', '.join(q.name for q in bench)}", GREY)
                )
        else:
            lines.append(c("  (empty)", GREY))

        unmet = ", ".join(f"{k}x{v}" for k, v in needs.items() if v > 0) or "starters filled"
        lines.append(c(f"  still needed: {unmet}", YELLOW if any(needs.values()) else GREEN))

        # Roster shape against depth targets -- the guard against finishing with
        # three running backs and a bench full of receivers.
        targets = meta.get("roster_targets") or {}
        if targets:
            counts = position_counts(st.my_roster)
            parts = []
            for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
                if pos not in targets:
                    continue
                have, want = counts.get(pos, 0), targets[pos]
                txt = f"{pos} {have}/{want}"
                parts.append(c(txt, RED if have < want else GREEN))
            lines.append("  depth: " + "  ".join(parts))
        if conflicts:
            detail = ", ".join(f"wk{w}: {n} starters" for w, n in sorted(conflicts.items()))
            lines.append(c(f"  bye conflicts: {detail}", RED))
        lines.append(rule(width))

    # ---- recent picks --------------------------------------------------
    recent = st.picks[-6:]
    if recent:
        parts = []
        for pk in recent:
            nm = (pk.get("metadata") or {}).get("last_name") or pk.get("player_id") or "?"
            parts.append(f"{pk.get('pick_no', '?')}.{nm}")
        lines.append(c("recent: " + "  ".join(parts), GREY))

    footer = (
        f"proj {diagnostics.get('with_projection', 0)} players | "
        f"adp {diagnostics.get('adp_matched', 0)}/{diagnostics.get('adp_rows', 0)} matched | "
        f"{meta.get('sim_note', '')} | ctrl-c to quit"
    )
    lines.append(c(footer, GREY))

    return "\n".join(lines)
