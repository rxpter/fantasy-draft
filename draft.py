#!/usr/bin/env python3
"""Entry point: live draft assistant.

    python draft.py --username YOURNAME       # auto-find your draft
    python draft.py --draft-id 123456789      # or point at it directly
    python draft.py --mock                    # rehearse offline, no draft needed
    python draft.py --list --username YOURNAME
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import board, sleeper  # noqa: E402
from src.league import LeagueConfig, flex_allocation, league_from_draft  # noqa: E402
from src.netcache import FetchError  # noqa: E402
from src.pool import build_pool  # noqa: E402
from src.recommend import build_state, recommend  # noqa: E402
from src.webboard import BoardState, serialize, start_server  # noqa: E402

CONFIG_PATH = ROOT / "config.json"

# Running the Monte Carlo on every pick of a 210-pick draft is wasted work --
# it only changes decisions when your turn is close.
SIM_WINDOW = 12


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sleeper live draft assistant")
    ap.add_argument("--username", help="your Sleeper username")
    ap.add_argument("--draft-id", help="Sleeper draft id")
    ap.add_argument("--slot", type=int, help="your draft slot (1-indexed)")
    ap.add_argument("--season", help="override season, e.g. 2026")
    ap.add_argument("--sims", type=int, help="Monte Carlo iterations per refresh")
    ap.add_argument("--top", type=int, help="how many recommendations to show")
    ap.add_argument("--once", action="store_true", help="render once and exit")
    ap.add_argument("--mock", action="store_true", help="offline rehearsal draft")
    ap.add_argument("--mock-slot", type=int, default=7, help="your slot in the mock")
    ap.add_argument("--mock-delay", type=float, default=0.0, help="seconds between mock picks")
    ap.add_argument("--list", action="store_true", help="list your leagues, drafts, and mocks")
    ap.add_argument(
        "--no-auto-settings",
        action="store_true",
        help="trust config.json instead of reading league settings off the draft",
    )
    ap.add_argument(
        "--web", action="store_true", help="serve the web board on localhost instead of the terminal"
    )
    ap.add_argument("--port", type=int, default=8770, help="port for the web board")
    ap.add_argument("--no-color", action="store_true")
    return ap.parse_args()


def resolve_season(cfg_season: str | None, override: str | None) -> str:
    if override:
        return override
    if cfg_season:
        return cfg_season
    try:
        return sleeper.get_state().get("season") or "2026"
    except FetchError:
        return "2026"


def do_list(username: str, season: str) -> int:
    try:
        user = sleeper.get_user(username)
    except FetchError as exc:
        print(f"error: {exc}")
        return 1
    if not user or not user.get("user_id"):
        print(f"no Sleeper user named {username!r}")
        return 1

    print(f"user: {user.get('display_name')}  (id {user['user_id']})")

    def describe(d: dict, indent: str) -> None:
        slot = (d.get("draft_order") or {}).get(user["user_id"])
        s = d.get("settings") or {}
        print(
            f"{indent}draft {d['draft_id']}  status={d.get('status')}  "
            f"type={d.get('type')}  teams={s.get('teams', '?')}  "
            f"rounds={s.get('rounds', '?')}  your_slot={slot or '?'}"
        )

    seen: set[str] = set()
    leagues = sleeper.get_user_leagues(user["user_id"], season) or []
    for lg in leagues:
        print(f"\n  league: {lg.get('name')}   teams={lg.get('total_rosters')}  id={lg['league_id']}")
        try:
            drafts = sleeper.get_league_drafts(lg["league_id"]) or []
        except FetchError as exc:
            print(f"    (could not list drafts: {exc})")
            continue
        for d in drafts:
            seen.add(d["draft_id"])
            describe(d, "    ")

    # Mock drafts belong to no league, so they need their own lookup.
    try:
        mocks = [d for d in sleeper.get_user_drafts(user["user_id"], season) if d["draft_id"] not in seen]
    except FetchError:
        mocks = []
    if mocks:
        print(f"\n  mock drafts ({len(mocks)}):")
        for d in mocks:
            describe(d, "    ")

    if not leagues and not mocks:
        print(f"\n  no {season} leagues or drafts found for this user")
        return 1
    return 0


def apply_draft_settings(
    draft: dict, fallback: LeagueConfig, use_draft: bool = True
) -> LeagueConfig:
    """Read the real league shape off the draft object.

    A Sleeper mock is rarely the same shape as your home league, and team count
    drives replacement level, so believing the draft beats believing
    config.json. `--no-auto-settings` keeps config.json in charge.
    """
    derived, notes, unsupported = league_from_draft(draft, fallback)
    name = (draft.get("metadata") or {}).get("name") or "draft"
    kind = draft.get("type")

    print(f"draft: {name}  |  type={kind}  |  status={draft.get('status')}")

    if kind and kind != "snake":
        print(f"  !! WARNING: draft type is {kind!r}; the pick model assumes snake")
    for item in unsupported:
        print(f"  !! UNSUPPORTED: {item}")
    if unsupported:
        print("  !! this engine models standard 1-QB lineups; advice here will be wrong")

    if not use_draft:
        if notes:
            print("  auto-settings off; draft differs from config.json: " + "; ".join(notes))
        return fallback

    if notes:
        print("  settings taken from draft: " + "; ".join(notes))
    print(
        f"  {derived.teams} teams | {derived.rounds} rounds | {derived.scoring} | "
        f"starters {derived.starters} | bench {derived.bench}"
    )
    return derived


class MockDraft:
    """Offline draft that plays itself, so the engine can be rehearsed."""

    def __init__(self, players, league: LeagueConfig, my_slot: int, seed: int = 7):
        rng = random.Random(seed)
        pool = [p for p in players if p.adp < 900]
        pool.sort(key=lambda p: p.adp + p.sigma * rng.gauss(0, 1))
        self.order = pool
        self.league = league
        self.my_slot = my_slot
        self.picks: list = []
        self._ptr = 0

    def advance(self, my_choice=None) -> None:
        pick_no = len(self.picks) + 1
        if pick_no > self.league.total_picks:
            return
        slot = self.league.slot_from_pick(pick_no)

        if slot == self.my_slot and my_choice is not None:
            player = my_choice
        else:
            player = None
            while self._ptr < len(self.order):
                cand = self.order[self._ptr]
                self._ptr += 1
                if not any(pk["player_id"] == cand.pid for pk in self.picks):
                    player = cand
                    break
            if player is None:
                return

        self.picks.append(
            {
                "pick_no": pick_no,
                "round": (pick_no - 1) // self.league.teams + 1,
                "draft_slot": slot,
                "player_id": player.pid,
                "metadata": {"last_name": player.name.split()[-1]},
            }
        )


def run_mock(players, diagnostics, league, cfg, args, web_state=None) -> int:
    my_slot = args.mock_slot
    mock = MockDraft(players, league, my_slot)
    by_pid = {p.pid: p for p in players}
    ecfg = cfg["engine"]

    # A mock with no delay finishes faster than you can read it. On the web
    # board that defeats the point, so pace it unless told otherwise.
    delay = args.mock_delay or (1.5 if web_state is not None else 0.0)

    print(f"\nmock draft: {league.teams} teams, {league.rounds} rounds, you are slot {my_slot}\n")

    def publish(st, recs, outlook, status):
        meta = {
            "status": status,
            "top_n": cfg["ui"]["top_n"],
            "roster_targets": ecfg["roster_targets"],
            "sim_note": "mock",
        }
        if web_state is not None:
            web_state.set(serialize(st, recs, outlook, league, diagnostics, meta))
            print(f"\r  mock pick {st.current_pick}/{league.total_picks}   ", end="", flush=True)
        else:
            board.clear()
            print(board.render(st, recs, outlook, league, diagnostics, meta))

    while len(mock.picks) < league.total_picks:
        st = build_state(mock.picks, by_pid, league, my_slot)
        if st.on_the_clock:
            recs, outlook = recommend(players, st, league, ecfg, run_sim=True)
            if not recs:
                break
            publish(st, recs, outlook, "mock")
            mock.advance(my_choice=recs[0].player)
            if delay:
                time.sleep(delay)
        else:
            mock.advance()

    st = build_state(mock.picks, by_pid, league, my_slot)
    recs, outlook = recommend(players, st, league, ecfg, run_sim=False)
    publish(st, recs, outlook, "mock complete")
    if web_state is not None:
        print("\n  mock complete -- board stays up, ctrl-c to quit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(CONFIG_PATH)
    board.enable_colors(cfg["ui"]["color"] and not args.no_color)

    league = LeagueConfig.from_dict(cfg["league"])
    season = resolve_season(None, args.season)

    if args.sims:
        cfg["engine"]["sims"] = args.sims
    if args.top:
        cfg["ui"]["top_n"] = args.top

    username = args.username or cfg["draft"].get("sleeper_username")

    if args.list:
        if not username:
            print("--list needs --username")
            return 1
        return do_list(username, season)

    # ---- resolve the live draft first ---------------------------------
    # The draft tells us the real team count and roster shape, and replacement
    # level depends on both -- so this has to happen before the pool is built.
    draft = None
    draft_id = None
    my_slot = None
    user_id = None

    if not args.mock:
        draft_id = args.draft_id or cfg["draft"].get("draft_id")
        my_slot = args.slot or cfg["draft"].get("my_draft_slot")

        if not draft_id and username:
            draft_id, note = sleeper.resolve_draft_id(username, season)
            print(f"draft lookup: {note}")
        if not draft_id:
            print("\nno draft id. Pass --draft-id, or --username to auto-detect.")
            print("Run with --mock to rehearse offline.")
            return 1

        try:
            draft = sleeper.get_draft(draft_id)
        except FetchError as exc:
            print(f"error loading draft {draft_id}: {exc}")
            return 1

        league = apply_draft_settings(draft, league, use_draft=not args.no_auto_settings)

        if username:
            try:
                user_id = (sleeper.get_user(username) or {}).get("user_id")
            except FetchError:
                user_id = None
        if not my_slot and user_id:
            my_slot = sleeper.my_slot_from_draft(draft, user_id)

        if my_slot:
            print(f"  your slot: {my_slot}")
        elif user_id:
            print("  slot not set yet -- will pick it up when the draft order is drawn")
        else:
            print("  slot unknown -- pass --slot N, or --username to auto-detect")

    print(f"loading player pool for {season} ({league.teams}-team {league.scoring})...")
    t0 = time.time()
    try:
        players, diagnostics = build_pool(cfg, league, season)
    except FetchError as exc:
        print(f"error: {exc}")
        return 1

    flex = flex_allocation(players, league)
    print(
        f"  {diagnostics['players_considered']} players | "
        f"{diagnostics['with_projection']} projected | "
        f"{diagnostics['adp_matched']}/{diagnostics['adp_rows']} ADP matched | "
        f"{time.time() - t0:.1f}s"
    )
    levels = diagnostics["replacement_levels"]
    print(
        "  replacement level: "
        + "  ".join(f"{k} {v:.0f}" for k, v in sorted(levels.items()) if k in ("QB", "RB", "WR", "TE"))
        + f"   | flex fills as {flex}"
    )
    if diagnostics.get("overridden"):
        print(f"  using {diagnostics['overridden']} projections from data/my_projections.csv")

    web_state = None
    if args.web:
        web_state = BoardState()
        try:
            _server, url = start_server(web_state, args.port)
        except OSError as exc:
            print(f"error: could not start web board on port {args.port}: {exc}")
            print("try a different port with --port 8771")
            return 1
        print(f"\n  web board: {url}   (open this on your second monitor)\n")

    if args.mock:
        return run_mock(players, diagnostics, league, cfg, args, web_state)

    by_pid = {p.pid: p for p in players}
    ecfg = cfg["engine"]
    poll = cfg["ui"]["poll_seconds"]
    last_count = -1
    cached = None
    status = draft.get("status", "?")
    ticks = 0

    while True:
        try:
            picks = sleeper.get_draft_picks(draft_id)
        except FetchError as exc:
            print(f"\n(fetch failed: {exc}; retrying)")
            time.sleep(poll)
            continue

        st = build_state(picks, by_pid, league, my_slot)

        if len(picks) != last_count or cached is None:
            last_count = len(picks)

            # Publish the new pick straight away using the previous ranking,
            # minus anyone who has just been taken. The simulation costs about
            # a second, and waiting for it before showing that a pick landed is
            # what made the board feel slow.
            if web_state is not None and cached is not None:
                stale_recs = [r for r in cached[0] if r.player.pid not in st.taken_pids]
                web_state.set(
                    serialize(
                        st, stale_recs, cached[1], league, diagnostics,
                        {
                            "status": status,
                            "top_n": cfg["ui"]["top_n"],
                            "roster_targets": ecfg["roster_targets"],
                            "sim_note": "thinking",
                            "computing": True,
                        },
                    )
                )

            run_sim = bool(my_slot) and (st.on_the_clock or st.picks_until_mine <= SIM_WINDOW)
            t = time.time()
            recs, outlook = recommend(players, st, league, ecfg, run_sim=run_sim)
            note = (
                f"sim {ecfg['sims']}x in {time.time() - t:.1f}s"
                if run_sim
                else "sim idle (not near your pick)"
            )
            cached = (recs, outlook, note)

        recs, outlook, note = cached

        # The pick feed is the thing that changes second to second; draft status
        # barely moves, so refresh it occasionally rather than every tick.
        ticks += 1
        if ticks % 10 == 1 or (my_slot is None and user_id):
            try:
                fresh = sleeper.get_draft(draft_id)
                status = fresh.get("status", status)
                # In a mock lobby the order is drawn after you join, so keep
                # looking until the slot appears.
                if my_slot is None and user_id:
                    my_slot = sleeper.my_slot_from_draft(fresh, user_id)
                    if my_slot:
                        last_count = -1  # force a re-rank now that picks are ours
            except FetchError:
                pass

        meta = {
            "status": status,
            "top_n": cfg["ui"]["top_n"],
            "roster_targets": cfg["engine"]["roster_targets"],
            "sim_note": note,
        }
        if web_state is not None:
            web_state.set(serialize(st, recs, outlook, league, diagnostics, meta))
            clock = "ON THE CLOCK" if st.on_the_clock else f"{st.picks_until_mine} until yours"
            print(
                f"\r  pick {st.current_pick}/{league.total_picks} | {status} | {clock} | {note}    ",
                end="",
                flush=True,
            )
        else:
            board.clear()
            print(board.render(st, recs, outlook, league, diagnostics, meta))

        if args.once or status == "complete":
            return 0
        time.sleep(poll)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nbye")
        sys.exit(0)
