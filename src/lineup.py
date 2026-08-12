"""Roster state, optimal starting-lineup fill, and roster valuation.

The objective the whole engine maximises lives here. It is deliberately *not*
"total points on my roster" -- it is "points my best legal starting lineup
scores", with a bye-week penalty and a small credit for bench value. In a
14-team league with only 5 bench spots that distinction changes real picks:
hoarding a fourth good RB you can never start is close to worthless.
"""

from __future__ import annotations

from .league import LeagueConfig


def optimal_lineup(roster: list, league: LeagueConfig) -> tuple[dict, float]:
    """Fill the starting lineup to maximise projected points.

    Greedy-by-position-then-flex is optimal for this lineup shape, because the
    flex accepts a strict superset of the positions competing for it.
    """
    by_pos: dict[str, list] = {}
    for p in roster:
        by_pos.setdefault(p.position, []).append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: x.projection, reverse=True)

    used: set[str] = set()
    starters: dict[str, list] = {}
    total = 0.0

    for pos, count in league.starters.items():
        if pos == "FLEX":
            continue
        picked = []
        for p in by_pos.get(pos, []):
            if len(picked) >= count:
                break
            picked.append(p)
            used.add(p.pid)
            total += p.projection
        starters[pos] = picked

    flex_count = league.starters.get("FLEX", 0)
    pool = [
        p
        for pos in league.flex_positions
        for p in by_pos.get(pos, [])
        if p.pid not in used
    ]
    pool.sort(key=lambda x: x.projection, reverse=True)
    flex = pool[:flex_count]
    for p in flex:
        used.add(p.pid)
        total += p.projection
    starters["FLEX"] = flex

    return starters, total


def bye_conflicts(starters: dict, allowed: int) -> dict[int, int]:
    """Bye weeks where more starters than `allowed` are simultaneously off."""
    counts: dict[int, int] = {}
    for group in starters.values():
        for p in group:
            if p.bye:
                counts[p.bye] = counts.get(p.bye, 0) + 1
    return {wk: n for wk, n in counts.items() if n > allowed}


def position_counts(roster: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in roster:
        counts[p.position] = counts.get(p.position, 0) + 1
    return counts


def remaining_starter_needs(roster: list, league: LeagueConfig) -> dict[str, int]:
    """How many starting slots at each position are still unfilled."""
    counts = position_counts(roster)
    needs: dict[str, int] = {}
    for pos, want in league.starters.items():
        if pos == "FLEX":
            continue
        needs[pos] = max(0, want - counts.get(pos, 0))

    flex_want = league.starters.get("FLEX", 0)
    surplus = sum(
        max(0, counts.get(pos, 0) - league.starters.get(pos, 0))
        for pos in league.flex_positions
    )
    needs["FLEX"] = max(0, flex_want - surplus)
    return needs


def unfilled_required(roster: list, league: LeagueConfig) -> list[str]:
    """Required starting slots (flex excluded) with nobody in them yet."""
    counts = position_counts(roster)
    return [
        pos
        for pos, need in league.starters.items()
        if pos != "FLEX" and counts.get(pos, 0) < need
    ]


def can_draft(
    player, roster: list, league: LeagueConfig, ecfg: dict, picks_left: int
) -> bool:
    """Roster-legality and sanity rules for a candidate pick."""
    if len(roster) >= league.rounds:
        return False

    counts = position_counts(roster)
    cap = ecfg["max_at_position"].get(player.position)
    if cap is not None and counts.get(player.position, 0) >= cap:
        return False

    # When picks remaining exactly equals empty starting slots, every pick left
    # has to fill one -- otherwise you finish with a hole scoring zero.
    must = unfilled_required(roster, league)
    if must and picks_left <= len(must):
        return player.position in must

    # Kickers and defenses are fungible; taking one early is a real mistake, so
    # the engine simply refuses until the end of the draft is in sight.
    late = ecfg["late_round_only"].get(player.position)
    if late is not None and picks_left > late:
        return False

    return True
