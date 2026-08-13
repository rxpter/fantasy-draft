"""Replay a finished roster through a real season, week by week.

Season totals are the easy way to score a backtest and they are close to
useless here: they cannot see a bye week, cannot see a player missing six
games, and therefore cannot test the depth targets, bye penalty, or
`starter_miss_rate` that the objective function spends most of its effort on.
So this replays every week and fills a legal lineup each time.

One methodological choice worth stating plainly: the lineup is set by
**preseason projection**, not by what the player went on to score. Picking the
hindsight-optimal lineup each week would inflate every strategy and reward
hoarding lottery tickets. Starting your highest-projected available players is
what a manager could actually have done with the information they had.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.league import LeagueConfig  # noqa: E402
from src.netcache import FetchError, get_json  # noqa: E402
from src.sleeper import BASE, FANTASY_POSITIONS, SCORING_KEY  # noqa: E402

CACHE_TTL = 24 * 30
REGULAR_WEEKS = tuple(range(1, 15))    # fantasy regular season
PLAYOFF_WEEKS = (15, 16, 17)


def fetch_weekly_actuals(season: int, scoring: str = "ppr", weeks=range(1, 18)) -> dict:
    """{week: {pid: actual points}}. One request per week, all positions."""
    key = SCORING_KEY.get(scoring, "pts_ppr")
    positions = "&".join(f"position[]={p}" for p in FANTASY_POSITIONS)
    out: dict[int, dict[str, float]] = {}

    for wk in weeks:
        url = f"{BASE}/stats/nfl/{season}/{wk}?season_type=regular&{positions}"
        try:
            rows = get_json(url, ttl_hours=CACHE_TTL, timeout=90)
        except FetchError:
            out[wk] = {}
            continue
        week_map: dict[str, float] = {}
        if isinstance(rows, list):
            for row in rows:
                pid = row.get("player_id")
                pts = (row.get("stats") or {}).get(key)
                if pid and isinstance(pts, (int, float)):
                    week_map[str(pid)] = float(pts)
        out[wk] = week_map
    return out


def _fill_lineup(active: list, league: LeagueConfig) -> list:
    """Best legal lineup by preseason projection from the players who suited up."""
    buckets: dict[str, list] = {}
    for p in active:
        buckets.setdefault(p.position, []).append(p)
    for pos in buckets:
        buckets[pos].sort(key=lambda x: x.projection, reverse=True)

    chosen, used = [], set()
    for pos, count in league.starters.items():
        if pos == "FLEX":
            continue
        for p in buckets.get(pos, [])[:count]:
            chosen.append(p)
            used.add(p.pid)

    flex = [
        p
        for pos in league.flex_positions
        for p in buckets.get(pos, [])[league.starters.get(pos, 0):]
        if p.pid not in used
    ]
    flex.sort(key=lambda x: x.projection, reverse=True)
    chosen.extend(flex[: league.starters.get("FLEX", 0)])
    return chosen


def score_roster(roster: list, weekly: dict, league: LeagueConfig) -> dict:
    """Replay a roster across the season. Returns points and diagnostics."""
    regular = playoffs = 0.0
    started_slots = 0
    empty_slots = 0
    by_position: dict[str, float] = {}
    weeks_scored = 0

    slot_capacity = league.starter_slots

    for wk, stats in sorted(weekly.items()):
        active = [p for p in roster if p.pid in stats]
        if not active:
            continue
        weeks_scored += 1
        lineup = _fill_lineup(active, league)
        points = 0.0
        for p in lineup:
            pts = stats.get(p.pid, 0.0)
            points += pts
            by_position[p.position] = by_position.get(p.position, 0.0) + pts

        started_slots += len(lineup)
        empty_slots += max(0, slot_capacity - len(lineup))
        if wk in PLAYOFF_WEEKS:
            playoffs += points
        else:
            regular += points

    return {
        "regular": round(regular, 1),
        "playoffs": round(playoffs, 1),
        "total": round(regular + playoffs, 1),
        "weeks": weeks_scored,
        "empty_starter_slots": empty_slots,
        "by_position": {k: round(v, 1) for k, v in by_position.items()},
    }


def hit_rate(roster: list, actuals: dict) -> dict:
    """How the drafted players did against their own preseason projections."""
    hits = busts = 0
    total_proj = total_actual = 0.0
    for p in roster:
        actual = actuals.get(p.pid, 0.0)
        total_proj += p.projection
        total_actual += actual
        if p.projection > 0:
            ratio = actual / p.projection
            if ratio >= 1.15:
                hits += 1
            elif ratio <= 0.6:
                busts += 1
    return {
        "hits": hits,
        "busts": busts,
        "projected": round(total_proj, 1),
        "actual": round(total_actual, 1),
    }
