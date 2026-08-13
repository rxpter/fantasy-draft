"""Rebuild a season's player pool as it looked before that season started.

The hard constraint here is avoiding lookahead. A backtest that quietly uses
post-season information will flatter the engine and teach you nothing.

Sleeper's archived payloads are a mixed bag, and the split matters:

* `stats.pts_ppr` and `stats.adp_ppr` on the projections endpoint ARE
  period-correct -- they are the preseason snapshot for that year. Verified two
  ways: every player carries a uniform `gp = 18`, and the 2024 file projects
  Christian McCaffrey for 277.9 points at ADP 1.2 when he went on to play four
  games.
* The embedded `player` object is NOT. It is today's metadata bolted onto a
  historical stats row -- the 2023 file lists Saquon Barkley on PHI and Derrick
  Henry on BAL, neither of which was true in 2023.

So points and ADP come from Sleeper, while team and bye week come from
FantasyFootballCalculator's archive for that year. Experience is derived
backwards from today's value. Depth-chart and injury signals are dropped
entirely, because no period-correct source exists for them.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import adp as adp_mod  # noqa: E402
from src.league import LeagueConfig, compute_replacement_levels  # noqa: E402
from src.netcache import FetchError, get_json  # noqa: E402
from src.pool import Player, apply_market_ranking, blend_adp  # noqa: E402
from src.sleeper import ADP_KEY, BASE, FANTASY_POSITIONS, SCORING_KEY  # noqa: E402
from src.survival import effective_sigma  # noqa: E402
from src.upside import estimate_sd, option_value, waiver_levels  # noqa: E402

# `years_exp` in the archive is today's value, so it has to be walked back.
REFERENCE_SEASON = 2026

CACHE_TTL = 24 * 30  # historical data never changes; cache it hard


def fetch_season_projections(season: int, scoring: str = "ppr") -> dict:
    """pid -> {points, adp, name, position, years_exp_now}.

    Keeps the fields the live client throws away, because the backtest needs
    names to join against FFC and experience to walk backwards.
    """
    pts_key = SCORING_KEY.get(scoring, "pts_ppr")
    adp_key = ADP_KEY.get(scoring, "adp_ppr")
    out: dict[str, dict] = {}

    for pos in FANTASY_POSITIONS:
        url = (
            f"{BASE}/projections/nfl/{season}"
            f"?season_type=regular&position[]={pos}&order_by={pts_key}"
        )
        try:
            rows = get_json(url, ttl_hours=CACHE_TTL, timeout=90)
        except FetchError:
            continue
        if not isinstance(rows, list):
            continue

        for row in rows:
            pid = row.get("player_id")
            stats = row.get("stats") or {}
            player = row.get("player") or {}
            pts = stats.get(pts_key)
            if not pid or not isinstance(pts, (int, float)) or pts <= 0:
                continue

            adp = stats.get(adp_key)
            if not isinstance(adp, (int, float)) or not 0 < adp < 900:
                adp = None

            name = " ".join(
                p for p in (player.get("first_name"), player.get("last_name")) if p
            )
            position = player.get("position") or pos
            if position == "DEF" and not name:
                name = f"{pid} D/ST"

            key = str(pid)
            if key not in out or pts > out[key]["points"]:
                out[key] = {
                    "points": float(pts),
                    "adp": adp,
                    "name": name or key,
                    "position": position,
                    "years_exp_now": player.get("years_exp"),
                }
    return out


def fetch_season_actuals(season: int, scoring: str = "ppr") -> dict:
    """pid -> actual season fantasy points. Ground truth."""
    pts_key = SCORING_KEY.get(scoring, "pts_ppr")
    out: dict[str, float] = {}
    for pos in FANTASY_POSITIONS:
        url = (
            f"{BASE}/stats/nfl/{season}"
            f"?season_type=regular&position[]={pos}&order_by={pts_key}"
        )
        try:
            rows = get_json(url, ttl_hours=CACHE_TTL, timeout=90)
        except FetchError:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            pid, stats = row.get("player_id"), (row.get("stats") or {})
            pts = stats.get(pts_key)
            if pid and isinstance(pts, (int, float)):
                out[str(pid)] = float(pts)
    return out


def build_historical_pool(
    season: int, league: LeagueConfig, cfg: dict
) -> tuple[list[Player], dict]:
    """The pool as a drafter could have seen it before `season` kicked off."""
    ecfg = cfg["engine"]
    scoring = league.scoring

    projections = fetch_season_projections(season, scoring)
    ffc_rows = adp_mod.fetch_adp(league.teams, str(season), scoring, CACHE_TTL)
    ffc_idx = adp_mod.build_adp_index(ffc_rows)

    # FFC is the only period-correct source for which team a player was on,
    # and therefore for his bye week.
    team_bye = {}
    for r in ffc_rows:
        if r.get("team") and r.get("bye"):
            team_bye.setdefault(r["team"], int(r["bye"]))

    back = REFERENCE_SEASON - season

    players: list[Player] = []
    matched = 0

    for pid, rec in projections.items():
        pos = rec["position"]
        if pos not in {"QB", "RB", "WR", "TE", "K", "DEF"}:
            continue

        row = adp_mod.lookup(ffc_idx, rec["name"], pos, None)
        if row is None and pos != "DEF":
            row = ffc_idx.get(adp_mod.name_key(rec["name"], pos))
        if row:
            matched += 1

        ffc_adp = float(row["adp"]) if row and row.get("adp") else None
        stdev = row.get("stdev") if row else None
        team = row.get("team") if row else None

        adp_val, spread = blend_adp(rec["adp"], ffc_adp, ecfg)
        if adp_val is None:
            continue  # undrafted in both archives -- nobody was taking him

        sigma = effective_sigma(stdev, ecfg["adp_sigma_floor"], ecfg["adp_sigma_inflate"])
        sigma += ecfg.get("adp_disagreement_sigma_weight", 0.0) * min(
            spread, float(ecfg.get("adp_disagreement_cap", 30.0))
        )

        exp_now = rec.get("years_exp_now")
        years_exp = max(0, int(exp_now) - back) if exp_now is not None else None

        players.append(
            Player(
                pid=pid,
                name=rec["name"],
                position=pos,
                team=team,
                raw_projection=rec["points"],
                # No injury discount: today's injury flags say nothing about a
                # season that finished years ago.
                projection=rec["points"],
                adp=adp_val,
                adp_sleeper=rec["adp"],
                adp_ffc=ffc_adp,
                adp_spread=spread,
                sigma=sigma,
                adp_stdev=float(stdev) if stdev else None,
                bye=team_bye.get(team) if team else None,
                years_exp=years_exp,
                depth_chart_order=None,  # no period-correct source
            )
        )

    levels = compute_replacement_levels(players, league)
    weights = ecfg.get("position_value_multiplier") or {}
    for p in players:
        p.replacement = levels.get(p.position, 0.0)
        p.vor = (p.projection - p.replacement) * weights.get(p.position, 1.0)

    apply_market_ranking(players, ecfg.get("market_ranked_positions"))

    waivers = waiver_levels(players, league, ecfg)
    for p in players:
        p.waiver_level = waivers.get(p.position, 0.0)
        p.sd = estimate_sd(
            p.projection, p.position, p.years_exp, None, p.adp, p.adp_stdev, ecfg
        )
        full = option_value(p.projection, p.sd, p.waiver_level)
        p.val = max(0.0, p.projection - p.waiver_level)
        p.upside = full - p.val

    players.sort(key=lambda x: x.vor, reverse=True)

    diagnostics = {
        "season": season,
        "players": len(players),
        "ffc_matched": matched,
        "ffc_rows": len(ffc_rows),
        "with_bye": sum(1 for p in players if p.bye),
        "replacement_levels": levels,
    }
    return players, diagnostics
