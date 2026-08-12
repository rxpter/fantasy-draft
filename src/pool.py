"""Builds the unified player pool by joining Sleeper + FFC.

Join strategy: Sleeper's `player_id` is the spine -- it is the same key used by
the projections feed and the live pick feed, so those two joins are exact. Only
FFC needs name matching, and only for ADP/stdev/bye.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from . import adp as adp_mod
from . import sleeper
from .league import LeagueConfig, compute_replacement_levels
from .upside import estimate_sd, option_value, waiver_levels

DRAFTABLE = {"QB", "RB", "WR", "TE", "K", "DEF"}

OVERRIDE_PATH = Path(__file__).resolve().parent.parent / "data" / "my_projections.csv"


@dataclass
class Player:
    pid: str
    name: str
    position: str
    team: str | None
    projection: float = 0.0          # injury-adjusted season points
    raw_projection: float = 0.0
    adp: float = 999.0
    sigma: float = 30.0
    adp_stdev: float | None = None
    bye: int | None = None
    injury: str | None = None
    search_rank: int = 9999
    years_exp: int | None = None
    depth_chart_order: int | None = None
    vor: float = 0.0
    replacement: float = 0.0

    # Risk model. The option value of rostering a player splits cleanly in two,
    # and keeping them apart is what makes the late rounds readable:
    #   val    -- intrinsic: points he beats a streamer by, if he hits his
    #             projection exactly. Dominates for established starters.
    #   upside -- time value: what his *uncertainty* alone is worth. This is
    #             the boom component, and it is nearly all of a late flyer's
    #             worth. val + upside = the full option value.
    sd: float = 0.0
    val: float = 0.0
    upside: float = 0.0
    waiver_level: float = 0.0

    # Filled in per-refresh by the recommender.
    survive_next: float = field(default=0.0, compare=False)
    survive_after: float = field(default=0.0, compare=False)

    def label(self) -> str:
        tag = f" ({self.injury})" if self.injury else ""
        return f"{self.name}{tag}"


def _display_name(pid: str, rec: dict) -> str:
    if rec.get("position") == "DEF":
        return f"{rec.get('team') or pid} D/ST"
    full = rec.get("full_name")
    if full:
        return full
    parts = [rec.get("first_name"), rec.get("last_name")]
    return " ".join(p for p in parts if p) or pid


def build_pool(cfg: dict, league: LeagueConfig, season: str) -> tuple[list[Player], dict]:
    """Returns (players sorted by VOR desc, diagnostics dict)."""
    ecfg = cfg["engine"]
    ccfg = cfg["cache"]

    raw_players = sleeper.get_players(ttl_hours=ccfg["players_ttl_hours"])
    projections = sleeper.get_projections(
        season, league.scoring, ttl_hours=ccfg["projections_ttl_hours"]
    )
    adp_rows = adp_mod.fetch_adp(league.teams, season, league.scoring, ccfg["adp_ttl_hours"])
    adp_idx = adp_mod.build_adp_index(adp_rows)

    # Bye week is a team property; deriving it from FFC covers every player on
    # that team, including ones with no ADP entry of their own.
    team_bye: dict[str, int] = {}
    for r in adp_rows:
        if r.get("team") and r.get("bye"):
            team_bye.setdefault(r["team"], int(r["bye"]))

    inj_mult = ecfg["injury_multiplier"]
    players: list[Player] = []
    matched = 0

    for pid, rec in raw_players.items():
        pos = rec.get("position")
        if pos not in DRAFTABLE:
            continue
        if pos != "DEF" and not rec.get("team"):
            continue  # free agents / practice squad noise

        name = _display_name(pid, rec)
        team = rec.get("team")
        proj = float(projections.get(str(pid), 0.0))

        row = adp_mod.lookup(adp_idx, name, pos, team)
        if row:
            matched += 1
            adp_val = float(row.get("adp", ecfg["undrafted_adp"]))
            stdev = row.get("stdev")
        else:
            adp_val = float(ecfg["undrafted_adp"])
            stdev = None

        if proj <= 0 and not row:
            continue  # neither projected nor drafted -- irrelevant

        from .survival import effective_sigma

        sigma = (
            effective_sigma(stdev, ecfg["adp_sigma_floor"], ecfg["adp_sigma_inflate"])
            if row
            else float(ecfg["undrafted_sigma"])
        )

        injury = rec.get("injury_status") or None
        mult = inj_mult.get(injury, 1.0) if injury else 1.0

        players.append(
            Player(
                pid=str(pid),
                name=name,
                position=pos,
                team=team,
                raw_projection=proj,
                projection=proj * mult,
                adp=adp_val,
                sigma=sigma,
                adp_stdev=float(stdev) if stdev else None,
                bye=team_bye.get(team) if team else None,
                injury=injury,
                search_rank=int(rec.get("search_rank") or 9999),
                years_exp=rec.get("years_exp"),
                depth_chart_order=rec.get("depth_chart_order"),
            )
        )

    overridden = apply_projection_override(players)

    real_projections = sum(1 for p in players if p.raw_projection > 0)

    # Players with an ADP but no projection would otherwise look free. Give them
    # a floor estimate from positional ADP neighbours so they cannot be
    # recommended purely because the projections feed missed them.
    _backfill_projections(players)

    levels = compute_replacement_levels(players, league)
    # An explicit, opt-in positional lean. Defaults to 1.0 everywhere, i.e. no
    # effect -- it exists so a manager can encode a view the projections do not
    # share, not because the engine needs correcting.
    weights = ecfg.get("position_value_multiplier") or {}
    for p in players:
        p.replacement = levels.get(p.position, 0.0)
        p.vor = (p.projection - p.replacement) * weights.get(p.position, 1.0)

    # Risk model. Waiver level is what you could stream at this position after
    # the draft; upside is the option value of holding the player over it.
    waivers = waiver_levels(players, league, ecfg)
    for p in players:
        p.waiver_level = waivers.get(p.position, 0.0)
        p.sd = estimate_sd(
            p.projection, p.position, p.years_exp, p.depth_chart_order,
            p.adp, p.adp_stdev, ecfg,
        )
        full = option_value(p.projection, p.sd, p.waiver_level)
        p.val = max(0.0, p.projection - p.waiver_level)
        p.upside = full - p.val

    players.sort(key=lambda x: x.vor, reverse=True)

    diagnostics = {
        "players_considered": len(players),
        "with_projection": real_projections,
        "backfilled": len(players) - real_projections,
        "adp_rows": len(adp_rows),
        "adp_matched": matched,
        "replacement_levels": levels,
        "team_byes": len(team_bye),
        "overridden": overridden,
    }
    return players, diagnostics


def apply_projection_override(players: list[Player]) -> int:
    """Replace projections from `data/my_projections.csv` if it exists.

    This is the supported way to bring your own football analysis -- O-line
    quality, coaching scheme, strength of schedule, whatever you trust more than
    a consensus number. The scarcity engine does not care where points come
    from, so better inputs here improve every recommendation downstream.

    Columns: name, position, points   (a `player_id` column wins if present.)
    """
    if not OVERRIDE_PATH.exists():
        return 0

    by_pid = {p.pid: p for p in players}
    by_name: dict[tuple, Player] = {}
    for p in players:
        key = adp_mod.name_key(p.name, p.position)
        if key:
            by_name.setdefault(key, p)

    changed = 0
    try:
        with OVERRIDE_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                low = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
                raw = low.get("points") or low.get("proj") or low.get("projection")
                if not raw:
                    continue
                try:
                    pts = float(raw)
                except ValueError:
                    continue

                target = None
                if low.get("player_id"):
                    target = by_pid.get(low["player_id"])
                if target is None and low.get("name"):
                    pos = (low.get("position") or "").upper()
                    target = by_name.get(adp_mod.name_key(low["name"], pos))
                if target is None:
                    continue

                target.raw_projection = pts
                target.projection = pts
                changed += 1
    except (OSError, csv.Error):
        return 0
    return changed


def _backfill_projections(players: list[Player]) -> None:
    """Estimate a projection for ADP'd players the projections feed skipped."""
    by_pos: dict[str, list[Player]] = {}
    for p in players:
        by_pos.setdefault(p.position, []).append(p)

    for pos, group in by_pos.items():
        known = sorted(
            (p for p in group if p.raw_projection > 0), key=lambda x: x.adp
        )
        if not known:
            continue
        for p in group:
            if p.raw_projection > 0:
                continue
            # Nearest projected player by ADP, discounted -- deliberately
            # pessimistic so a data gap never manufactures a sleeper.
            nearest = min(known, key=lambda k: abs(k.adp - p.adp))
            p.raw_projection = nearest.raw_projection * 0.85
            p.projection = p.raw_projection
