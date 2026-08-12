"""Projection uncertainty and the option value of a bench spot.

Everything else in this engine works on expected points, which is right for a
locked-in starter and wrong for the back half of a draft. A bench player's
downside is capped -- you drop him for nothing -- while his upside is not. So
his real worth is not his projection, it is

    E[max(0, points - what you could stream off waivers)]

which is a call option, and options are worth *more* when the outcome is
uncertain. That single change is what makes the engine prefer the volatile
rookie over the safe veteran with the same projection in round 12, which is
where leagues are actually won.
"""

from __future__ import annotations

import math

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def option_value(mu: float, sd: float, strike: float) -> float:
    """E[max(0, X - strike)] for X ~ Normal(mu, sd). Closed form (Bachelier).

    At sd = 0 this collapses to plain max(0, mu - strike), so a player with no
    uncertainty is valued exactly as the old point-estimate model valued him.
    """
    if sd <= 1e-9:
        return max(0.0, mu - strike)
    d = (mu - strike) / sd
    return (mu - strike) * _cdf(d) + sd * _pdf(d)


def estimate_sd(
    projection: float,
    position: str,
    years_exp: int | None,
    depth_chart_order: int | None,
    adp: float,
    adp_stdev: float | None,
    cfg: dict,
) -> float:
    """Season-projection standard deviation for one player.

    Built from four signals, all of which we actually have:
      * position   -- running backs are volatile, kickers are not
      * experience -- rookies and second-year players have the widest outcomes
      * depth      -- a backup is a bimodal bet on the guy ahead of him
      * the market -- ADP disagreement is a direct read on uncertainty
    """
    cv = cfg["projection_cv"].get(position, 0.25)
    sd = max(projection * cv, cfg["projection_sd_floor"])

    boost = 1.0
    if years_exp is not None and years_exp <= 1:
        boost += cfg["rookie_upside_boost"]
    if depth_chart_order is not None and depth_chart_order >= 2:
        boost += cfg["backup_upside_boost"]

    # Relative ADP spread: managers disagreeing about a player is information.
    if adp_stdev and adp > 0:
        boost += cfg["market_disagreement_weight"] * min(adp_stdev / adp, 1.0)

    return sd * boost


def waiver_levels(players, league, cfg) -> dict[str, float]:
    """What you can realistically stream at each position once the draft ends.

    Deeper than draft-day replacement: it is the best player at that position
    who goes *undrafted*. That gap is the whole argument for running back
    depth -- an unowned RB is worth ~half an unowned WR in this format.
    """
    by_pos: dict[str, list] = {}
    for p in players:
        by_pos.setdefault(p.position, []).append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: x.projection, reverse=True)

    levels: dict[str, float] = {}
    for pos, ranked in by_pos.items():
        starters = league.teams * league.starters.get(pos, 0)
        if pos in league.flex_positions:
            starters += league.teams * league.starters.get("FLEX", 0) // len(
                league.flex_positions
            )
        # One more full round of that position gets rostered as bench, and the
        # waiver wire starts after that.
        rank = starters + league.teams * cfg["waiver_depth_extra_rounds"]
        if not ranked:
            levels[pos] = 0.0
        elif rank < len(ranked):
            levels[pos] = ranked[rank].projection
        else:
            levels[pos] = ranked[-1].projection
    return levels
