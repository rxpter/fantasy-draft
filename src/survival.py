"""Availability ("will he make it back to me?") probability.

This is the piece that turns a ranking into a draft strategy. Taking the best
player available is only correct if the alternative would not have survived to
your next pick. Everything here is conditioned on the draft state you can
actually observe: a player with ADP 20 who is still on the board at pick 40 is
telling you something, and the conditional form below uses it.
"""

from __future__ import annotations

import math

SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def p_survives_to(adp: float, sigma: float, target_pick: int, current_pick: int) -> float:
    """P(player is still on the board AT target_pick | still here at current_pick).

    Uses a continuity correction of half a pick, and conditions on the player
    having already survived to the current pick.
    """
    sigma = max(sigma, 0.5)
    if target_pick <= current_pick:
        return 1.0

    p_now = 1.0 - norm_cdf((current_pick - 0.5 - adp) / sigma)
    p_then = 1.0 - norm_cdf((target_pick - 0.5 - adp) / sigma)

    if p_now <= 1e-9:
        # Already far past his ADP -- the unconditional model has essentially
        # ruled this out, so fall back to a flat, mildly optimistic estimate
        # rather than dividing by ~zero.
        return max(0.0, min(1.0, p_then / max(p_now, 1e-3)))
    return max(0.0, min(1.0, p_then / p_now))


def effective_sigma(stdev: float | None, floor: float, inflate: float) -> float:
    """FFC's sample stdev understates true single-draft variance.

    It is pooled across thousands of drafts, so it measures how stable the
    consensus is, not how erratic your particular leaguemates are. Inflate it.
    """
    base = stdev if stdev and stdev > 0 else floor
    return max(base, floor) * inflate
