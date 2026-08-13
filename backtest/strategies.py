"""The drafters under test.

All three obey the same roster-legality rules (position caps, K/DEF held to the
end, must-fill when picks run out). That is deliberate and it is the
conservative choice: it makes the ADP baseline a competent drafter rather than
a strawman that takes three quarterbacks, so any edge the engine shows has to
come from its actual ranking rather than from the baseline being foolish.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lineup import can_draft  # noqa: E402
from src.recommend import recommend  # noqa: E402


def _legal(players, st, league, ecfg):
    picks_left = league.rounds - len(st.my_roster)
    return [
        p
        for p in players
        if p.pid not in st.taken_pids
        and can_draft(p, st.my_roster, league, ecfg, picks_left)
    ]


def adp_pick(players, st, league, ecfg):
    """Take the highest-ranked player left. What most managers actually do."""
    legal = _legal(players, st, league, ecfg)
    return min(legal, key=lambda p: p.adp) if legal else None


def vor_pick(players, st, league, ecfg):
    """Best value over replacement, with no scarcity or simulation.

    The interesting middle baseline: the gap between this and the full engine
    is what the Monte Carlo and the risk model are actually buying.
    """
    legal = _legal(players, st, league, ecfg)
    return max(legal, key=lambda p: p.vor) if legal else None


def engine_pick(players, st, league, ecfg):
    """The full engine: scarcity, survival, option value, roster targets."""
    recs, _ = recommend(players, st, league, ecfg, run_sim=True)
    return recs[0].player if recs else None


STRATEGIES = {
    "adp": adp_pick,
    "vor": vor_pick,
    "engine": engine_pick,
}
