"""League settings, snake-draft pick arithmetic, and replacement levels.

Replacement level is the single most important number in the whole system: it
converts a raw projection into "points above what you could have had for free",
which is what actually decides a pick. In a 14-team league replacement level is
much lower than in the 10- and 12-team leagues most published rankings assume,
which is why generic cheat sheets misprice the top of the board here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SKILL = ("RB", "WR", "TE")


@dataclass
class LeagueConfig:
    teams: int = 14
    scoring: str = "ppr"
    starters: dict = field(
        default_factory=lambda: {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "K": 1, "DEF": 1}
    )
    flex_positions: tuple = SKILL
    bench: int = 5

    @classmethod
    def from_dict(cls, d: dict) -> "LeagueConfig":
        return cls(
            teams=int(d.get("teams", 14)),
            scoring=d.get("scoring", "ppr"),
            starters=dict(d.get("starters", {})),
            flex_positions=tuple(d.get("flex_positions", SKILL)),
            bench=int(d.get("bench", 5)),
        )

    @property
    def starter_slots(self) -> int:
        return sum(self.starters.values())

    @property
    def rounds(self) -> int:
        return self.starter_slots + self.bench

    @property
    def total_picks(self) -> int:
        return self.rounds * self.teams

    def pick_number(self, rnd: int, slot: int) -> int:
        """1-indexed overall pick for a snake draft (round and slot both 1-indexed)."""
        if rnd % 2 == 1:
            return (rnd - 1) * self.teams + slot
        return (rnd - 1) * self.teams + (self.teams - slot + 1)

    def my_pick_numbers(self, slot: int) -> list[int]:
        return [self.pick_number(r, slot) for r in range(1, self.rounds + 1)]

    def slot_from_pick(self, pick_no: int) -> int:
        """Which draft slot owns a given overall pick."""
        rnd = (pick_no - 1) // self.teams + 1
        idx = (pick_no - 1) % self.teams + 1
        return idx if rnd % 2 == 1 else self.teams - idx + 1


SLOT_KEYS = {
    "slots_qb": "QB",
    "slots_rb": "RB",
    "slots_wr": "WR",
    "slots_te": "TE",
    "slots_flex": "FLEX",
    "slots_k": "K",
    "slots_def": "DEF",
}

# Slot types this engine does not model. Superflex in particular would change
# QB replacement level enormously, so it is refused rather than approximated.
UNSUPPORTED_SLOTS = {
    "slots_super_flex": "superflex (QB-eligible flex)",
    "slots_idp_flex": "IDP flex",
    "slots_dl": "defensive line",
    "slots_lb": "linebacker",
    "slots_db": "defensive back",
    "slots_wrrb_flex": "WR/RB-only flex",
    "slots_rec_flex": "WR/TE-only flex",
}

SCORING_ALIASES = {"ppr": "ppr", "half_ppr": "half_ppr", "std": "std", "standard": "std"}


def league_from_draft(
    draft: dict, fallback: LeagueConfig
) -> tuple[LeagueConfig, list[str], list[str]]:
    """Derive league settings from a Sleeper draft object.

    Mock drafts rarely match your home league, and team count drives
    replacement level, so reading the real shape off the draft beats trusting
    config.json. Returns (config, notes describing changes, unsupported slots).
    """
    settings = draft.get("settings") or {}
    metadata = draft.get("metadata") or {}
    notes: list[str] = []
    unsupported: list[str] = []

    for key, label in UNSUPPORTED_SLOTS.items():
        if int(settings.get(key) or 0) > 0:
            unsupported.append(f"{label} x{settings[key]}")

    starters: dict[str, int] = {}
    for key, pos in SLOT_KEYS.items():
        if key in settings:
            count = int(settings.get(key) or 0)
            if count > 0:
                starters[pos] = count

    teams = int(settings.get("teams") or fallback.teams)
    bench = int(settings.get("slots_bn") or fallback.bench)
    if not starters:
        starters = dict(fallback.starters)

    scoring = fallback.scoring
    raw_scoring = (metadata.get("scoring_type") or "").lower()
    if raw_scoring in SCORING_ALIASES:
        scoring = SCORING_ALIASES[raw_scoring]

    derived = LeagueConfig(
        teams=teams,
        scoring=scoring,
        starters=starters,
        flex_positions=fallback.flex_positions,
        bench=bench,
    )

    # Sleeper's `rounds` is authoritative; trust it over starters+bench.
    rounds = int(settings.get("rounds") or 0)
    if rounds and rounds != derived.rounds:
        derived.bench = max(0, rounds - derived.starter_slots)
        notes.append(f"bench adjusted to {derived.bench} to match {rounds} rounds")

    if teams != fallback.teams:
        notes.append(f"teams {fallback.teams} -> {teams}")
    if scoring != fallback.scoring:
        notes.append(f"scoring {fallback.scoring} -> {scoring}")
    if starters != fallback.starters:
        notes.append(f"starters {fallback.starters} -> {starters}")

    return derived, notes, unsupported


def compute_replacement_levels(players, league: LeagueConfig) -> dict[str, float]:
    """Points scored by the first player at each position who will NOT start.

    Flex slots are allocated endogenously: rather than assuming a fixed
    RB/WR/TE flex split, we pool everyone past their positional baseline and
    let the projections decide who actually fills the league's flex spots.
    """
    by_pos: dict[str, list] = {}
    for p in players:
        if p.projection > 0:
            by_pos.setdefault(p.position, []).append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: x.projection, reverse=True)

    base = {
        pos: league.teams * cnt
        for pos, cnt in league.starters.items()
        if pos != "FLEX"
    }

    # Pool the leftovers at flex-eligible positions and take the best N.
    flex_slots = league.teams * league.starters.get("FLEX", 0)
    leftovers = []
    for pos in league.flex_positions:
        leftovers.extend(by_pos.get(pos, [])[base.get(pos, 0):])
    leftovers.sort(key=lambda x: x.projection, reverse=True)

    flex_counts: dict[str, int] = {}
    for p in leftovers[:flex_slots]:
        flex_counts[p.position] = flex_counts.get(p.position, 0) + 1

    levels: dict[str, float] = {}
    for pos, ranked in by_pos.items():
        need = base.get(pos, 0) + flex_counts.get(pos, 0)
        if not ranked:
            levels[pos] = 0.0
        elif need < len(ranked):
            levels[pos] = ranked[need].projection
        else:
            levels[pos] = ranked[-1].projection
    return levels


def flex_allocation(players, league: LeagueConfig) -> dict[str, int]:
    """How the league's flex spots actually break down -- useful for display."""
    by_pos: dict[str, list] = {}
    for p in players:
        if p.projection > 0 and p.position in league.flex_positions:
            by_pos.setdefault(p.position, []).append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: x.projection, reverse=True)

    leftovers = []
    for pos in league.flex_positions:
        leftovers.extend(by_pos.get(pos, [])[league.teams * league.starters.get(pos, 0):])
    leftovers.sort(key=lambda x: x.projection, reverse=True)

    counts: dict[str, int] = {}
    for p in leftovers[: league.teams * league.starters.get("FLEX", 0)]:
        counts[p.position] = counts.get(p.position, 0) + 1
    return counts
