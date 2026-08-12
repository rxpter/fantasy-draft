"""Turns draft state into ranked advice.

The headline number is not "who is best" -- it is **cost of waiting**: how much
value you expect to lose at each position if you skip it this round. That is the
question a draft actually asks you, and it is why this beats reading a ranked
list off a page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .league import LeagueConfig
from .lineup import can_draft, optimal_lineup, remaining_starter_needs
from .simulate import build_context, simulate
from .survival import p_survives_to


@dataclass
class DraftState:
    picks: list = field(default_factory=list)      # raw Sleeper pick dicts
    taken_pids: set = field(default_factory=set)
    my_roster: list = field(default_factory=list)  # Player objects
    current_pick: int = 1
    my_slot: int | None = None
    my_future_picks: list = field(default_factory=list)
    on_the_clock: bool = False
    picks_until_mine: int = 0


def build_state(
    picks: list, players_by_pid: dict, league: LeagueConfig, my_slot: int | None
) -> DraftState:
    st = DraftState(picks=picks, my_slot=my_slot)

    for pk in picks:
        pid = pk.get("player_id")
        if pid:
            st.taken_pids.add(str(pid))

    st.current_pick = len(picks) + 1

    if my_slot:
        all_mine = league.my_pick_numbers(my_slot)
        st.my_future_picks = [p for p in all_mine if p >= st.current_pick]
        st.on_the_clock = bool(st.my_future_picks) and st.my_future_picks[0] == st.current_pick
        st.picks_until_mine = (
            st.my_future_picks[0] - st.current_pick if st.my_future_picks else 0
        )

        mine = {
            str(pk["player_id"])
            for pk in picks
            if pk.get("player_id") and pk.get("draft_slot") == my_slot
        }
        st.my_roster = [players_by_pid[p] for p in mine if p in players_by_pid]

    return st


def following_pick(st: DraftState) -> int | None:
    """The pick after the one you are about to make -- the 'wait until' pick."""
    if not st.my_future_picks:
        return None
    if st.on_the_clock:
        return st.my_future_picks[1] if len(st.my_future_picks) > 1 else None
    return st.my_future_picks[0]


def expected_best_at(group: list, target_pick: int, current_pick: int) -> float:
    """E[VOR of the best player at this position surviving to target_pick].

    Walks the position in VOR order; each player contributes his value times the
    chance he survives times the chance everyone better than him does not.
    """
    exp = 0.0
    all_better_gone = 1.0
    for p in group:
        surv = p_survives_to(p.adp, p.sigma, target_pick, current_pick)
        exp += p.vor * surv * all_better_gone
        all_better_gone *= 1.0 - surv
        if all_better_gone < 1e-4:
            break
    return exp


@dataclass
class PositionOutlook:
    position: str
    best_now: float
    expected_next: float
    cost_of_waiting: float
    expected_gone: float
    best_player: object | None


def position_outlook(
    available: list, st: DraftState, league: LeagueConfig
) -> list[PositionOutlook]:
    next_pick = following_pick(st)
    if next_pick is None:
        return []

    by_pos: dict[str, list] = {}
    for p in available:
        by_pos.setdefault(p.position, []).append(p)

    out = []
    for pos, group in by_pos.items():
        group.sort(key=lambda x: x.vor, reverse=True)
        if not group:
            continue
        best_now = group[0].vor
        exp_next = expected_best_at(group[:40], next_pick, st.current_pick)
        gone = sum(
            1.0 - p_survives_to(p.adp, p.sigma, next_pick, st.current_pick)
            for p in group[:60]
        )
        out.append(
            PositionOutlook(
                position=pos,
                best_now=best_now,
                expected_next=exp_next,
                cost_of_waiting=best_now - exp_next,
                expected_gone=gone,
                best_player=group[0],
            )
        )
    out.sort(key=lambda o: o.cost_of_waiting, reverse=True)
    return out


@dataclass
class Recommendation:
    player: object
    vor: float
    survive_next: float
    sim_score: float | None
    edge: float
    note: str


MIN_REALISTIC_SURVIVAL = 0.05


def choose_candidates(
    available: list, st: DraftState, league: LeagueConfig, ecfg: dict, limit: int
) -> list:
    """Top players by VOR, plus the best available at each unfilled slot.

    When you are not on the clock these are *targets*, not picks, so players
    with essentially no chance of lasting until your turn are dropped -- they
    are not decisions you get to make, and leaving them in buries the players
    you can actually have.
    """
    picks_left = league.rounds - len(st.my_roster)
    legal = [
        p
        for p in available
        if can_draft(p, st.my_roster, league, ecfg, picks_left)
    ]
    if not st.on_the_clock:
        realistic = [p for p in legal if p.survive_next >= MIN_REALISTIC_SURVIVAL]
        if realistic:
            legal = realistic
    legal.sort(key=lambda x: x.vor, reverse=True)

    # Cap how many of one position can crowd the board. Late in a draft a
    # single position often owns the whole VOR ordering -- a screen of twelve
    # tight ends when you can roster two is useless, and it hides the best
    # available flyer at every other position.
    per_pos_cap = ecfg.get("max_candidates_per_position", 4)
    chosen: list = []
    per_pos: dict[str, int] = {}
    for p in legal:
        if len(chosen) >= limit:
            break
        if per_pos.get(p.position, 0) >= per_pos_cap:
            continue
        chosen.append(p)
        per_pos[p.position] = per_pos.get(p.position, 0) + 1

    seen = {p.pid for p in chosen}

    needs = remaining_starter_needs(st.my_roster, league)
    for pos, need in needs.items():
        if need <= 0 or pos == "FLEX":
            continue
        for p in legal:
            if p.position == pos and p.pid not in seen:
                chosen.append(p)
                seen.add(p.pid)
                break
    return chosen


def recommend(
    players: list,
    st: DraftState,
    league: LeagueConfig,
    ecfg: dict,
    run_sim: bool = True,
) -> tuple[list[Recommendation], list[PositionOutlook]]:
    available = [p for p in players if p.pid not in st.taken_pids]

    next_pick = following_pick(st)

    for p in available:
        p.survive_next = (
            p_survives_to(p.adp, p.sigma, next_pick, st.current_pick) if next_pick else 1.0
        )

    outlook = position_outlook(available, st, league)
    candidates = choose_candidates(available, st, league, ecfg, ecfg["candidates"])
    if not candidates:
        return [], outlook

    sim_scores: dict[str, float] = {}
    if run_sim and st.my_future_picks:
        ctx, index = build_context(players, league, ecfg)
        avail_idx = [index[p.pid] for p in available if p.pid in index]
        roster_idx = [index[p.pid] for p in st.my_roster if p.pid in index]
        cand_idx = [index[p.pid] for p in candidates if p.pid in index]
        # The candidate always occupies your next pick, so it drops out of the
        # list of picks the simulation still has to make.
        sim_start = st.my_future_picks[0]
        future = st.my_future_picks[1:]

        raw = simulate(
            ctx=ctx,
            available=avail_idx,
            my_roster=roster_idx,
            my_future_picks=future,
            current_pick=st.current_pick,
            sim_start=sim_start,
            candidates=cand_idx,
            sims=ecfg["sims"],
            horizon=ecfg["sim_horizon_picks"],
        )
        sim_scores = {ctx.pid[i]: v for i, v in raw.items()}

    recs = []
    for p in candidates:
        recs.append(
            Recommendation(
                player=p,
                vor=p.vor,
                survive_next=p.survive_next,
                sim_score=sim_scores.get(p.pid),
                edge=0.0,
                note="",
            )
        )

    if sim_scores:
        recs.sort(key=lambda r: (r.sim_score if r.sim_score is not None else -1e9), reverse=True)
        best = recs[0].sim_score or 0.0
        runner_up = recs[1].sim_score if len(recs) > 1 else best
        for r in recs:
            r.edge = (r.sim_score or 0.0) - (runner_up if r is recs[0] else best)
    else:
        recs.sort(key=lambda r: r.vor, reverse=True)
        for r in recs:
            r.edge = r.vor - recs[0].vor

    for r in recs:
        r.note = _note(r, outlook, st)

    return recs, outlook


def _note(rec: Recommendation, outlook: list[PositionOutlook], st: DraftState) -> str:
    p = rec.player
    bits = []

    if rec.survive_next >= 0.85:
        bits.append(f"{rec.survive_next:.0%} to return")
    elif rec.survive_next <= 0.15:
        bits.append("gone if you wait")

    for o in outlook:
        if o.position == p.position:
            if o.cost_of_waiting >= 15:
                bits.append(f"{p.position} cliff (-{o.cost_of_waiting:.0f})")
            break

    if p.injury:
        bits.append(p.injury.lower())

    # Where the upside is coming from, so a volatile pick is legible rather
    # than mysterious.
    if p.depth_chart_order and p.depth_chart_order >= 2:
        bits.append("backup upside")
    elif p.years_exp is not None and p.years_exp <= 1:
        bits.append("rookie")

    if p.bye and st.my_roster:
        same = sum(1 for q in st.my_roster if q.bye == p.bye)
        if same >= 2:
            bits.append(f"bye {p.bye} stack x{same + 1}")

    return ", ".join(bits)
