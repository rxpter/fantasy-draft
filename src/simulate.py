"""Monte Carlo simulation of the rest of the draft.

For each candidate you could take right now, we simulate the draft forward many
times and score the starting lineup you end up with. The pick that wins is the
one that maximises your *final roster*, not the one with the biggest name --
which is how the engine discovers things like "take the TE now, the RB tier
survives another 22 picks".

Two implementation notes that matter:

* Common random numbers -- every candidate is evaluated against the *same*
  sampled draft order within a sim. That cancels most of the noise between
  candidates, so 200 sims here discriminate better than 1000 independent ones.
* Everything is index-based over flat lists. Attribute lookups on dataclasses
  dominate the runtime otherwise, and this has to finish inside a draft clock.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .league import LeagueConfig
from .upside import option_value

POS_CODES = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "K": 4, "DEF": 5}
CODE_POS = {v: k for k, v in POS_CODES.items()}


@dataclass
class SimContext:
    """Flat arrays describing every draftable player, indexed together."""

    proj: list
    vor: list
    pos: list
    bye: list
    adp: list
    sigma: list
    pid: list
    sd: list                # season-projection standard deviation per player

    # league shape, precomputed into codes
    starters: dict          # pos code -> count
    flex_slots: int
    flex_codes: tuple
    rounds: int
    caps: dict              # pos code -> max on roster
    late_only: dict         # pos code -> only draft when picks_left <= n
    bye_allowed: int
    bye_penalty: float
    upside_weight: float
    waiver: dict            # pos code -> streamable points
    miss: dict              # pos code -> fraction of season starter is unavailable
    targets: dict           # pos code -> desired roster count
    shortfall: dict         # pos code -> penalty per player below target


def build_context(players, league: LeagueConfig, ecfg: dict) -> tuple[SimContext, dict]:
    """Returns (context, pid -> index)."""
    proj, vor, pos, bye, adp, sigma, pid, sd = [], [], [], [], [], [], [], []
    index: dict[str, int] = {}

    for i, p in enumerate(players):
        index[p.pid] = i
        proj.append(p.projection)
        vor.append(p.vor)
        pos.append(POS_CODES.get(p.position, 0))
        bye.append(p.bye or 0)
        adp.append(p.adp)
        sigma.append(p.sigma)
        pid.append(p.pid)
        sd.append(p.sd)

    starters = {
        POS_CODES[k]: v for k, v in league.starters.items() if k != "FLEX" and k in POS_CODES
    }
    caps = {POS_CODES[k]: v for k, v in ecfg["max_at_position"].items() if k in POS_CODES}
    late = {POS_CODES[k]: v for k, v in ecfg["late_round_only"].items() if k in POS_CODES}

    waiver_pts = {}
    for p in players:
        code = POS_CODES.get(p.position, 0)
        waiver_pts.setdefault(code, p.waiver_level)

    ctx = SimContext(
        proj=proj,
        vor=vor,
        pos=pos,
        bye=bye,
        adp=adp,
        sigma=sigma,
        pid=pid,
        sd=sd,
        starters=starters,
        flex_slots=league.starters.get("FLEX", 0),
        flex_codes=tuple(POS_CODES[p] for p in league.flex_positions),
        rounds=league.rounds,
        caps=caps,
        late_only=late,
        bye_allowed=ecfg["bye_starters_allowed_per_week"],
        bye_penalty=ecfg["bye_penalty_per_extra_starter"],
        upside_weight=ecfg["upside_weight"],
        waiver=waiver_pts,
        miss={POS_CODES[k]: v for k, v in ecfg["starter_miss_rate"].items() if k in POS_CODES},
        # Only chase depth at positions this league actually starts. Plenty of
        # Sleeper mocks drop the kicker or defense slot, and without this guard
        # the engine would be penalised for not drafting one.
        targets={
            POS_CODES[k]: v
            for k, v in ecfg["roster_targets"].items()
            if k in POS_CODES
            and (league.starters.get(k, 0) > 0 or k in league.flex_positions)
        },
        shortfall={
            POS_CODES[k]: v for k, v in ecfg["target_shortfall_penalty"].items() if k in POS_CODES
        },
    )
    return ctx, index


def score_roster_fast(roster: list[int], ctx: SimContext) -> float:
    """Risk-adjusted value of a finished roster.

    Three things beyond raw starting points, each fixing a way the naive
    objective misreads a 14-team draft:

    1. **Starters are blended with their actual fallback.** A starter misses
       part of most seasons; what you score in those weeks depends on whether
       you drafted a backup or are streaming off waivers. That gap is far wider
       at running back (waiver RB ~68 pts) than at receiver (~163), which is the
       real, quantified reason to hold running back depth.
    2. **Bench players are valued as options**, not point estimates -- upside is
       worth paying for once a player is not in your lineup anyway.
    3. **Roster targets.** Finishing with three running backs is fragile no
       matter how the points add up, so falling short of a target costs you.
    """
    proj, pos = ctx.proj, ctx.pos

    buckets: dict[int, list[int]] = {}
    for i in roster:
        buckets.setdefault(pos[i], []).append(i)
    for code in buckets:
        buckets[code].sort(key=lambda i: proj[i], reverse=True)

    started: list[int] = []
    for code, count in ctx.starters.items():
        started.extend(buckets.get(code, [])[:count])

    if ctx.flex_slots:
        leftovers = []
        for code in ctx.flex_codes:
            leftovers.extend(buckets.get(code, [])[ctx.starters.get(code, 0):])
        leftovers.sort(key=lambda i: proj[i], reverse=True)
        started.extend(leftovers[: ctx.flex_slots])

    started_set = set(started)

    # Best player at each position who is NOT starting -- your injury/bye cover.
    fallback: dict[int, float] = {}
    for code, group in buckets.items():
        for i in group:
            if i not in started_set:
                fallback[code] = proj[i]
                break

    total = 0.0
    for i in started:
        code = pos[i]
        pm = ctx.miss.get(code, 0.0)
        cover = fallback.get(code, ctx.waiver.get(code, 0.0))
        total += proj[i] * (1.0 - pm) + pm * cover

    bye_counts: dict[int, int] = {}
    for i in started:
        b = ctx.bye[i]
        if b:
            bye_counts[b] = bye_counts.get(b, 0) + 1
    penalty = 0.0
    for n in bye_counts.values():
        if n > ctx.bye_allowed:
            penalty += (n - ctx.bye_allowed) * ctx.bye_penalty

    # Bench upside. The strike price is the starter he would have to beat to
    # get into your lineup -- NOT the waiver wire. A backup quarterback in a
    # 1-QB league scores nothing behind a healthy starter, however good he is
    # in the abstract; pricing him against waivers made him look like a steal
    # and burned a bench spot every draft. His injury-cover value is already
    # counted in the fallback blend above, so this is not double counting.
    weakest: dict[int, float] = {}
    weakest_flex = None
    for i in started:
        code = pos[i]
        if code not in weakest or proj[i] < weakest[code]:
            weakest[code] = proj[i]
        if code in ctx.flex_codes and (weakest_flex is None or proj[i] < weakest_flex):
            weakest_flex = proj[i]

    bench = 0.0
    for i in roster:
        if i in started_set:
            continue
        code = pos[i]
        strike = weakest.get(code, ctx.waiver.get(code, 0.0))
        if code in ctx.flex_codes and weakest_flex is not None and weakest_flex < strike:
            strike = weakest_flex
        bench += option_value(proj[i], ctx.sd[i], strike)

    shortfall = 0.0
    for code, want in ctx.targets.items():
        have = len(buckets.get(code, ()))
        if have < want:
            shortfall += (want - have) * ctx.shortfall.get(code, 0.0)

    return total - penalty + bench * ctx.upside_weight - shortfall


def unfilled_required(counts: dict[int, int], ctx: SimContext) -> list[int]:
    """Required starting slots (not flex) with nobody in them yet."""
    return [code for code, need in ctx.starters.items() if counts.get(code, 0) < need]


def _greedy_pick(
    vor_order: list[int], taken: set[int], counts: dict[int, int], ctx: SimContext, picks_left: int
) -> int | None:
    """Best legal player by VOR, respecting caps and the late-round K/DEF rule.

    VOR alone will happily end a draft with no kicker: the best kicker is only
    ~16 points above replacement, so he loses every comparison to a bench WR.
    But an *unfilled* starting slot scores zero, which is ~100 points worse. So
    once you have exactly as many picks left as empty required slots, the
    remaining picks are forced to fill them.
    """
    must = unfilled_required(counts, ctx)
    restrict = set(must) if must and picks_left <= len(must) else None

    for i in vor_order:
        if i in taken:
            continue
        code = ctx.pos[i]
        cap = ctx.caps.get(code)
        if cap is not None and counts.get(code, 0) >= cap:
            continue
        if restrict is not None:
            # Must-fill overrides the "don't draft K/DEF early" rule.
            if code not in restrict:
                continue
            return i
        late = ctx.late_only.get(code)
        if late is not None and picks_left > late:
            continue
        return i
    return None


def _forced_fill(
    vor_order: list[int], taken: set[int], counts: dict[int, int], ctx: SimContext, picks_left: int
) -> int | None:
    """Relaxed fallback so a roster always completes (ignores the late rule)."""
    for i in vor_order:
        if i in taken:
            continue
        code = ctx.pos[i]
        cap = ctx.caps.get(code)
        if cap is not None and counts.get(code, 0) >= cap:
            continue
        return i
    return None


def simulate(
    ctx: SimContext,
    available: list[int],
    my_roster: list[int],
    my_future_picks: list[int],
    current_pick: int,
    sim_start: int,
    candidates: list[int],
    sims: int,
    horizon: int,
    seed: int = 12345,
) -> dict[int, float]:
    """Average final roster score for each candidate. Returns {index: score}.

    `sim_start` is the pick at which the candidate is taken -- your next pick,
    which may be several picks away. Opponents consume the gap between
    `current_pick` and `sim_start` first, so a candidate is never handed to you
    for free on top of your real picks. `my_future_picks` must therefore exclude
    `sim_start` itself.
    """
    if not candidates:
        return {}

    rng = random.Random(seed)
    gauss = rng.gauss

    adp, sigma = ctx.adp, ctx.sigma
    vor = ctx.vor

    # Only players who could plausibly still go matter for the simulation.
    horizon_picks = my_future_picks[:horizon]
    last_pick = horizon_picks[-1] if horizon_picks else sim_start
    depth = max(60, (last_pick - current_pick) + 80)
    pre_gap = max(0, sim_start - current_pick)

    universe = sorted(available, key=lambda i: adp[i])[:depth]
    universe_set = set(universe)
    # Candidates must be simulable even if their ADP is deep.
    for c in candidates:
        if c not in universe_set:
            universe.append(c)
            universe_set.add(c)

    vor_order = sorted(universe, key=lambda i: vor[i], reverse=True)

    base_counts: dict[int, int] = {}
    for i in my_roster:
        code = ctx.pos[i]
        base_counts[code] = base_counts.get(code, 0) + 1

    totals = {c: 0.0 for c in candidates}

    for _ in range(sims):
        # One sampled draft order, shared by every candidate this round.
        sampled = sorted(universe, key=lambda i: adp[i] + sigma[i] * gauss(0.0, 1.0))

        for cand in candidates:
            taken = {cand}
            counts = dict(base_counts)
            counts[ctx.pos[cand]] = counts.get(ctx.pos[cand], 0) + 1
            roster = my_roster + [cand]

            ptr = 0
            # Opponents pick between now and the pick where you take the
            # candidate. The candidate is already in `taken`, so nobody else
            # can grab him -- we are conditioning on him lasting to you.
            consumed = 0
            while consumed < pre_gap and ptr < len(sampled):
                i = sampled[ptr]
                ptr += 1
                if i not in taken:
                    taken.add(i)
                    consumed += 1

            prev_pick = sim_start

            for my_pick in horizon_picks:
                # Opponents consume the sampled order between my picks.
                gap = my_pick - prev_pick - 1
                consumed = 0
                while consumed < gap and ptr < len(sampled):
                    i = sampled[ptr]
                    ptr += 1
                    if i not in taken:
                        taken.add(i)
                        consumed += 1

                picks_left = ctx.rounds - len(roster)
                choice = _greedy_pick(vor_order, taken, counts, ctx, picks_left)
                if choice is None:
                    choice = _forced_fill(vor_order, taken, counts, ctx, picks_left)
                if choice is None:
                    break
                taken.add(choice)
                counts[ctx.pos[choice]] = counts.get(ctx.pos[choice], 0) + 1
                roster.append(choice)
                prev_pick = my_pick

            # Beyond the simulated horizon, fill deterministically. The tail of
            # a draft is low-leverage, so spending sims on it buys nothing.
            while len(roster) < ctx.rounds:
                picks_left = ctx.rounds - len(roster)
                choice = _greedy_pick(vor_order, taken, counts, ctx, picks_left)
                if choice is None:
                    choice = _forced_fill(vor_order, taken, counts, ctx, picks_left)
                if choice is None:
                    break
                taken.add(choice)
                counts[ctx.pos[choice]] = counts.get(ctx.pos[choice], 0) + 1
                roster.append(choice)

            totals[cand] += score_roster_fast(roster, ctx)

    return {c: totals[c] / sims for c in candidates}
