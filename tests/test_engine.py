"""Engine tests. Run: python -m unittest discover -s tests -v

No network access -- everything here works on synthetic players so it stays
runnable on draft day when you do not want surprises.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.league import (
    LeagueConfig,
    compute_replacement_levels,
    flex_allocation,
    league_from_draft,
)
from src.lineup import (
    bye_conflicts,
    can_draft,
    optimal_lineup,
    remaining_starter_needs,
    unfilled_required,
)
from src.pool import Player, blend_adp
from src.recommend import expected_best_at
from src.simulate import build_context, score_roster_fast, simulate
from src.survival import p_survives_to
from src.upside import estimate_sd, option_value

ECFG = {
    "max_at_position": {"QB": 2, "TE": 2, "K": 1, "DEF": 1, "RB": 8, "WR": 8},
    "late_round_only": {"K": 2, "DEF": 3},
    "bye_starters_allowed_per_week": 2,
    "bye_penalty_per_extra_starter": 6.0,
    "roster_targets": {"RB": 5, "WR": 5, "TE": 1, "QB": 1, "K": 1, "DEF": 1},
    "target_shortfall_penalty": {"RB": 22.0, "WR": 9.0, "TE": 6.0, "QB": 5.0, "K": 0.0, "DEF": 0.0},
    "starter_miss_rate": {"RB": 0.22, "WR": 0.15, "TE": 0.15, "QB": 0.10, "K": 0.05, "DEF": 0.05},
    "waiver_depth_extra_rounds": 1,
    "upside_weight": 0.40,
    "projection_cv": {"QB": 0.20, "RB": 0.32, "WR": 0.28, "TE": 0.30, "K": 0.15, "DEF": 0.18},
    "projection_sd_floor": 12.0,
    "rookie_upside_boost": 0.35,
    "backup_upside_boost": 0.45,
    "market_disagreement_weight": 0.5,
    "sims": 40,
    "sim_horizon_picks": 4,
    "candidates": 8,
}


def mk(pid, pos, proj, adp=50.0, bye=5, sigma=8.0, sd=0.0, waiver=0.0):
    p = Player(pid=str(pid), name=f"P{pid}", position=pos, team="XX",
               projection=proj, raw_projection=proj, adp=adp, sigma=sigma, bye=bye)
    p.sd = sd
    p.waiver_level = waiver
    p.val = max(0.0, proj - waiver)
    p.upside = option_value(proj, sd, waiver) - p.val
    return p


def make_pool():
    """A synthetic but realistically shaped 14-team pool."""
    players = []
    n = 0
    counts = {"QB": 40, "RB": 70, "WR": 90, "TE": 30, "K": 20, "DEF": 20}
    tops = {"QB": 330, "RB": 320, "WR": 315, "TE": 250, "K": 120, "DEF": 110}
    for pos, cnt in counts.items():
        for i in range(cnt):
            n += 1
            players.append(
                mk(n, pos, tops[pos] - i * 4.0, adp=1 + n * 0.8, bye=(i % 6) + 5)
            )
    return players


class TestSnakeMath(unittest.TestCase):
    def setUp(self):
        self.lg = LeagueConfig()

    def test_league_shape(self):
        self.assertEqual(self.lg.starter_slots, 10)
        self.assertEqual(self.lg.rounds, 15)
        self.assertEqual(self.lg.total_picks, 210)

    def test_snake_order(self):
        lg = self.lg
        self.assertEqual(lg.pick_number(1, 1), 1)
        self.assertEqual(lg.pick_number(1, 14), 14)
        self.assertEqual(lg.pick_number(2, 14), 15)   # snake turn
        self.assertEqual(lg.pick_number(2, 1), 28)
        self.assertEqual(lg.pick_number(3, 1), 29)

    def test_slot_7_picks(self):
        picks = self.lg.my_pick_numbers(7)
        self.assertEqual(picks[:4], [7, 22, 35, 50])
        self.assertEqual(len(picks), 15)

    def test_slot_from_pick_is_inverse(self):
        for slot in (1, 7, 14):
            for pick in self.lg.my_pick_numbers(slot):
                self.assertEqual(self.lg.slot_from_pick(pick), slot)

    def test_every_pick_assigned_once(self):
        seen = []
        for slot in range(1, self.lg.teams + 1):
            seen.extend(self.lg.my_pick_numbers(slot))
        self.assertEqual(sorted(seen), list(range(1, 211)))


class TestSurvival(unittest.TestCase):
    def test_monotonic_in_distance(self):
        near = p_survives_to(adp=20, sigma=6, target_pick=25, current_pick=10)
        far = p_survives_to(adp=20, sigma=6, target_pick=45, current_pick=10)
        self.assertGreater(near, far)

    def test_conditioning_raises_probability(self):
        """Surviving longer than expected is evidence he will keep surviving."""
        cold = p_survives_to(adp=20, sigma=6, target_pick=30, current_pick=1)
        warm = p_survives_to(adp=20, sigma=6, target_pick=30, current_pick=25)
        self.assertGreater(warm, cold)

    def test_bounds(self):
        self.assertEqual(p_survives_to(20, 6, 5, 10), 1.0)  # target already passed
        for tgt in (1, 20, 50, 300):
            v = p_survives_to(30, 9, tgt, 1)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_deep_sleeper_survives(self):
        self.assertGreater(p_survives_to(adp=230, sigma=30, target_pick=40, current_pick=1), 0.99)


class TestLineup(unittest.TestCase):
    def setUp(self):
        self.lg = LeagueConfig()

    def test_flex_takes_best_leftover(self):
        roster = [
            mk(1, "QB", 300), mk(2, "RB", 250), mk(3, "RB", 240), mk(4, "RB", 230),
            mk(5, "WR", 220), mk(6, "WR", 210), mk(7, "WR", 205),
            mk(8, "TE", 150), mk(9, "K", 120), mk(10, "DEF", 110),
        ]
        starters, total = optimal_lineup(roster, self.lg)
        flex_names = {p.pid for p in starters["FLEX"]}
        self.assertEqual(flex_names, {"4", "7"})  # RB 230 and WR 205
        self.assertAlmostEqual(total, sum(p.projection for p in roster))

    def test_unfilled_slot_scores_nothing(self):
        no_k = [mk(1, "QB", 300), mk(2, "RB", 250), mk(3, "WR", 220)]
        starters, total = optimal_lineup(no_k, self.lg)
        self.assertEqual(starters.get("K", []), [])
        self.assertAlmostEqual(total, 770.0)

    def test_bye_conflicts_flagged(self):
        roster = [mk(i, "WR", 200 - i, bye=9) for i in range(1, 5)]
        starters, _ = optimal_lineup(roster, self.lg)
        self.assertIn(9, bye_conflicts(starters, allowed=2))

    def test_remaining_needs_counts_flex_surplus(self):
        roster = [mk(1, "RB", 250), mk(2, "RB", 240), mk(3, "RB", 230)]
        needs = remaining_starter_needs(roster, self.lg)
        self.assertEqual(needs["RB"], 0)
        self.assertEqual(needs["FLEX"], 1)   # third RB absorbs one flex slot
        self.assertEqual(needs["QB"], 1)


class TestDraftRules(unittest.TestCase):
    def setUp(self):
        self.lg = LeagueConfig()

    def test_kicker_blocked_early(self):
        self.assertFalse(can_draft(mk(1, "K", 120), [], self.lg, ECFG, picks_left=10))

    @staticmethod
    def roster_needing_k_and_def():
        """A realistic 13-player roster: every slot filled except K and DEF."""
        roster = [mk(1, "QB", 300)]
        roster += [mk(10 + i, "RB", 240 - i * 5) for i in range(5)]
        roster += [mk(20 + i, "WR", 235 - i * 5) for i in range(5)]
        roster += [mk(30 + i, "TE", 180 - i * 5) for i in range(2)]
        assert len(roster) == 13
        return roster

    def test_kicker_allowed_late(self):
        roster = self.roster_needing_k_and_def()
        self.assertTrue(can_draft(mk(1, "K", 120), roster, self.lg, ECFG, picks_left=2))

    def test_must_fill_forces_empty_slots(self):
        """13 players, K and DEF empty, 2 picks left -> only K/DEF are legal."""
        roster = self.roster_needing_k_and_def()
        self.assertEqual(sorted(unfilled_required(roster, self.lg)), ["DEF", "K"])
        self.assertFalse(can_draft(mk(99, "WR", 260), roster, self.lg, ECFG, picks_left=2))
        self.assertTrue(can_draft(mk(98, "DEF", 100), roster, self.lg, ECFG, picks_left=2))

    def test_no_must_fill_pressure_with_picks_to_spare(self):
        """Same roster, more picks left -> free to take the best player."""
        roster = self.roster_needing_k_and_def()
        self.assertTrue(can_draft(mk(99, "WR", 260), roster, self.lg, ECFG, picks_left=5))

    def test_position_cap(self):
        roster = [mk(1, "QB", 300), mk(2, "QB", 290)]
        self.assertFalse(can_draft(mk(3, "QB", 280), roster, self.lg, ECFG, picks_left=8))


class TestReplacement(unittest.TestCase):
    def test_levels_and_flex_allocation(self):
        lg = LeagueConfig()
        players = make_pool()
        levels = compute_replacement_levels(players, lg)
        for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
            self.assertIn(pos, levels)
        # Flex must allocate exactly the available flex slots.
        alloc = flex_allocation(players, lg)
        self.assertEqual(sum(alloc.values()), lg.teams * lg.starters["FLEX"])

    def test_deeper_league_lowers_replacement(self):
        players = make_pool()
        shallow = compute_replacement_levels(players, LeagueConfig(teams=10))
        deep = compute_replacement_levels(players, LeagueConfig(teams=14))
        self.assertGreater(shallow["RB"], deep["RB"])


class TestDraftSettingsDetection(unittest.TestCase):
    """Reading league shape off a Sleeper draft object (mocks differ from home)."""

    HOME = LeagueConfig()

    def test_reads_a_different_shaped_mock(self):
        draft = {
            "type": "snake",
            "metadata": {"scoring_type": "half_ppr"},
            "settings": {
                "teams": 12, "rounds": 16, "slots_qb": 1, "slots_rb": 2, "slots_wr": 3,
                "slots_te": 1, "slots_flex": 1, "slots_k": 1, "slots_def": 1, "slots_bn": 6,
            },
        }
        lg, notes, unsupported = league_from_draft(draft, self.HOME)
        self.assertEqual(lg.teams, 12)
        self.assertEqual(lg.scoring, "half_ppr")
        self.assertEqual(lg.starters["WR"], 3)
        self.assertEqual(lg.rounds, 16)
        self.assertEqual(lg.total_picks, 192)
        self.assertFalse(unsupported)
        self.assertTrue(notes)

    def test_matching_draft_produces_no_notes(self):
        draft = {
            "type": "snake",
            "metadata": {"scoring_type": "ppr"},
            "settings": {
                "teams": 14, "rounds": 15, "slots_qb": 1, "slots_rb": 2, "slots_wr": 2,
                "slots_te": 1, "slots_flex": 2, "slots_k": 1, "slots_def": 1, "slots_bn": 5,
            },
        }
        lg, notes, unsupported = league_from_draft(draft, self.HOME)
        self.assertEqual(notes, [])
        self.assertEqual(lg.teams, 14)
        self.assertEqual(lg.rounds, 15)

    def test_superflex_reported_unsupported(self):
        draft = {"type": "snake", "settings": {"teams": 12, "slots_super_flex": 1}}
        _, _, unsupported = league_from_draft(draft, self.HOME)
        self.assertTrue(any("superflex" in u for u in unsupported))

    def test_rounds_wins_over_starters_plus_bench(self):
        draft = {"type": "snake", "settings": {"teams": 14, "rounds": 20}}
        lg, _, _ = league_from_draft(draft, self.HOME)
        self.assertEqual(lg.rounds, 20)

    def test_empty_settings_falls_back(self):
        lg, _, _ = league_from_draft({}, self.HOME)
        self.assertEqual(lg.teams, self.HOME.teams)
        self.assertEqual(lg.starters, self.HOME.starters)


class TestCandidateDiversity(unittest.TestCase):
    def test_one_position_cannot_own_the_whole_board(self):
        """Late drafts let a single position dominate VOR; the board must not
        become twelve tight ends when you can only roster two."""
        from src.recommend import DraftState, choose_candidates

        cfg = dict(ECFG)
        cfg["max_candidates_per_position"] = 3
        lg = LeagueConfig()

        # Tight ends sweep the top of the VOR ordering.
        avail = [mk(100 + i, "TE", 200 - i, adp=100 + i) for i in range(20)]
        avail += [mk(200 + i, "RB", 100 - i, adp=100 + i) for i in range(10)]
        avail += [mk(300 + i, "WR", 90 - i, adp=100 + i) for i in range(10)]
        for p in avail:
            p.vor = p.projection
            p.survive_next = 1.0

        st = DraftState(current_pick=150, my_slot=7, my_future_picks=[150, 165],
                        on_the_clock=True)
        chosen = choose_candidates(avail, st, lg, cfg, limit=12)

        # The cap allows 3, plus at most one more from the rule that always
        # surfaces the best available player at an unfilled starting slot --
        # this roster still needs a TE, so that extra one is deliberate.
        te_count = sum(1 for p in chosen if p.position == "TE")
        self.assertLessEqual(te_count, cfg["max_candidates_per_position"] + 1)
        self.assertGreaterEqual(len({p.position for p in chosen}), 3)


class TestExpectedBest(unittest.TestCase):
    def test_expected_value_between_zero_and_best(self):
        group = [mk(1, "RB", 300, adp=5, sigma=3), mk(2, "RB", 280, adp=15, sigma=3)]
        for p in group:
            p.vor = p.projection - 150
        exp = expected_best_at(group, target_pick=40, current_pick=1)
        self.assertGreaterEqual(exp, 0.0)
        self.assertLessEqual(exp, group[0].vor)

    def test_sure_thing_returns_full_value(self):
        p = mk(1, "RB", 300, adp=250, sigma=5)
        p.vor = 100.0
        self.assertAlmostEqual(expected_best_at([p], 30, 1), 100.0, places=2)


class TestAdpBlend(unittest.TestCase):
    """Sleeper ADP is what your leaguemates see; FFC is size-specific. Use both."""

    CFG = {
        "adp_weights": {"sleeper": 0.65, "ffc": 0.35},
        "adp_disagreement_cap": 30.0,
    }

    def test_weighted_toward_sleeper(self):
        adp, spread = blend_adp(10.0, 20.0, self.CFG)
        self.assertAlmostEqual(adp, 10.0 * 0.65 + 20.0 * 0.35)
        self.assertLess(adp, 15.0)          # pulled toward Sleeper
        self.assertAlmostEqual(spread, 10.0)

    def test_single_source_is_used_verbatim(self):
        self.assertEqual(blend_adp(12.0, None, self.CFG), (12.0, 0.0))
        self.assertEqual(blend_adp(None, 12.0, self.CFG), (12.0, 0.0))

    def test_no_source_returns_none(self):
        adp, spread = blend_adp(None, None, self.CFG)
        self.assertIsNone(adp)
        self.assertEqual(spread, 0.0)

    def test_agreement_produces_no_spread(self):
        adp, spread = blend_adp(30.0, 30.0, self.CFG)
        self.assertAlmostEqual(adp, 30.0)
        self.assertAlmostEqual(spread, 0.0)

    def test_zero_weights_do_not_divide_by_zero(self):
        adp, _ = blend_adp(10.0, 20.0, {"adp_weights": {"sleeper": 0, "ffc": 0}})
        self.assertIsNotNone(adp)

    def test_blend_stays_between_the_sources(self):
        for a, b in [(5.0, 90.0), (90.0, 5.0), (1.0, 2.0)]:
            adp, _ = blend_adp(a, b, self.CFG)
            self.assertGreaterEqual(adp, min(a, b))
            self.assertLessEqual(adp, max(a, b))


class TestUpside(unittest.TestCase):
    def test_zero_uncertainty_is_plain_intrinsic_value(self):
        self.assertAlmostEqual(option_value(150, 0, 100), 50.0)
        self.assertAlmostEqual(option_value(80, 0, 100), 0.0)

    def test_variance_is_worth_money_below_the_line(self):
        """Same projection, both below waiver level: the volatile one wins."""
        safe = option_value(90, 10, 100)
        wild = option_value(90, 60, 100)
        self.assertGreater(wild, safe)
        self.assertGreater(wild, 0.0)

    def test_never_below_intrinsic(self):
        for mu in (40, 100, 220):
            for sd in (0, 15, 70):
                self.assertGreaterEqual(
                    option_value(mu, sd, 100) + 1e-9, max(0.0, mu - 100)
                )

    def test_rookie_and_backup_widen_outcomes(self):
        base = estimate_sd(150, "RB", 5, 1, 60, 5, ECFG)
        rookie = estimate_sd(150, "RB", 0, 1, 60, 5, ECFG)
        backup = estimate_sd(150, "RB", 5, 2, 60, 5, ECFG)
        self.assertGreater(rookie, base)
        self.assertGreater(backup, base)

    def test_market_disagreement_widens_outcomes(self):
        tight = estimate_sd(150, "WR", 4, 1, 60, 2, ECFG)
        loose = estimate_sd(150, "WR", 4, 1, 60, 40, ECFG)
        self.assertGreater(loose, tight)

    def test_val_plus_up_reconstructs_the_full_option(self):
        """The board splits one number in two; it must still add back up."""
        for proj, sd, waiver in [(300, 80, 177), (90, 60, 100), (120, 10, 150)]:
            val = max(0.0, proj - waiver)
            up = option_value(proj, sd, waiver) - val
            self.assertAlmostEqual(val + up, option_value(proj, sd, waiver))
            self.assertGreaterEqual(up, -1e-9)

    def test_stud_is_all_val_and_flyer_is_all_up(self):
        """The whole point of splitting: the columns separate two profiles."""
        stud = mk(1, "QB", 361.5, sd=81.0, waiver=177.0)
        flyer = mk(2, "RB", 88.7, sd=52.4, waiver=65.2)
        self.assertGreater(stud.val, 150.0)
        self.assertLess(stud.upside, 5.0)          # essentially no optionality
        self.assertGreater(flyer.upside, stud.upside)

    def test_running_backs_more_volatile_than_quarterbacks(self):
        self.assertGreater(
            estimate_sd(200, "RB", 4, 1, 50, 5, ECFG),
            estimate_sd(200, "QB", 4, 1, 50, 5, ECFG),
        )


class TestRosterConstruction(unittest.TestCase):
    """The fixes for 'it keeps ending up with three running backs'."""

    LG = LeagueConfig()

    @staticmethod
    def base_starters(waiver_rb, waiver_wr):
        r = [mk(1, "QB", 300, waiver=200)]
        r += [mk(2, "RB", 250, waiver=waiver_rb), mk(3, "RB", 240, waiver=waiver_rb)]
        r += [mk(4, "WR", 220, waiver=waiver_wr), mk(5, "WR", 210, waiver=waiver_wr)]
        r += [mk(6, "WR", 200, waiver=waiver_wr), mk(7, "WR", 190, waiver=waiver_wr)]
        r += [mk(8, "TE", 150, waiver=90), mk(9, "K", 120, waiver=100),
              mk(10, "DEF", 110, waiver=86)]
        return r

    def score(self, roster, ecfg):
        ctx, index = build_context(roster, self.LG, ecfg)
        return score_roster_fast([index[p.pid] for p in roster], ctx)

    def test_depth_is_worth_more_where_the_waiver_wire_is_barren(self):
        """A bench RB beats an identical bench WR, purely from fallback quality.

        Targets are switched off here so this isolates the injury-cover
        mechanism from the roster-target penalty.
        """
        no_targets = dict(ECFG)
        no_targets["roster_targets"] = {}
        no_targets["target_shortfall_penalty"] = {}

        with_rb = self.base_starters(60, 120) + [mk(20, "RB", 130, waiver=60)]
        with_wr = self.base_starters(60, 120) + [mk(21, "WR", 130, waiver=120)]

        self.assertGreater(self.score(with_rb, no_targets), self.score(with_wr, no_targets))

    def test_roster_target_shortfall_is_penalised(self):
        thin = self.base_starters(60, 120)
        ctx, index = build_context(thin, self.LG, ECFG)
        idx = [index[p.pid] for p in thin]
        with_targets = score_roster_fast(idx, ctx)

        relaxed = dict(ECFG)
        relaxed["roster_targets"] = {}
        relaxed["target_shortfall_penalty"] = {}
        ctx2, index2 = build_context(thin, self.LG, relaxed)
        without = score_roster_fast([index2[p.pid] for p in thin], ctx2)

        self.assertLess(with_targets, without)

    def test_targets_ignored_for_positions_the_league_never_starts(self):
        """A kicker-less mock must not be penalised for having no kicker."""
        no_k = LeagueConfig(
            starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "DEF": 1}
        )
        roster = [
            mk(1, "QB", 300), mk(2, "RB", 250), mk(3, "RB", 240),
            mk(4, "WR", 220), mk(5, "WR", 210), mk(6, "WR", 200), mk(7, "WR", 190),
            mk(8, "TE", 150), mk(9, "DEF", 110),
        ]
        ctx, index = build_context(roster, no_k, ECFG)
        from src.simulate import POS_CODES
        self.assertNotIn(POS_CODES["K"], ctx.targets)
        self.assertIn(POS_CODES["RB"], ctx.targets)

    def test_high_upside_bench_beats_flat_bench(self):
        """Same projection on the bench; the uncertain one is worth more."""
        flat = self.base_starters(60, 120) + [mk(30, "RB", 100, sd=5, waiver=60)]
        wild = self.base_starters(60, 120) + [mk(31, "RB", 100, sd=70, waiver=60)]
        self.assertGreater(self.score(wild, ECFG), self.score(flat, ECFG))

    def test_backup_lifts_the_starter_slot_it_covers(self):
        bare = self.base_starters(60, 120)
        covered = self.base_starters(60, 120) + [mk(40, "RB", 150, waiver=60)]
        no_targets = dict(ECFG)
        no_targets["roster_targets"] = {}
        no_targets["target_shortfall_penalty"] = {}
        gain = self.score(covered, no_targets) - self.score(bare, no_targets)
        # Two RB slots each gain miss_rate * (150 - 60) = 0.22 * 90 ~ 19.8
        self.assertGreater(gain, 25.0)


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.lg = LeagueConfig()
        self.players = make_pool()
        for p in self.players:
            p.replacement = 0.0
            p.vor = p.projection
        self.ctx, self.index = build_context(self.players, self.lg, ECFG)

    def test_score_penalises_bye_stack(self):
        clean = [mk(i, "WR", 200, bye=(i % 6) + 5) for i in range(1, 6)]
        stacked = [mk(i, "WR", 200, bye=9) for i in range(1, 6)]
        ctx_c, _ = build_context(clean, self.lg, ECFG)
        ctx_s, _ = build_context(stacked, self.lg, ECFG)
        sc = score_roster_fast(list(range(len(clean))), ctx_c)
        ss = score_roster_fast(list(range(len(stacked))), ctx_s)
        self.assertGreater(sc, ss)

    def test_simulation_completes_a_full_legal_roster(self):
        """The headline invariant: never finish a draft with an empty slot."""
        avail = list(range(len(self.players)))
        future = self.lg.my_pick_numbers(7)
        scores = simulate(
            ctx=self.ctx, available=avail, my_roster=[], my_future_picks=future[1:],
            current_pick=1, sim_start=future[0], candidates=[avail[0]],
            sims=1, horizon=15,
        )
        self.assertEqual(len(scores), 1)
        # A roster missing K or DEF loses ~200 pts; a complete one clears this.
        self.assertGreater(list(scores.values())[0], 1500)

    def test_candidate_occupies_a_pick_not_a_freebie(self):
        """Off-clock scores must not exceed on-clock scores for the same player.

        If the candidate were handed over for free, the off-clock roster would
        have one extra player and score higher -- the bug this guards.
        """
        avail = list(range(len(self.players)))
        future = self.lg.my_pick_numbers(7)
        cand = [avail[20]]
        on_clock = simulate(
            ctx=self.ctx, available=avail, my_roster=[], my_future_picks=future[1:],
            current_pick=future[0], sim_start=future[0], candidates=cand,
            sims=30, horizon=6,
        )
        off_clock = simulate(
            ctx=self.ctx, available=avail, my_roster=[], my_future_picks=future[1:],
            current_pick=1, sim_start=future[0], candidates=cand,
            sims=30, horizon=6,
        )
        self.assertLessEqual(list(off_clock.values())[0], list(on_clock.values())[0] + 1e-6)

    def test_scarcer_player_preferred_when_values_match(self):
        """Given equal value, the engine should take the one who will not last."""
        scarce = mk(9001, "RB", 300, adp=15, sigma=2)
        safe = mk(9002, "WR", 300, adp=200, sigma=5)
        pool = self.players + [scarce, safe]
        for p in pool:
            p.replacement = 0.0
            p.vor = p.projection
        ctx, index = build_context(pool, self.lg, ECFG)
        avail = list(range(len(pool)))
        future = self.lg.my_pick_numbers(7)
        scores = simulate(
            ctx=ctx, available=avail, my_roster=[], my_future_picks=future[1:],
            current_pick=future[0], sim_start=future[0],
            candidates=[index["9001"], index["9002"]], sims=120, horizon=6,
        )
        self.assertGreater(scores[index["9001"]], scores[index["9002"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
