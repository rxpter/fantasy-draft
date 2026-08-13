"""Leave-one-season-out evaluation of engine variants.

    python backtest/tune.py --replicates 2 --sims 150

Why the candidate list is short: with three seasons of real football, a wide
grid search would overfit *through selection* even under cross-validation --
try enough configurations and one of them wins the holdout by luck. So each
variant here encodes a specific hypothesis the first backtest raised, and the
whole set is small enough that picking a winner still means something.

The protocol: for each held-out season, choose the variant that looks best on
the other two, then report how that choice actually performed on the season it
never saw. The average of those held-out scores is the only number that
estimates future performance. Full-sample numbers are printed too, clearly
labelled, because the gap between them is itself the measure of overfitting.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.history import build_historical_pool  # noqa: E402
from backtest.run import simulate_draft  # noqa: E402
from backtest.scorer import fetch_weekly_actuals, score_roster  # noqa: E402
from src.league import LeagueConfig  # noqa: E402

CONFIG = ROOT / "config.json"
OUT_TEMPLATE = "tuning_stage{}.json"

SEASONS = (2023, 2024, 2025)


NO_TARGETS = {"roster_targets": {}, "target_shortfall_penalty": {}}
NO_MISS = {"starter_miss_rate": {"RB": 0, "WR": 0, "TE": 0, "QB": 0, "K": 0, "DEF": 0}}


def stage2_variants() -> dict:
    """Follow-up once stage 1 identified bench option value as the culprit.

    Binary on/off is not the whole question. Upside was added deliberately and
    is worth keeping if any weight helps, so this sweeps the weight rather than
    discarding the idea. It also asks whether the roster targets -- the running
    back depth mechanism -- can be kept once upside is out of the way.
    """
    k = {"market_ranked_positions": ["K"]}
    return {
        "up0_notargets": {**k, **NO_TARGETS, "upside_weight": 0.0},
        "up0_targets": {**k, "upside_weight": 0.0},
        "up0_softtargets": {
            **k,
            "upside_weight": 0.0,
            "roster_targets": {"RB": 4, "WR": 5, "TE": 1, "QB": 1, "K": 1, "DEF": 1},
            "target_shortfall_penalty": {"RB": 8.0, "WR": 5.0, "TE": 6.0, "QB": 5.0, "K": 0.0, "DEF": 0.0},
        },
        "up10_notargets": {**k, **NO_TARGETS, "upside_weight": 0.10},
        "up20_notargets": {**k, **NO_TARGETS, "upside_weight": 0.20},
    }


def variants() -> dict:
    """A ladder that strips one layer at a time.

    The first backtest showed plain VOR beating the full engine by 45 points,
    and turning the roster targets off only recovered 3 of them. So the targets
    are not the whole story, and guessing again would waste a run. Each rung
    below removes exactly one mechanism, so the gap between adjacent rows
    attributes the damage to a specific layer rather than to "the engine".
    """
    return {
        # The true pre-fix control: kicker ranking by projection, as shipped
        # when the first backtest ran. config.json already carries the fix, so
        # this has to switch it back off explicitly.
        "pre_fix": {"market_ranked_positions": []},

        # Kickers ranked by market instead of projection.
        "k_market": {"market_ranked_positions": ["K"]},

        # Is running back depth overpaid? Halve the target and soften the fine.
        "soft_rb": {
            "market_ranked_positions": ["K"],
            "roster_targets": {"RB": 4, "WR": 5, "TE": 1, "QB": 1, "K": 1, "DEF": 1},
            "target_shortfall_penalty": {"RB": 8.0, "WR": 5.0, "TE": 6.0, "QB": 5.0, "K": 0.0, "DEF": 0.0},
        },

        # Targets off entirely.
        "no_targets": {"market_ranked_positions": ["K"], **NO_TARGETS},

        # ...and no bench option value.
        "no_upside": {"market_ranked_positions": ["K"], **NO_TARGETS, "upside_weight": 0.0},

        # ...and no injury-fallback blend. The objective is now close to raw
        # startable points, so what remains between this and plain VOR is the
        # Monte Carlo itself.
        "no_risk": {
            "market_ranked_positions": ["K"],
            **NO_TARGETS,
            "upside_weight": 0.0,
            **NO_MISS,
        },
    }


def make_config(base: dict, overrides: dict, sims: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["engine"].update(overrides)
    cfg["engine"]["sims"] = sims
    return cfg


def evaluate(season, league, cfg, weekly, replicates, slots, strategy="engine"):
    """Mean regular-season points for one variant on one season."""
    players, _ = build_historical_pool(season, league, cfg)
    scores = []
    for rep in range(replicates):
        for slot in slots:
            seed = hash((season, rep, slot)) & 0xFFFFFFFF
            roster = simulate_draft(players, league, cfg, slot, strategy, seed)
            scores.append(score_roster(roster, weekly, league)["regular"])
    return statistics.fmean(scores), scores


def main():
    ap = argparse.ArgumentParser(description="Leave-one-season-out tuning")
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--sims", type=int, default=150)
    ap.add_argument("--slots", type=int, nargs="+", default=None)
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    args = ap.parse_args()

    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    league = LeagueConfig.from_dict(base["league"])
    slots = args.slots or list(range(1, league.teams + 1))
    vs = variants() if args.stage == 1 else stage2_variants()

    print(f"variants={len(vs)} seasons={len(SEASONS)} slots={len(slots)} "
          f"replicates={args.replicates} sims={args.sims}")
    print(f"total engine drafts: {len(vs) * len(SEASONS) * len(slots) * args.replicates}\n")

    weekly = {}
    for s in SEASONS:
        print(f"  loading {s} actuals...", end="", flush=True)
        weekly[s] = fetch_weekly_actuals(s, league.scoring)
        print(" ok")

    results: dict[str, dict[int, float]] = {}
    t0 = time.time()

    # Reference line: plain VOR, the thing the full engine has to beat.
    ref_cfg = make_config(base, {"market_ranked_positions": ["K"]}, args.sims)
    results["[plain VOR]"] = {}
    for season in SEASONS:
        mean, _ = evaluate(
            season, league, ref_cfg, weekly[season], args.replicates, slots, strategy="vor"
        )
        results["[plain VOR]"][season] = round(mean, 1)
        print(f"\r  {'[plain VOR]':<20} {season} -> {mean:7.0f}    ", end="", flush=True)
    print()

    for name, overrides in vs.items():
        cfg = make_config(base, overrides, args.sims)
        results[name] = {}
        for season in SEASONS:
            mean, _ = evaluate(season, league, cfg, weekly[season], args.replicates, slots)
            results[name][season] = round(mean, 1)
            print(f"\r  {name:<20} {season} -> {mean:7.0f}    ", end="", flush=True)
        print()
    print(f"\ncompleted in {time.time() - t0:.0f}s\n")

    out = ROOT / "backtest" / OUT_TEMPLATE.format(args.stage)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(report(results))
    return 0


def report(results: dict) -> str:
    # JSON turns the integer season keys into strings on reload, so normalise
    # before indexing -- this function has to work on a saved run too.
    results = {
        name: {int(season): value for season, value in row.items()}
        for name, row in results.items()
    }
    out, w = [], lambda s: out.append(s)
    # The VOR reference is a yardstick, not a candidate -- it is a different
    # strategy, not a configuration the engine could adopt.
    names = [n for n in results if not n.startswith("[")]
    base_name = "pre_fix" if "pre_fix" in results else names[0]

    w("=" * 74)
    w("LEAVE-ONE-SEASON-OUT EVALUATION")
    w("=" * 74)
    w("")
    w("-- mean regular-season points per variant --")
    w(f"  {'variant':<22}" + "".join(f"{s:>10}" for s in SEASONS) + f"{'mean':>10}")
    for n in results:
        row = results[n]
        mean = statistics.fmean(row.values())
        w(f"  {n:<22}" + "".join(f"{row[s]:>10.0f}" for s in SEASONS) + f"{mean:>10.0f}")
    w("")

    w("-- leave-one-season-out: pick on two, score on the third --")
    held = []
    for test in SEASONS:
        train = [s for s in SEASONS if s != test]
        pick = max(names, key=lambda n: statistics.fmean(results[n][s] for s in train))
        score = results[pick][test]
        base_score = results[base_name][test]
        held.append(score - base_score)
        w(f"  hold out {test}: trained on {train} -> chose '{pick}'")
        w(f"     scored {score:.0f} vs baseline {base_score:.0f}   "
          f"({score - base_score:+.0f})")
    w("")
    w(f"  honest held-out gain over pre-fix: {statistics.fmean(held):+.1f} pts/season")
    w("")

    w("-- full-sample best (optimistic, do not trust) --")
    best = max(names, key=lambda n: statistics.fmean(results[n].values()))
    gain = statistics.fmean(results[best].values()) - statistics.fmean(results[base_name].values())
    w(f"  '{best}' at {gain:+.1f} pts/season over baseline.")
    w(f"  The gap between this and the held-out figure above is the overfitting.")
    w("=" * 74)
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
