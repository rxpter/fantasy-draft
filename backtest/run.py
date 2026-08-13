"""Backtest runner.

    python backtest/run.py --seasons 2023 2024 2025 --replicates 3

Design note -- paired comparison. For a given (season, replicate, slot) all
three strategies draft against the *same* sampled opponent order. Comparing
within those pairs cancels most of the luck, which matters enormously here:
there are only three seasons of real football, and simulating more leagues
averages over opponent randomness without creating new independent evidence.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.history import build_historical_pool, fetch_season_actuals  # noqa: E402
from backtest.scorer import fetch_weekly_actuals, hit_rate, score_roster  # noqa: E402
from backtest.strategies import STRATEGIES  # noqa: E402
from src.league import LeagueConfig  # noqa: E402
from src.recommend import build_state  # noqa: E402

CONFIG = ROOT / "config.json"
RESULTS = ROOT / "backtest" / "results.json"


def simulate_draft(players, league, cfg, slot, strategy, seed):
    """One full draft. Returns the roster the target slot ended up with."""
    ecfg = cfg["engine"]
    by_pid = {p.pid: p for p in players}
    rng = random.Random(seed)

    # Opponents draft near ADP with noise. Shared across strategies via `seed`.
    order = sorted(players, key=lambda p: p.adp + p.sigma * rng.gauss(0, 1))

    picks, taken, ptr = [], set(), 0
    pick_fn = STRATEGIES[strategy]

    while len(picks) < league.total_picks:
        pick_no = len(picks) + 1
        seat = league.slot_from_pick(pick_no)

        if seat == slot:
            st = build_state(picks, by_pid, league, slot)
            chosen = pick_fn(players, st, league, ecfg)
        else:
            chosen = None
            while ptr < len(order):
                cand = order[ptr]
                ptr += 1
                if cand.pid not in taken:
                    chosen = cand
                    break
        if chosen is None:
            break

        taken.add(chosen.pid)
        picks.append(
            {
                "pick_no": pick_no,
                "round": (pick_no - 1) // league.teams + 1,
                "draft_slot": seat,
                "player_id": chosen.pid,
                "metadata": {"last_name": chosen.name.split()[-1]},
            }
        )

    return build_state(picks, by_pid, league, slot).my_roster


def run_season(season, league, cfg, replicates, slots, verbose=True):
    if verbose:
        print(f"\n=== {season} ===")
        print("  loading preseason pool...", end="", flush=True)
    players, diag = build_historical_pool(season, league, cfg)
    if verbose:
        print(
            f" {diag['players']} players | ffc {diag['ffc_matched']}/{diag['ffc_rows']}"
            f" matched | byes {diag['with_bye']}"
        )
        print("  loading weekly actuals...", end="", flush=True)
    weekly = fetch_weekly_actuals(season, league.scoring)
    actuals = fetch_season_actuals(season, league.scoring)
    if verbose:
        scored = sum(1 for w in weekly.values() if w)
        print(f" {scored} weeks")

    rows = []
    total = len(slots) * replicates
    done = 0
    for rep in range(replicates):
        for slot in slots:
            seed = hash((season, rep, slot)) & 0xFFFFFFFF
            for strategy in STRATEGIES:
                t0 = time.time()
                roster = simulate_draft(players, league, cfg, slot, strategy, seed)
                res = score_roster(roster, weekly, league)
                hits = hit_rate(roster, actuals)
                rows.append(
                    {
                        "season": season,
                        "rep": rep,
                        "slot": slot,
                        "strategy": strategy,
                        "regular": res["regular"],
                        "playoffs": res["playoffs"],
                        "total": res["total"],
                        "empty_slots": res["empty_starter_slots"],
                        "by_position": res["by_position"],
                        "hits": hits["hits"],
                        "busts": hits["busts"],
                        "roster": [p.name for p in roster],
                        "seconds": round(time.time() - t0, 2),
                    }
                )
            done += 1
            if verbose:
                print(f"\r  drafting {done}/{total} slots", end="", flush=True)
    if verbose:
        print()
    return rows


def paired(rows, a, b, metric="regular"):
    """Per-matchup difference a - b, over identical opponent draws."""
    idx = {}
    for r in rows:
        idx.setdefault((r["season"], r["rep"], r["slot"]), {})[r["strategy"]] = r
    diffs = []
    for key, byname in idx.items():
        if a in byname and b in byname:
            diffs.append(byname[a][metric] - byname[b][metric])
    return diffs


def describe(diffs):
    if not diffs:
        return {}
    mean = statistics.fmean(diffs)
    sd = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    se = sd / (len(diffs) ** 0.5) if diffs else 0.0
    return {
        "n": len(diffs),
        "mean": round(mean, 1),
        "median": round(statistics.median(diffs), 1),
        "win_rate": round(sum(1 for d in diffs if d > 0) / len(diffs), 3),
        "sd": round(sd, 1),
        "ci95": [round(mean - 1.96 * se, 1), round(mean + 1.96 * se, 1)],
    }


def report(rows):
    out = []
    w = out.append
    seasons = sorted({r["season"] for r in rows})
    names = list(STRATEGIES)

    w("=" * 76)
    w("BACKTEST REPORT")
    w("=" * 76)
    w("")
    w("Scoring: weekly optimal-legal lineup by preseason projection, weeks 1-14.")
    w("Paired: all strategies face identical opponent draws per season/rep/slot.")
    w("")

    w("-- mean starting-lineup points, fantasy regular season --")
    w(f"  {'season':<8}" + "".join(f"{n:>12}" for n in names))
    for s in seasons:
        line = f"  {s:<8}"
        for n in names:
            vals = [r["regular"] for r in rows if r["season"] == s and r["strategy"] == n]
            line += f"{statistics.fmean(vals):>12.0f}" if vals else f"{'--':>12}"
        w(line)
    line = f"  {'all':<8}"
    for n in names:
        vals = [r["regular"] for r in rows if r["strategy"] == n]
        line += f"{statistics.fmean(vals):>12.0f}" if vals else f"{'--':>12}"
    w(line)
    w("")

    w("-- head to head (paired differences, regular season points) --")
    for a, b in (("engine", "adp"), ("engine", "vor"), ("vor", "adp")):
        d = describe(paired(rows, a, b))
        if not d:
            continue
        w(f"  {a} vs {b}:")
        w(f"     mean {d['mean']:+.1f} pts   median {d['median']:+.1f}   "
          f"wins {d['win_rate']:.1%} of {d['n']}   95% CI [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]")
    w("")

    w("-- engine vs adp by season --")
    for s in seasons:
        sub = [r for r in rows if r["season"] == s]
        d = describe(paired(sub, "engine", "adp"))
        if d:
            w(f"  {s}: {d['mean']:+.1f} pts   wins {d['win_rate']:.1%}   95% CI "
              f"[{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]")
    w("")

    w("-- engine vs adp by draft slot --")
    slots = sorted({r["slot"] for r in rows})
    for sl in slots:
        sub = [r for r in rows if r["slot"] == sl]
        d = describe(paired(sub, "engine", "adp"))
        if d:
            w(f"  slot {sl:>2}: {d['mean']:+7.1f} pts   wins {d['win_rate']:.0%}")
    w("")

    w("-- roster shape and outcomes --")
    w(f"  {'strategy':<10}{'hits':>8}{'busts':>8}{'empty slots':>14}{'playoff pts':>14}")
    for n in names:
        sub = [r for r in rows if r["strategy"] == n]
        if not sub:
            continue
        w(f"  {n:<10}{statistics.fmean([r['hits'] for r in sub]):>8.1f}"
          f"{statistics.fmean([r['busts'] for r in sub]):>8.1f}"
          f"{statistics.fmean([r['empty_slots'] for r in sub]):>14.1f}"
          f"{statistics.fmean([r['playoffs'] for r in sub]):>14.0f}")
    w("")

    w("-- points by position (mean, regular + playoffs) --")
    positions = ("QB", "RB", "WR", "TE", "K", "DEF")
    w(f"  {'strategy':<10}" + "".join(f"{p:>9}" for p in positions))
    for n in names:
        sub = [r for r in rows if r["strategy"] == n]
        if not sub:
            continue
        line = f"  {n:<10}"
        for p in positions:
            vals = [r["by_position"].get(p, 0.0) for r in sub]
            line += f"{statistics.fmean(vals):>9.0f}"
        w(line)
    w("")

    n_eff = len({(r["season"]) for r in rows})
    w("-- caveats --")
    w(f"  * {n_eff} seasons of real football. Replicates average over opponent")
    w("    randomness, not over football, so the effective sample is small.")
    w("    Treat anything inside the confidence intervals as unproven.")
    w("  * Historical FFC ADP is 12-team; no 14-team archive exists.")
    w("  * Opponents draft near ADP, so they never reach for need. Real")
    w("    leaguemates do, which likely flatters every strategy equally.")
    w("=" * 76)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Historical backtest")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--slots", type=int, nargs="+", default=None)
    ap.add_argument("--sims", type=int, default=200, help="Monte Carlo iterations")
    ap.add_argument("--fetch-only", action="store_true", help="warm caches and exit")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg["engine"]["sims"] = args.sims
    league = LeagueConfig.from_dict(cfg["league"])
    slots = args.slots or list(range(1, league.teams + 1))

    if args.fetch_only:
        for s in args.seasons:
            print(f"warming {s}...", end="", flush=True)
            build_historical_pool(s, league, cfg)
            fetch_weekly_actuals(s, league.scoring)
            fetch_season_actuals(s, league.scoring)
            print(" done")
        return 0

    print(f"backtest: seasons={args.seasons} replicates={args.replicates} "
          f"slots={len(slots)} sims={args.sims}")
    t0 = time.time()
    rows = []
    for season in args.seasons:
        rows.extend(run_season(season, league, cfg, args.replicates, slots))

    RESULTS.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} drafts in {time.time() - t0:.0f}s -> {RESULTS.name}\n")
    print(report(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
