# Fantasy Draft Assistant

A live draft assistant for **Sleeper**. It watches your draft as it happens and
tells you who to take — using positional scarcity, survival probability, and a
Monte Carlo simulation of the rest of the draft, rather than reading you a
static ranking.

Pre-configured for **14-team PPR — 1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX, 1 K, 1 DEF,
5 bench** (10 starters, 15 rounds, 210 picks), and it auto-detects any other
league shape off the draft itself.

Zero dependencies. Pure Python standard library. No `pip install`, no API keys,
no account beyond the Sleeper one you already have.

---

## Contents

1. [What makes it different](#what-makes-it-different)
2. [Requirements](#requirements)
3. [Quick start](#quick-start)
4. [Part 1 — Testing with mock drafts](#part-1--testing-with-mock-drafts)
5. [Part 2 — Running a live draft](#part-2--running-a-live-draft)
6. [Reading the board](#reading-the-board)
7. [How players are valued](#how-players-are-valued)
8. [The objective function](#the-objective-function)
9. [The Monte Carlo](#the-monte-carlo)
10. [Configuration reference](#configuration-reference)
11. [Tuning recipes](#tuning-recipes)
12. [Bringing your own projections](#bringing-your-own-projections)
13. [Command line reference](#command-line-reference)
14. [Project layout](#project-layout)
15. [Tests](#tests)
16. [Data sources](#data-sources)
17. [Troubleshooting](#troubleshooting)
18. [Limitations](#limitations)
19. [Design decisions](#design-decisions)
20. [License](#license)

---

## What makes it different

A ranked cheat sheet answers "who is the best player left?" That is the wrong
question. The right one is **"who will still be here at my next pick?"**

Two examples the engine produces from real 2026 data in this league:

**Josh Allen projects the most fantasy points of anyone on the board — 361.5 —
and falls outside the top 20.** Only 14 quarterbacks start in a 14-team league,
so the QB you could get for free already scores 280. His actual edge is 81
points, smaller than a mid-tier running back's.

**At pick 22 it ranks Nico Collins (VOR 95) above Brock Bowers (VOR 98).** The
lower-value player is the correct pick, because Bowers has a 77% chance of
surviving to your next turn and Collins has 0%.

That is the whole idea: value is only realisable if the alternative would not
have been there anyway.

---

## Requirements

- **Python 3.14** is what this was developed and verified on. Every module uses
  `from __future__ import annotations`, so 3.9+ should work — but 3.14 is the
  only version actually tested.
- An internet connection at startup (everything is then cached locally).
- A Sleeper account, for live and mock drafts. The offline rehearsal needs
  neither an account nor a draft.

No third-party packages. This was deliberate — a bare Python install minutes
before a draft is not the moment to discover a broken `pip`.

---

## Quick start

Clone the repo, or `cd` into the folder if you already have it:

```bash
cd fantasy-draft
```

```bash
python -m unittest discover -s tests
```

```bash
python draft.py --mock --mock-slot 7
```

That runs a full offline 210-pick draft against itself and needs no Sleeper
account. It also warms the data cache (~14 MB), so do it once before draft day.

---

## Part 1 — Testing with mock drafts

There are two kinds of rehearsal, and they test different things.

### 1A. The offline mock (best for judging pick quality)

Plays all 210 picks against realistic ADP-driven opponents and hands you a
finished roster. No account, no waiting, ~20 seconds.

**Step 1.** Open a terminal in the project folder:

```bash
cd fantasy-draft
```

**Step 2.** Run a full draft from a slot:

```bash
python draft.py --mock --mock-slot 7
```

**Step 3.** Read the **final screen** — that is the deliverable. Check:

- **The `depth:` line.** Every position should be at or near its target. If
  `RB` shows `3/5` in red, roster construction is off.
- **The starting lineup.** No empty slots. A kicker and a defense, both taken
  in the last two rounds.
- **The bench.** Should hold upside plays, not a redundant backup QB.

**Step 4.** Try several slots — draft position changes strategy a lot:

```bash
python draft.py --mock --mock-slot 1
```

```bash
python draft.py --mock --mock-slot 14
```

**Step 5.** To watch it think pick by pick rather than blitzing through:

```bash
python draft.py --mock --mock-slot 7 --mock-delay 2
```

### 1B. A real Sleeper mock draft (best for testing the live plumbing)

This exercises the actual pick feed — the one part the offline mock cannot test.

**Step 1.** In the Sleeper app or at sleeper.com, go to **Draft → Mock Draft**.
Choose **14 teams** and **PPR** to mirror your league. Any settings work; the
tool reads the real shape off the draft.

**Step 2.** Once you are in the draft room, copy the draft id from the URL:

```
sleeper.com/draft/nfl/1234567890123456789
                      ^^^^^^^^^^^^^^^^^^^ this is the draft id
```

On the phone app instead, list your drafts — mocks belong to no league, so they
need their own lookup, which `--list` handles:

```bash
python draft.py --list --username YOUR_SLEEPER_NAME
```

**Step 3.** Run the assistant in a second window, beside your browser:

```bash
python draft.py --draft-id PASTE_ID_HERE --username YOUR_SLEEPER_NAME
```

**Step 4.** Confirm the startup lines are right. It prints what it detected:

```
draft: Mock Draft  |  type=snake  |  status=drafting
  settings taken from draft: teams 14 -> 12; scoring ppr -> half_ppr
  12 teams | 15 rounds | half_ppr | starters {...} | bench 5
  your slot: 7
```

If team count or scoring is wrong there, stop and fix it — **every number
downstream depends on those two**.

**Step 5.** Draft. The board refreshes every 3 seconds.

### What to verify in a mock

| Check | Working looks like |
|---|---|
| Board updates | New picks appear in the `recent:` line within ~3s |
| Slot detection | `your slot: N` — mock lobbies assign it after you join, and it is picked up automatically |
| `UP` column | Near zero for round-1 studs, ≥10 (highlighted) for round-12 flyers |
| `VOR` vs rank | The highest-VOR player should **not** always be #1. If he is, the risk model is not engaging |
| `depth:` line | Turns red when thin, and pushes you back to that position |
| Late rounds | Offers backups and rookies, tagged `backup upside` / `rookie` |
| K and DEF | Never offered before the last 2–3 rounds |

> **Note on bot mocks:** Sleeper mocks filled with bots can finish 15 rounds in
> two minutes. That is fine for confirming the feed works, but too fast to judge
> recommendations. Use the offline mock (1A) to evaluate pick quality.

---

## Part 2 — Running a live draft

### Before draft day

**Step 1.** Run the tests. Thirty seconds now beats a surprise later:

```bash
python -m unittest discover -s tests
```

**Step 2.** Warm the cache. The Sleeper player file is ~14 MB; cached, startup
drops to about a tenth of a second:

```bash
python draft.py --mock --mock-slot 7
```

**Step 3.** Find your league's draft id and your slot:

```bash
python draft.py --list --username YOUR_SLEEPER_NAME
```

Output looks like:

```
user: yourname  (id 123456789)

  league: My League   teams=14  id=987654321
    draft 111222333  status=pre_draft  type=snake  teams=14  rounds=15  your_slot=?
```

**Step 4.** Save it so draft day is one command. Edit `config.json`:

```json
"draft": {
  "sleeper_username": "YOUR_SLEEPER_NAME",
  "draft_id": "111222333",
  "my_draft_slot": null
}
```

Leave `my_draft_slot` as `null` — Sleeper usually draws the order shortly before
the draft, and it is detected automatically.

**Step 5.** Confirm `config.json` matches your league — `teams`, `starters`,
`bench`, `scoring`. These are auto-detected from the draft too, but having them
right means the offline mock reflects your real league.

### On draft day

**Step 1.** Start it 10–15 minutes early, in a window beside the draft room:

```bash
python draft.py
```

That uses everything in `config.json`. Or pass it explicitly:

```bash
python draft.py --username YOUR_SLEEPER_NAME
```

**Step 2.** Check the startup block once:

```
draft: My League  |  type=snake  |  status=pre_draft
  14 teams | 15 rounds | ppr | starters {...} | bench 5
  your slot: 7
loading player pool for 2026 (14-team ppr)...
  585 players | 583 projected | 256/256 ADP matched | 0.1s
  replacement level: QB 280  RB 155  TE 155  WR 156   | flex fills as {'WR': 25, 'RB': 3}
```

If it says `slot not set yet`, that is fine — it keeps looking and picks it up
when the order is drawn.

**Step 3.** Draft. When you are on the clock you get:

```
>>> YOU ARE ON THE CLOCK <<<
```

Take the `*` row, or read `VERDICT` — it states the margin, and says
"near tie" under ~1.5 points rather than pretending to be certain.

**Step 4.** Between your picks, watch **COST OF WAITING**. That panel is what
tells you which position to attack next round.

`Ctrl-C` quits. Restarting is safe and cheap — all state is read from Sleeper,
nothing is stored locally except the data cache.

### If something goes wrong mid-draft

- **Network blip** — fetch failures are caught and retried; cached data is
  served stale rather than crashing.
- **Slot wrong or undetected** — restart with `--slot N`.
- **Settings misdetected** — restart with `--no-auto-settings` to force
  `config.json`.
- **Too slow between picks** — restart with `--sims 300`.

---

## Reading the board

```
 DRAFT ASSISTANT  |  pick 22/210  |  drafting
 >>> YOU ARE ON THE CLOCK <<<
-------------------------------------------------------------------------------
#  PLAYER              POS  TM  BYE    PROJ   VOR   VAL   UP  BACK     SIM   NOTES
*1 Nico Collins        WR   HOU 8     251.5    95    94    4    0%  1969.6   gone if you wait
 2 George Pickens      WR   DAL 14    245.7    89    88   11    0%  1965.2   gone if you wait
 4 Brock Bowers        TE   LV  13    253.5    98   176    1   77%  1947.6
 5 Josh Allen          QB   BUF 7     361.5    81   184    0   19%  1947.5
```

Bowers has the highest VOR on that screen and is ranked fourth — he is 77% to
come back to you. Josh Allen has the highest projection in the entire draft and
is ranked fifth.

### Columns

| Column | Meaning |
|---|---|
| **PROJ** | Season projection, discounted for injury status |
| **VOR** | Points above **draft-day replacement** — the last player at this position who starts anywhere in the league |
| **VAL** | Points above a **waiver-wire streamer**, if he hits his projection exactly |
| **UP** | Pure upside — what his *uncertainty alone* is worth, on top of VAL. Highlighted at ≥10 |
| **BACK** | Probability he is still available at your **next** pick |
| **SIM** | Risk-adjusted value of your finished roster if you take him now. **The list is sorted by this** |

`VAL` and `UP` are two halves of one number, split because they mean opposite
things:

| Player | VAL | UP | Reading |
|---|---|---|---|
| Josh Allen | 185 | 0 | All floor. No uncertainty left to pay for |
| Omar Cooper | 0 | 14 | All ceiling. Worthless at his median, valuable at his peak |

**Round 12 is where the UP column earns its place.**

### Panels

- **VERDICT** — the pick and its margin over the runner-up.
- **COST OF WAITING** — per position, the value you lose by skipping it this
  round. A long RB bar and a short TE bar means take the RB; the tight end tier
  survives. This is the most decision-relevant panel on the screen.
- **YOUR ROSTER** — your optimal starting lineup, bench, unfilled slots, and bye
  conflicts.
- **depth:** — counts against roster targets, red when thin.
- **recent:** — the last six picks, so you can confirm the feed is live.

### Notes column

| Note | Meaning |
|---|---|
| `gone if you wait` | ≤15% chance of returning to you |
| `87% to return` | ≥85% chance of returning |
| `RB cliff (-27)` | Waiting costs 27 points of RB value |
| `backup upside` | Second on his depth chart — a bet on the man ahead of him |
| `rookie` | ≤1 year experience, wide range of outcomes |
| `bye 13 stack x3` | He would be your third starter on that bye |
| `questionable` | Sleeper injury designation |

---

## How players are valued

### Stage 1 — Raw projection

Sleeper's season `pts_ppr` (or `pts_half_ppr` / `pts_std`). Currently **583 of
585** players carry a real projection.

### Stage 2 — Injury discount

`projection = raw × injury_multiplier[status]`

| Status | Multiplier |
|---|---|
| IR | 0.35 |
| Out / PUP | 0.55 |
| Doubtful | 0.75 |
| Suspended | 0.70 |
| Questionable | 0.97 |

### Stage 3 — Replacement level → VOR

League-wide starters are `teams × slots`: 28 RB, 28 WR, 14 TE, 14 QB, 14 K,
14 DEF. The 28 flex spots are then allocated **endogenously** — everyone past
their positional baseline is pooled, sorted by projection, and the best 28 win.
In this league that resolves to **25 WR, 3 RB, 0 TE**, so:

- RB replacement = **RB32**
- WR replacement = **WR54**
- TE and QB replacement = **TE15 / QB15**

```
VOR = (projection − replacement) × position_value_multiplier
```

### Stage 4 — ADP and survival probability

FantasyFootballCalculator supplies **14-team PPR** ADP with a per-player stdev.
Effective sigma is `max(stdev, 1.0) × 1.15` — FFC's stdev is pooled across
thousands of drafts, so it measures consensus stability, not how erratic your
particular leaguemates are.

```
P(survives to T | still here at C) = (1 − Φ((T−0.5−adp)/σ)) / (1 − Φ((C−0.5−adp)/σ))
```

The conditioning matters: a player still on the board well past his ADP is
*more* likely to keep falling.

### Stage 5 — Projection uncertainty

```
sd = max(projection × cv[position], 12)  ×  boost
```

| Position | cv |
|---|---|
| RB | 0.32 |
| TE | 0.30 |
| WR | 0.28 |
| QB | 0.20 |
| DEF | 0.18 |
| K | 0.15 |

Boost adds `+0.35` for `years_exp ≤ 1`, `+0.45` for `depth_chart_order ≥ 2`, and
`+0.5 × min(adp_stdev / adp, 1)` for market disagreement.

### Stage 6 — Waiver level, VAL and UP

The waiver level is what you could stream at that position **after** the draft —
measured at the same depth for every position (the 52nd RB, the 52nd WR):

| Pos | Replacement (draft) | Waiver (in-season) | Collapse |
|---|---|---|---|
| RB | 155.4 | 65.2 | **−90.2** |
| WR | 156.4 | 157.3 | **+0.9** |
| TE | 155.0 | 77.0 | −78.0 |
| QB | 280.2 | 177.0 | −103.2 |
| K | 100.0 | 75.0 | −25.0 |
| DEF | 86.0 | 66.0 | −20.0 |

**Receivers do not degrade at all over that stretch; running backs lose 90
points.** That single row is the entire empirical case for backfield depth, and
it is measured, not assumed.

Then, with `X ~ Normal(projection, sd)`:

```
full   = E[max(0, X − waiver)]  =  (μ−K)·Φ(d) + sd·φ(d)      [Bachelier]
VAL    = max(0, projection − waiver)                          [intrinsic]
UP     = full − VAL                                           [time value]
```

Why uncertainty is worth money — same projection, same strike, only `sd` moves:

| proj | sd | strike | option value |
|---|---|---|---|
| 120 | 10 | 150 | 0.0 |
| 120 | 40 | 150 | 5.2 |
| 120 | 80 | 150 | **19.1** |

A player projected *below* the man he would replace is worthless if he is
predictable and genuinely valuable if he is not.

---

## The objective function

`VOR` is **not** the ranking. `SIM` is, and it comes from scoring a finished
15-man roster:

```
score =  Σ starters [ proj × (1 − miss_rate) + miss_rate × fallback ]
       − Σ bye weeks with >2 starters [ (n − 2) × 6.0 ]
       + 0.40 × Σ bench [ option_value(proj, sd, strike) ]
       − Σ positions below target [ shortfall × penalty ]
```

**1. Starters are blended with their real fallback.** `miss_rate` is RB 0.22,
WR/TE 0.15, QB 0.10, K/DEF 0.05. `fallback` is your best *non-starting* player
at that position, or the waiver number if you have none. Covering an RB slot
from your bench instead of waivers is worth `0.22 × (bench_RB − 65.2)`; the same
move at WR is worth `0.15 × (bench_WR − 157.3)` — usually near zero. That is why
RB depth pays and WR depth does not.

**2. Bench players are priced as options**, and the strike is **the starter he
would have to beat**, never the waiver wire. Same player, three strikes:

| Strike | Option value |
|---|---|
| waiver RB (65.2) | 34.7 |
| your weakest RB starter (150) | 3.1 |
| a strong starter (220) | 0.1 |

Pricing a backup QB against waivers makes him look like a 130-point steal and
burns a bench spot every draft.

**3. Roster targets.** Finishing with three running backs is fragile however the
points add up. Shortfall costs RB 22, WR 9, TE 6, QB 5 per missing player.
Targets are ignored for positions your league does not start.

Two rules are hard-coded because they are simply correct:

- **K and DEF are refused** until the last 2–3 rounds.
- **Must-fill:** once your remaining picks equal your empty starting slots,
  those picks are forced to fill them. An unfilled slot scores zero — about 100
  points — which no VOR comparison would ever catch.

---

## The Monte Carlo

For each candidate:

1. Place him on your roster **at your next pick** — not for free. Opponents
   consume the gap between now and then first.
2. Sample a plausible draft order: every remaining player gets a notional pick
   number of `adp + sigma × gauss()`, then they go in that order.
3. At each of your next `sim_horizon_picks` (6) turns, pick greedily by VOR,
   respecting caps, the K/DEF rule, and must-fill.
4. Fill the remaining rounds deterministically — the tail of a draft is
   low-leverage and spending simulations on it buys nothing.
5. Score the finished roster with the objective above.

Repeat 800 times, average, sort. Runs in about 0.8 seconds.

**Common random numbers:** every candidate is evaluated against the *same*
sampled draft order within an iteration. That cancels most of the noise between
candidates, so 800 iterations here discriminate better than several thousand
independent ones would.

The simulation only runs when your pick is within 12 — elsewhere it would be
wasted work, and the board says `sim idle (not near your pick)`.

---

## Configuration reference

Everything lives in `config.json`.

### `league`

| Key | Default | Meaning |
|---|---|---|
| `teams` | 14 | **Drives replacement level and every pick number** |
| `scoring` | `ppr` | `ppr`, `half_ppr`, or `std` |
| `starters` | 1/2/2/1/2/1/1 | Starting slots by position, `FLEX` included |
| `flex_positions` | RB, WR, TE | What may fill a flex slot |
| `bench` | 5 | Bench spots; with starters this sets round count |

Against a live draft these are **auto-detected from the draft object** and
override `config.json`. `--no-auto-settings` disables that.

### `draft`

| Key | Meaning |
|---|---|
| `sleeper_username` | Saves typing `--username` |
| `draft_id` | Saves typing `--draft-id` |
| `my_draft_slot` | Usually leave `null` — it is detected |

### `engine` — simulation

| Key | Default | Meaning |
|---|---|---|
| `sims` | 800 | Monte Carlo iterations (~0.8s) |
| `candidates` | 14 | Players evaluated per refresh |
| `max_candidates_per_position` | 4 | Stops one position owning the board |
| `sim_horizon_picks` | 6 | Your picks simulated in detail before the deterministic tail |

### `engine` — scarcity

| Key | Default | Meaning |
|---|---|---|
| `adp_sigma_floor` | 1.0 | Minimum ADP spread |
| `adp_sigma_inflate` | 1.15 | FFC's pooled stdev understates single-draft variance |
| `undrafted_adp` | 230.0 | ADP for players with no FFC row |
| `undrafted_sigma` | 30.0 | Their spread |

### `engine` — roster construction

| Key | Default | Meaning |
|---|---|---|
| `roster_targets` | RB 5, WR 5, TE/QB/K/DEF 1 | Desired end-of-draft counts |
| `target_shortfall_penalty` | RB 22, WR 9, TE 6, QB 5 | Points charged per player short |
| `starter_miss_rate` | RB 0.22, WR/TE 0.15, QB 0.10 | Season fraction a starter is unavailable |
| `waiver_depth_extra_rounds` | 1 | How far past league-wide starters the waiver wire begins |
| `max_at_position` | QB/TE 2, K/DEF 1, RB/WR 8 | Hard roster caps |
| `late_round_only` | K 2, DEF 3 | Rounds from the end before these may be drafted |
| `bye_penalty_per_extra_starter` | 6.0 | Cost per starter beyond the allowance |
| `bye_starters_allowed_per_week` | 2 | Free starters on any one bye week |

### `engine` — upside

| Key | Default | Meaning |
|---|---|---|
| `upside_weight` | 0.40 | How much bench ceiling counts |
| `projection_cv` | RB 0.32 … K 0.15 | Per-position volatility |
| `projection_sd_floor` | 12.0 | Minimum uncertainty |
| `rookie_upside_boost` | 0.35 | Extra spread for `years_exp ≤ 1` |
| `backup_upside_boost` | 0.45 | Extra spread for `depth_chart_order ≥ 2` |
| `market_disagreement_weight` | 0.5 | How much ADP spread widens the distribution |
| `position_value_multiplier` | all 1.0 | Explicit positional lean; 1.0 trusts the projections |
| `injury_multiplier` | see above | Projection discount by injury status |

### `ui` and `cache`

| Key | Default | Meaning |
|---|---|---|
| `ui.poll_seconds` | 3 | Pick-feed refresh rate |
| `ui.top_n` | 12 | Recommendations shown |
| `ui.color` | true | ANSI colour |
| `cache.players_ttl_hours` | 24 | Player file (~14 MB) |
| `cache.projections_ttl_hours` | 6 | Projections |
| `cache.adp_ttl_hours` | 6 | ADP |

---

## Tuning recipes

Change one thing at a time and re-run the offline mock — a 20-second loop beats
sitting through a live draft.

| You want | Change |
|---|---|
| More running backs, earlier | `roster_targets.RB` → 6 |
| RB depth to matter more | `starter_miss_rate.RB` → 0.28 |
| More gambling late | `upside_weight` → 0.6 |
| Safer, higher-floor rosters | `upside_weight` → 0.2 |
| A blunt positional lean | `position_value_multiplier.RB` → 1.10 |
| Bye weeks taken seriously | `bye_penalty_per_extra_starter` → 12 |
| Faster refresh on a short clock | `sims` → 300 |
| A wider board | `top_n` → 18, `candidates` → 20 |
| Two quarterbacks | `roster_targets.QB` → 2 |

---

## Bringing your own projections

Drop a CSV at `data/my_projections.csv`:

```csv
name,position,points
Bijan Robinson,RB,331.4
Brock Bowers,TE,253.5
```

A `player_id` column takes priority over `name` when present. Matching is
case- and punctuation-insensitive and handles suffixes (Jr., III).

This is the intended place for **O-line quality, coaching scheme, strength of
schedule**, or anything else you trust more than a consensus number. The
scarcity engine does not care where points come from, so better inputs here
improve every recommendation downstream.

The count is confirmed at startup: `using 214 projections from data/my_projections.csv`.

---

## Command line reference

| Flag | Meaning |
|---|---|
| `--username NAME` | Your Sleeper username; enables draft and slot auto-detection |
| `--draft-id ID` | Target a draft directly |
| `--slot N` | Force your draft slot (1-indexed) |
| `--season YYYY` | Override the season |
| `--sims N` | Monte Carlo iterations this run |
| `--top N` | Recommendations to show |
| `--once` | Render once and exit |
| `--mock` | Offline rehearsal draft |
| `--mock-slot N` | Your slot in the offline mock |
| `--mock-delay S` | Seconds between offline mock picks |
| `--list` | List your leagues, drafts, and mock drafts |
| `--no-auto-settings` | Trust `config.json` over the draft's own settings |
| `--no-color` | Disable ANSI colour |

### Common invocations

```bash
python draft.py --list --username YOUR_NAME
```

```bash
python draft.py --mock --mock-slot 7
```

```bash
python draft.py --draft-id 1234567890 --username YOUR_NAME
```

```bash
python draft.py --draft-id 1234567890 --slot 7 --sims 300
```

---

## Project layout

```
draft.py              CLI, live polling loop, offline mock draft
config.json           league + engine settings
README.md             this file

src/
  netcache.py         stdlib HTTP with a TTL disk cache and SSL workaround
  sleeper.py          Sleeper API client (players, projections, drafts, picks)
  adp.py              FantasyFootballCalculator ADP + name matching
  pool.py             joins everything into the unified player pool
  league.py           snake math, replacement levels, draft-settings detection
  survival.py         availability probability
  upside.py           projection uncertainty, option value, waiver levels
  lineup.py           lineup optimiser, roster legality rules
  simulate.py         Monte Carlo and the objective function
  recommend.py        ranking, cost-of-waiting, candidate selection
  board.py            terminal rendering

tests/test_engine.py  45 tests, no network required
data/                 gitignored: API cache + your projections CSV
```

Roughly 2,900 lines.

### Module responsibilities

| Module | Owns |
|---|---|
| `netcache` | Every outbound request. TTL cache, stale-on-failure fallback, the OpenSSL strict-verification workaround |
| `sleeper` | Endpoint shapes. Degrades to `{}` if the undocumented projections endpoint moves |
| `adp` | Name normalisation. Currently matches **256/256** FFC players |
| `pool` | The join, injury discount, projection backfill, CSV override, risk model |
| `league` | Snake pick arithmetic, endogenous flex allocation, reading settings off a draft |
| `survival` | The conditional normal CDF |
| `upside` | Bachelier option value, per-player sd, waiver levels |
| `lineup` | Optimal lineup fill, position caps, must-fill |
| `simulate` | Flat index-based arrays and the hot scoring loop |
| `recommend` | What the user actually sees ranked |
| `board` | ANSI output. No curses, no TUI library — it must start instantly and never crash |

---

## Tests

```bash
python -m unittest discover -s tests
```

```bash
python -m unittest discover -s tests -v
```

45 tests, no network, ~0.04s.

| Class | Covers |
|---|---|
| `TestSnakeMath` | Pick numbering, slot inversion, every pick assigned exactly once |
| `TestSurvival` | Monotonicity, conditioning, bounds |
| `TestLineup` | Flex assignment, unfilled slots, bye detection, roster needs |
| `TestDraftRules` | K/DEF timing, position caps, must-fill |
| `TestReplacement` | Levels, flex allocation, deeper leagues lower replacement |
| `TestDraftSettingsDetection` | Reading a differently-shaped mock, superflex refusal |
| `TestCandidateDiversity` | One position cannot own the board |
| `TestExpectedBest` | Expected best surviving player |
| `TestUpside` | Option value, VAL+UP reconstruction, stud vs flyer profiles |
| `TestRosterConstruction` | Depth where waivers are barren, targets, backup lifting a starter |
| `TestSimulation` | Full legal rosters, no free candidates, scarcity preference |

Two guard the bugs that actually occurred:
`test_candidate_occupies_a_pick_not_a_freebie` and
`test_simulation_completes_a_full_legal_roster`.

---

## Data sources

| Source | Provides | Auth | Cache |
|---|---|---|---|
| `api.sleeper.app/v1/players/nfl` | Names, teams, depth chart, experience, injury | None | 24h |
| `api.sleeper.app/projections/nfl/{season}` | Season projections | None | 6h |
| `api.sleeper.app/v1/draft/{id}/picks` | Live pick feed | None | Never |
| `fantasyfootballcalculator.com/api/v1/adp` | 14-team ADP, stdev, bye weeks | None | 6h |

All read-only and unauthenticated. The Sleeper projections endpoint is
**undocumented** — if it moves, the pool falls back to ADP-derived estimates and
keeps running.

Bye weeks come from FFC, not Sleeper: Sleeper's `bye_week` field is null this
time of year. Since bye is a team property, one FFC row covers every player on
that team.

---

## Troubleshooting

**`CERTIFICATE_VERIFY_FAILED`** — already handled. Python 3.13+ enables OpenSSL
strict X.509 checks that reject some Windows trust-store CAs.
`src/netcache.py` clears only that flag; certificate and hostname verification
stay on.

**`no Sleeper user named 'x'`** — use your Sleeper *username*, not your display
name or email.

**Mock draft not in `--list`** — mocks belong to no league and use a separate
endpoint, which `--list` already queries. If it is still missing, copy the id
from the draft room URL.

**`slot not set yet`** — normal before the order is drawn. It resolves itself,
or pass `--slot N`.

**Settings look wrong** — `--no-auto-settings` forces `config.json`.

**Slow first run** — the 14 MB player file. Cached for 24h afterwards.

**Slow between picks** — lower `sims`, or pass `--sims 300`.

**Board too wide** — widen the terminal or lower `top_n`. It targets ~110
columns.

**`UNSUPPORTED: superflex`** — the engine assumes a standard 1-QB lineup and
refuses rather than mis-modelling. Use a 1-QB league.

**Stale data** — delete `data/cache/`; it rebuilds on the next run.

---

## Limitations

- **Sleeper only.** ESPN and Yahoo would need a different pick feed; everything
  downstream would be reused unchanged.
- **Snake drafts only.** Auction is refused with a warning.
- **No superflex or IDP.** Superflex would change QB replacement level
  enormously, so it warns instead of approximating.
- **No in-season advice.** Draft only — no lineup setting, waivers, or trades.
- **No O-line, coaching scheme, or strength-of-schedule modelling.** These need
  paid or hand-built data, and consensus projections already price most of it
  in. Use the CSV override to supply your own view.
- **Opponents are modelled by ADP**, not by their roster needs. Good in
  aggregate, imperfect for any single opponent.
- **The live pick feed has not been exercised against a real draft.** Every
  other endpoint is verified live. Run a Sleeper mock first — that is exactly
  what it tests.

---

## Design decisions

**Zero dependencies.** A bare Python install minutes before a draft is not when
you want to discover `pip` is broken. `math.erf` covers every distribution
function needed; the Monte Carlo is fast enough in pure Python with flat
index-based arrays.

**Terminal, not a web page.** It starts instantly, survives anything, and sits
beside the draft room without fighting for a browser tab.

**Sleeper `player_id` as the join key.** It is the same key used by the
projections feed *and* the live pick feed, so those joins are exact. Only ADP
needs name matching — currently 256/256.

**Flex allocated endogenously.** Rather than assuming an RB/WR/TE flex split,
the projections decide. In this league it comes out 25 WR / 3 RB, which is why
receivers price higher here than generic rankings suggest.

**The objective is risk-adjusted, not raw points.** With only 5 bench spots, a
fourth good RB you can never start is nearly worthless — but a third RB who
covers an injury is worth a great deal, because the alternative is a 65-point
waiver back. Expected points alone cannot see that difference.

**Bench options strike against your own starter.** The single most important
detail in the model. Pricing bench players against the waiver wire made a backup
QB look like a 130-point steal and burned a bench spot every draft.

**Simulation only near your pick.** Running 800 iterations on all 210 picks
would be wasted work. Inside 12 picks it runs; outside, the board still shows
live VOR, survival, and cost-of-waiting instantly.

---

## License

[MIT](LICENSE) — © 2026 Mitchell Curtis Richard Walker.

That covers **this code**. It does not cover the data it reads: Sleeper and
FantasyFootballCalculator each set their own terms for their APIs, and this
tool is built for personal use against them. No data from either is
redistributed in this repository — everything is fetched at runtime and cached
locally under a gitignored `data/`.
