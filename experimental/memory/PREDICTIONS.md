# Registered predictions, written BEFORE the GPU run

A prediction recorded after the result is not a prediction. These are committed first so they
can be scored honestly, including the ones that turn out wrong.

Reasoned from the mechanism, not from precedent. The prompt that produced them was a fair
criticism: I had been using other people's published failures (Grazzi Fig 4, "representable does
not imply learnable") as a prior, which predicts nothing about *our* design.

---

## The structural insight the predictions hang on

**Composition is only NECESSARY when chain depth exceeds layer count.**

A 2-layer model solves a 2-hop chain with no `S·S` at all: layer 1 retrieves the first edge,
layer 2 the second. Layered retrieval is a *shallower credit-assignment path* than a pointer
chase, so gradient descent takes it whenever it is available.

This makes `gpu_run.py` as specified mostly untestable for our claim. At `NL=2` its cells are
`mqar-4` (1 hop) and `chain-d2` (2 hops) — neither requires composition. Only `chain-d3` forces
a single hop of it. We would have spent the last account on cells where the mechanism cannot
appear, and read the null as refutation.

---

## P1 — tied ≈ untied at depth ≤ layers; the gap opens past it and widens

At `NL=2`: no separation on mqar-4 or chain-d2. First separation at d3. Clear by d4–d6.

*Falsified if:* tied wins at depth 2, or fails to separate by d6 while `lag_cos` is healthy.
The second case means the mechanism does not survive training and we should say so and stop.

### CORRECTION, within an hour of writing the above

P1 as first stated implies something I had not followed through: **if layered retrieval
substitutes for composition, then at SOTA depths of 48–92 layers composition is essentially never
NECESSARY.** A 92-layer model has a 92-hop budget from layering alone, and almost no real task
needs 50 hops. "Multi-hop from one layer" is therefore NOT a capability claim against a deep
model, and I stated it as if it were.

What survives, and it is better than that concession suggests: **hop budgets MULTIPLY across
layers.** Layer 1 with `n_read=n` resolves n hops and writes the result into the residual stream;
layer 2 resolves n more from there. So:

    our budget      n_layers x n_read
    their budget    n_layers x 1

At matched depth we get `n_read` times the hops. To reach 48 hops they need 48 layers; we need 12
at `n_read=4`. **That is a ~4x parameter claim, and it is testable at small scale**: 2 layers at
`n_read=4` against 8 layers at `n_read=1`, matched params, depth swept.

So the claim is efficiency, not capability, and the experiment that supports it is a
LAYERS-vs-READS trade at fixed parameter count — not a win over a depth-starved baseline.

### How `NL=2` distorts everything currently on record

- **Flatters us.** A 2-layer GDN is a weak baseline; a multi-hop win at depth 2 may vanish
  against a deep one.
- **Penalises us.** `ceiling_sweep` found only mqar-2 readable at 2 layers, so we are forced to
  test in a regime where nothing discriminates. Those same cells are trivial at 48 layers.
- **Invalidates the bpc comparison in both directions.** 1.4666 at 25M is a small-model number;
  frontier bpc comes from depth, width and data.
- **May hide P4.** Head specialisation has only `H=2` heads per layer to work with at this size.

The reason we are at 2 layers is budget, not design. That is fine for a mechanism check and
disqualifying for a claim about SOTA.

## P2 — composition will NOT emerge from language-model loss

Next-token prediction on text almost never requires multi-hop *within* a sequence; local context
carries it. The machinery therefore receives near-zero gradient signal from bpc.

This retroactively explains the flat mx1/mx2 results without invoking "it does not work" — those
were LM runs, and LM loss does not pay for composition.

*Consequence:* our bpc stays at parity with GDN indefinitely, and **that is not evidence against
the architecture**. Judging this by enwik8 is judging a forklift by its top speed.

## P3 — the tap stays near-uniform and the conv does the lag

Gradient descent has two routes to a lag. The conv has `4d` parameters shaping percepts directly;
the tap has 4. The conv wins.

*Prediction:* final `tap_max` ≈ 0.25–0.40 while `lag_cos` ≈ 0.8+. Mechanism working, classic
diagnostic failure — which is why the diagnostic was already changed from `tap_max` to `lag_cos`
(see conv_vs_tap.py: a flat tap composes 0.565 with conv and 0.011 without).

*Corollary:* a `conv=False` arm should force the tap AND raise composition, since conv_vs_tap
measured 0.832 without conv against 0.667 with it at an identical sharp tap.

## P4 — per-head decay goes bimodal

The model controls decay per head through `a_proj(z)` and the gradient for it exists. Composition
needs slow heads (it always holds the longest-dated claim); local work needs fast ones.

*Prediction:* the per-head `dec` histogram separates into two clusters after training. That split
is the mechanism's signature and is more informative than accuracy. `gpu_run` already logs
`dec>=0.95` at init and at the end for this reason.

## P5 — degradation is ordered by distance-to-referent, not uniform

Decay taxes age and capacity compounds it (theory_vs_dynamics: the same edge scores 0.460 written
first vs 0.796 shuffled at dec=0.80). The oldest edge in a chain fails first.

*Prediction:* error rate correlates with referent distance, rather than being flat across it.

---

## What the run should therefore do

1. **Make depth the primary axis, not a cell.** Chains at d2/d3/d4/d6 with `NL=2` fixed. The
   prediction is a gap that *opens* at d3 and widens — a trend is much harder to fake than a cell.
2. **Stop expecting the win on bpc** (P2). Train on tasks that demand composition, or mix them.
3. **Add a `conv=False` arm** (P3) — the cleanest test of whether the tap can learn.
4. **Claim the right thing.** The differentiator is not O(1) inference; all 38 models in `fla`
   have that. It is that **one layer with n reads does what n layers do**, at a fraction of the
   parameters. That is a parameter-efficiency claim, testable at matched params.

## Scoring

| | outcome | notes |
|---|---|---|
| P1 | **mechanism SUPPORTED, headline NOT YET** | clean in the controlled pair; the NL8 comparison is confounded, and the ceiling was undertrained |
| P1b | **UNRESOLVED** (was CONFIRMED) | v4's gap widens +0.003 → +0.293 → +0.383, but on a task with a query-free shortcut a model can use *partially* — see the audit note below. The `lag_cos` corroboration is one seed on a 0.78 init baseline. Rerun on `dmode=chain` decides it |
| P2 | not yet | needs an LM run alongside a composition task |
| P3 | **WRONG in its strong form** | the tap DOES learn — see below |
| P4 | **not testable at this config** | `d=128, head_dim=64, NL=2` is only **4 heads total** — "50%/50%" is two of four, which cannot establish a bimodal distribution |
| P5 | not yet | needs per-sample error vs referent distance |

> **CAVEAT ADDED AFTER THE eps ABLATION — the accuracy half of this scoring is weak.**
> That ablation measured chain accuracy at **0.074 on seed 0 and 0.496 on seed 1, at identical
> settings** — a 6.7× spread, and a bimodal one ("didn't find it" vs "partly found it") rather
> than noise about a mean, because 600 steps at `d=64` sits at the threshold of learning the task
> at all. Every number below is single-seed. What survives that: `lag_cos` and `tap_max`, which
> reproduced **identically to three figures** across conditions in the ablation. What does not:
> any conclusion resting on the accuracy column. The P3 verdict happens to rest on `tap_max`, so
> it stands — but it stands on the sturdy half, and I originally wrote it as though the whole
> table were equally solid.

> **AUDIT, 2026-09-15 — three corrections to how the table above and the P1b section read.**
>
> 1. **The v1–v4 chain task leaks.** Every distractor is a lone edge whose source is never a
>    target, while the answer's source always is. So "the one sink whose source is also a target"
>    scores **1.000 at every depth** without reading the query. A cheaper form of the same leak —
>    "the sink whose source token occurs twice" — is a frequency count needing no hops at all.
>    **This DOES put P1b back in question.** The first version of this note argued it could not,
>    because an arm using the shortcut would score 1.000 everywhere. A second review showed that
>    argument is unsound: use is not all-or-nothing. A noisy version of the cue can lift r1 from
>    0.25 to 0.586 at depth 3, and candidates multiply with depth, so r4 computing the cue more
>    reliably than r1 would also produce a gap that widens with depth. v4 is contaminated to an
>    unknown degree, and P1b is **UNRESOLVED** until the rerun on the fixed task. Fixed as `make_chain(dmode='chain')` — distractors are whole chains — now the
>    `gpu_run` default, with a query-swap eval that tests whether a model reads the query at all.
> 2. **The `lag_cos` corroboration below is ONE SEED on a HIGH BASELINE.** `gpu_run` printed
>    `lag_cos` from the last seed while averaging accuracy. An untrained conv block reads
>    **0.782**, and one with its tap forced onto the WRONG lag still reads 0.456, because SiLU
>    percepts share ~0.47 cosine. So r1's 0.822 at depth 2 is init-level, and "abandons the
>    transition-operator structure" is not established — 0.193 says something changed, not what.
>    Replaced by `probe()`: task batch, every layer, lag-1 AND lag-2, at init and end, pooled.
> 3. **Every decay figure (P4) omitted `a_proj(z)`**, and `A_log`, `dt_bias` and the tap were
>    weight-decayed despite `_no_weight_decay`. Both fixed.
>
> Also fixed: the ceiling gate read the last transformer listed (NL8) instead of the best, and a
> rebuilt bundle would never have run the NL2 ceiling.
>
> 4. **The transformer "ceiling" cannot do single-hop recall under this harness** (second review,
>    CPU, d=64). NL2 on MQAR with 4 pairs: 0.166 after 3000 steps, below the 0.25 chance among
>    values, while tied NL2 r1 reached 0.811 in 1000 steps on the same cell; d=128 and NL4 did not
>    rescue it. Every v3/v4 ceiling row is therefore void as a bound. `MODE=ceiling_triage` in
>    `gpu_run.py` is the test that must pass before any ceiling is quoted again.

## P1b — scored CONFIRMED after v4; now UNRESOLVED (see the audit note above). Kept as written.

v4, depths 2/3/4, 3 seeds, 3000 steps. `NL2 r1` and `NL2 r4`: identical parameters (341,947),
identical layers, differing **only** in `n_read`.

        depth        2        3        4      (sink floor 0.333 / 0.250 / 0.200)
        NL2 r4     1.000    0.879    0.594
        NL2 r1     0.997    0.586    0.211
        gap       +0.003   +0.293   +0.383

**Monotonically widening, as registered.** And the shape is the predicted one rather than merely
an increase: the gap is ~zero exactly where budget 2 suffices (depth 2), opens when it is one hop
short (depth 3), and widens again when it is two short (depth 4). At depth 4 `NL2 r1` reaches
**0.211 against a floor of 0.200** — it has collapsed onto the guess-among-sinks heuristic, which
is what a model that cannot reach the answer should do.

**`lag_cos` corroborates independently, and this is the part I did not predict:**

        NL2 r4   0.929 -> 0.959 -> 0.993     sharpens as the task demands more hops
        NL2 r1   0.822 -> 0.740 -> 0.193     collapses when the budget runs out

The budget-2 arm does not degrade gracefully; at depth 4 it **abandons the transition-operator
structure entirely**. Accuracy and mechanism moving together, in opposite directions for the two
arms, is much harder to explain by anything other than the stated mechanism than either signal
alone would be.

### What this does and does not license

**Does:** extra reads substitute for hop depth, at equal parameters and equal layers. That is the
mechanism claim, and it is now a trend across three depths rather than a single point.

**Does NOT:** any statement about beating a transformer, or parameter efficiency. The ceiling in
this run was still `NL8` at 3000 steps (0.229 / 0.163 / 0.087) — undertrained and depth-mismatched.
The fixes for that (`CEIL_MULT`, a depth-matched `ceiling NL2`) were committed in 08fe8b3 **after**
this run was pushed, so they are not in it; the log's `ARMS=ceiling_NL8,...` confirms as much.

**Caveats that survive:**
- **The adjacency prior** (1445999): `make_chain` puts each target immediately after its source,
  which is exactly a lag-1 tap's prior, so absolute scores flatter us. Both arms receive it
  equally and adjacency buys one hop not four, so the *gap* is unaffected.
- **Depths 2 and 3 are not an independent reproduction of v3.** Same seeds, deterministic
  training — they could not have come back differently. Only depth 4 is new evidence.
- **Three seeds**, on a task whose seed variance once spanned 0.074 to 0.496. The effect here
  (+0.383) is far larger than that spread, which is why I believe it — but seeds 3,4,5 would
  settle it properly.

## P1b — as registered BEFORE v4 ran

v3 gave one clean point: at identical parameters and layers, `n_read=4` beat `n_read=1` by
**+0.293** at depth 3 and by nothing (+0.003) at depth 2. One point is a coincidence; a trend is
a result. v4 adds depth 4 and drops the confounded `NL8` arm.

**P1b — the `NL2 r1` → `NL2 r4` gap WIDENS at depth 4, beyond the +0.293 seen at depth 3.**

The reasoning: `NL2 r1` has hop budget 2. At depth 2 that is exactly sufficient (gap ~0). At
depth 3 it is one short (gap +0.293). At depth 4 it is two short, so it should degrade further
while `NL2 r4` (budget 8) still has headroom. Expected shape: `NL2 r1` falls below its 0.586,
`NL2 r4` holds near or a little under its 0.879.

*Falsified if:* the gap narrows, or both arms collapse together (which would say depth 4 is
simply unlearnable at this scale, the v2 failure again, not a statement about reads), or `NL2 r1`
holds up — which would mean something other than hop budget explains depth 3.

**The honest caveat:** this task's seed variance has already burned us once (0.074 vs 0.496 at
identical settings). Three seeds is thin for an effect of this size, and the depth-2 and depth-3
points must reproduce alongside depth 4 for the trend to mean anything. If depth 3 comes back far
from 0.586/0.879, the variance is larger than the effect and the whole comparison needs more
seeds rather than more depths.

### P1 v3 — the first run where anything learned, and it splits into two verdicts

`saryu-p1` v3, depths 2 and 3, 3 seeds, 3000 steps, d=128, head_dim=64, NENT=32, ratio=1.0:

        depth 2 (sink floor 0.333)          depth 3 (sink floor 0.250)
        ceiling  NL8      0.150             ceiling  NL8      0.107
        tied NL8 r1       0.974             tied NL8 r1       0.374
        tied NL2 r4       1.000             tied NL2 r4       0.879
        tied NL2 r1       0.997             tied NL2 r1       0.586

**THE MECHANISM IS SUPPORTED, in the one comparison that is fully controlled.** `NL2 r1` and
`NL2 r4` have **identical parameters (341,947), identical layers, identical everything except
`n_read`**:

        depth 2   0.997 -> 1.000   (+0.003 - no gap, because budget 2 already suffices)
        depth 3   0.586 -> 0.879   (+0.293 - reads become necessary once depth exceeds layers)

That is exactly the predicted shape: extra reads buy nothing when the hop budget is already
adequate, and buy a great deal the moment it is not. Nothing but `n_read` differs, so nothing
else can explain it. `lag_cos` agrees — the high-read arm has the most transition-operator-like
map (0.929 / 0.959 against 0.822 / 0.740).

**THE HEADLINE IS NOT EARNED, because the NL8 comparison is confounded.** `NL8 r1` scored 0.374
at depth 3 — *worse than* `NL2 r1`'s 0.586, despite hop budget 8 against 2. The budget model
cannot produce that ordering. The plain explanation is that an 8-layer model trains worse than a
2-layer one in 3000 steps, so `NL8 r1` measures **trainability, not capacity**. Which means the
attractive line — "341,947 params beat 1,340,035" — is comparing against a baseline that is
weak for the wrong reason. A win with fewer parameters is only unimpeachable if the bigger model
was given a fair chance, and it was not.

**THE CEILING FAILED, AND THE GATE FIRED WRONGLY.** The transformer scored 0.150 and 0.107 —
below the sink floor — so `gpu_run` printed "nothing at this depth ranks". That verdict is wrong
here, and the reason matters: the gate assumes the transformer is an *upper bound*, so its
failure implies the task is unsolvable. But our arms hit **1.000**, which proves the task is
solvable and the harness is sound. So the ceiling's failure is a fact about the ceiling arm, not
about readability — almost certainly lr=3e-3 with 8 layers and no warmup, which our blocks
tolerate and a deep transformer does not. An untuned baseline is not a ceiling, and a gate built
on one will keep misfiring.

**Also worth noting:** fp16 skipped only 2–4 steps out of 3000 in every arm, so the new
GradScaler counter says overflow is negligible rather than silently eating the schedule.

**THE ABSOLUTE NUMBERS ARE FLATTERED, and the relative one is not.** `make_chain` emits each edge
as `seq += [a, b]`, so a target always immediately follows its source — which is precisely the
prior a lag-1 tied map has. With a sharp tap, `k_t` is the previous percept and `v_t` the current
one, so **one hop is read off the surface form without consulting memory at all**.
`smoke_corrected.py` warned about exactly this for MQAR and I did not carry the warning across to
`chain`, then read 1.000 as composition.

So "we beat the transformer 1.000 to 0.229" is doubly unsafe — the baseline was undertrained and
depth-mismatched, *and* our score is partly a gift from the task's surface form. What is untouched:
`NL2 r1` vs `NL2 r4` **both enjoy the same gift**, differ only in `n_read`, and adjacency buys one
hop rather than four. The +0.293 at depth 3 still means what it says.

### P1 v2 — the earlier attempt, UNANSWERED because the cell was unlearnable

First GPU run of the layers-vs-reads trade (`saryu-p1` v2: depth 6, 3 seeds, 3000 steps, d=128,
head_dim=64, NENT=32, ratio=1.0):

        SINK FLOOR        0.143      chance 0.031
        ceiling  NL8      0.049      1,324,067 params
        tied NL8 r1       0.028      1,340,035
        tied NL2 r4       0.063        341,947
        tied NL2 r1       0.060        341,947

**Every arm is BELOW the sink floor, the transformer ceiling included.** Nothing learned the task.
This is exactly the case `gpu_run`'s own docstring pre-registered as *"the one outcome that is NOT
a result"* — everything at the floor says the cell is too hard at this scale, which is
indistinguishable from the mechanism failing. So P1 is not refuted and not supported; it was not
tested.

**The error was mine and it was avoidable.** I picked depth 6 because it is the most
*discriminative* cell — `NL2 r1` has hop budget 2 and provably cannot reach — without checking it
was *learnable*. `ceiling_sweep` had already said so: at 6000 steps even **mqar-3**, a one-hop task
with three pairs, reached only 0.292. A 6-hop chain among 12 shuffled edges was never going to
train in 3000 steps at d=128. I optimised for separation and dropped readability, having written
the warning about precisely that myself.

**The fix is a readable depth, not a bigger model.** With `NL=2`, depth 3 already exceeds the
layer count, so `NL2 r1` is still budget-limited while `NL2 r4` is not — same discrimination, far
more learnable. Depths 2 and 3 go in together so the ceiling reports which cells are readable
rather than my guessing a second time.

**What did survive, from a run where nothing learned the task — so hold it loosely:** `lag_cos`
after training came out **0.854 / 0.903 / 0.760**, so the tied map really is a transition operator
under SGD rather than only at init. And P4's decay split showed 50%/50% with a 0.72 spread. Both
are suggestive; neither is worth banking from a null cell.

**And the infrastructure worked**, which is the other half of what this run bought: the fused path
went ACTIVE and matched the loop to rel 7.97e-04 under autocast, the dtype fix that killed v1 held,
and the bundle guard confirmed `RefNet mixer=Attn, ParityNet mixer=ParityBlock, distinct=True`.

### P3 scored, from the first glimpse (600 steps, `d=64`, `NL=2`)

Only the **content-only** arm is admissible here. The first glimpse ran with a bug where the query
was never bound while keys and values were, so every arm using state or time could not retrieve by
construction and their numbers are void (fixed in 357b09a). Content-only has no binding, so its
query path was always correct and its numbers stand.

        cell        acc     lag_cos   tap_max     (tap starts at 0.250, uniform)
        mqar-2      1.000   0.403     0.601
        chain-d2    0.277   0.828     0.443

**P3 predicted `tap_max` ≈ 0.25–0.40 with `lag_cos` ≈ 0.8+** — the tap staying flat because the
conv has `4d` parameters against the tap's 4 and would win the race to supply the lag.

The tap moved to **0.44 and 0.60**, well clear of its uniform 0.25 start. So gradient descent does
sharpen it, and my parameter-counting argument was wrong: a 4-parameter softmax sitting directly
on the quantity the loss depends on gets a cleaner gradient than a wide conv that has to discover
the same lag through a shaped kernel.

What survives from P3 is the *diagnostic* half, and `chain-d2` shows it plainly: `tap_max` 0.443
with `lag_cos` 0.828. The tap is nowhere near the 0.90 sharpness that `sharpness_sweep` says
composition needs, yet the map is still strongly a transition operator — exactly why the metric
was switched from `tap_max` to `lag_cos`. Reading `tap_max` alone would have called this a failure.

The two cells disagree about direction (`mqar` sharpens more but has *lower* `lag_cos`), which
suggests the tap and the conv trade off against each other per task rather than one always winning.
Not resolved; the `conv=False` arm in `gpu_run` is the test.

### Two things the glimpse showed that were not predicted

**Content-only hit 1.000 on mqar-2 at 600 steps while the transformer ceiling sat at 0.336.**
`ceiling_sweep` needed 6000 steps for a transformer to clear that cell. Do NOT bank it:
`smoke_corrected.py` already warned that MQAR places keys immediately before values, which is
precisely the lag-1 tied map's inductive bias, so this is likely that artefact rather than a
general recall advantage.

**On chain-d2, content-only scored 0.277 against a sink floor of 0.333.** It is *below* the
guess-among-sinks heuristic — i.e. not doing the task at all. Without the floor added in 3d8db2a
this would have looked like modest partial success.

If P1 and P4 both land, the architecture is real and the path is clear. If P1 fails at d4–d6 with
a healthy `lag_cos`, the mechanism does not survive gradient descent — and that is a result worth
having, not a failure to bury.

## References

- Grazzi et al. 2024. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*. arXiv [2411.12537](https://arxiv.org/abs/2411.12537).
