# SPEC: the shift vs Gated DeltaNet on MQAR and multi-hop recall

Written 2026-09-14, before any code, so the predictions cannot be tuned to the result.

## The claim being tested

Every reference implementation in `fla` — verified by reading all eight sources
(`delta_net`, `gated_deltanet`, `comba`, `mesa_net`, `deltaformer`, `gated_deltaproduct`,
`precond_gated_deltanet`, `kda`) — derives keys and values from **separate projections of
the same token**, and none applies a lag to the key path. That is `compose_probe`'s
**untied** construction:

    untied   k_t = W_k x_t ,  v_t = W_v x_t      ->  S k_a = v_a, but S·S is meaningless
    shifted  k_t = phi(x_{t-1}), v_t = phi(x_t)  ->  S is a TRANSITION OPERATOR, S^d walks d hops

Measured, pure numerics, no training (`compose_probe.py`, run 2026-09-14):

    construction   1 hop    2 hops   3 hops   depth 7
    untied         1.000    0.120    0.155    0.135     <- chance is 1/8
    shifted        1.000    1.000    1.000    1.000

CLAIM: a delta-rule state whose keys and values share one map composes, and the whole
DeltaNet family does not. Corroborating detail: our 4-hop 1.000 result came from a
ONE-LAYER model (`LAYERS=1` default in stage_c), which the one-hop-per-layer account says
an untied block cannot do.

NOT YET ESTABLISHED, and not to be claimed until it is:
  - Novelty in the LITERATURE. STILL OUTSTANDING as of 2026-09-14. Absent from `fla` is
    not absent from the field, and I will not claim novelty without checking. The arXiv
    search API (export.arxiv.org) has refused three attempts with "Rate exceeded" - the
    block is PER-IP and self-inflicted: I fired four parallel queries at it, which is the
    same mistake the earlier subagents recorded hitting. Cooldown is long; retry later or
    from another network. The arxiv.org/list/ pages are NOT blocked but are chronological
    (1,004 cs.CL entries in Sept 2026 alone), so they cannot answer a keyword question.
    What IS established: all eight fla layer implementations use separate k_proj/v_proj
    with no lag on the key path. That bounds the reference IMPLEMENTATIONS, not the field.
  - Any number on a SHARED benchmark. They publish MQAR and S-NIAH; we have none.
  - Survival under interference. Our own sweep: ONE distractor edge cost two-thirds of
    the accuracy (0.755 -> 0.229). Composition-by-algebra may not survive real load.

## The design problem, and the fix

A fixed lag of 1 only works when the task's node stride is 1. `stage_c`'s world is
`a R b ;` — stride 4 — and the battery's CHAIN is adjacent pairs — stride 2. A hard-coded
shift is therefore task-specific, which is not an architecture.

FIX: keep the maps TIED, let a SHARPNESS-CONSTRAINED tap LEARN the lag.

    p_t = normalize(phi(x_t))                       one shared percept map
    a   = softmax(logits / tau),  tau <= 0.3        4 taps, CONSTRAINED SHARP
    k_t = normalize(sum_j a_j * p_{t-j})            blend of recent percepts
    v_t = p_{t+1} in the write, p_t as read value
    q_t = normalize(sum_j a'_j * p_{t-j})

Values stay legal keys because both live in phi's image — that is the property that makes
composition work. A hard shift is the special case a = [1,0,0,0] (tap j=0 weights p_t).

NOTE the indexing, which I got wrong once: a[0] selects p_t, a[1] selects p_{t-1}. The
hard shift is [1,0,0,0], NOT [0,1,0,0] — [0,1,0,0] is a stride-2 map and scores 0.000.

The tap MUST be constrained sharp; a free conv is not safe. See STEP 0 below for the
measured band.

STEP 0 — RUN 2026-09-14, RESULT: PASSED WITH AN AMENDMENT. A free conv is NOT safe.

`conv_tied_probe.py` (self-check row first: conv [1,0,0,0] is definitionally `shifted`
and reproduces its 1.000, so the rows are readable):

    untied                      1.000  0.090  0.120   <- chance 0.125, dies past one hop
    shifted                     1.000  1.000  0.995
    conv [1,0,0,0] == shifted   1.000  1.000  1.000
    conv [.9,.1,0,0] mild       1.000  1.000  0.955
    conv [.5,.25,.25,0] blurred 0.950  0.125  0.000   <- AT CHANCE
    conv [.4,.3,.2,.1] diffuse  0.915  0.025  0.000   <- BELOW chance

Composition is a property of a SHARP tap. A depthwise conv is free to learn a diffuse
kernel, and diffuse is exactly where the algebra dies — so "let a conv learn the lag" as
originally written here is unsafe.

`sharpness_sweep.py` then measured WHERE it breaks — softmax over 4 taps, temperature tau:

    tau   max tap weight   1 hop   2 hops  3 hops
    0.20     0.980         1.000   1.000   0.997
    0.30     0.903         1.000   0.990   0.980    <- usable
    0.50     0.711         1.000   0.973   0.737    <- degrading
    0.80     0.538         0.997   0.653   0.143    <- collapsed
    1.50     0.394         0.923   0.167   0.047

USABLE BAND: dominant tap weight >= ~0.90. That is NOT a knife edge, so a learnable tap is
viable — but it must be CONSTRAINED to stay sharp.

AMENDED DESIGN: replace the free depthwise conv on the key path with a TEMPERATURE-SHARPENED
SOFTMAX over 4 taps, tau <= 0.3 (or a straight-through hard argmax). Log the dominant tap
weight during training and treat any drop below 0.90 as a failure mode to detect, not a
free parameter.

(An earlier version of the probe reported all-zero conv rows. That was two bugs of mine —
an off-by-one in the tap index and a divide-by-zero leaving k as the zero vector, poisoning
S with NaN — caught only because the row that is definitionally `shifted` disagreed with
`shifted`. A separate anomaly, accuracy RISING with depth, was argmax on numerical noise
and vanished once vectors decayed below 1e-8 are excluded from scoring.)

## Arms

    gdn          fla GatedDeltaNet, unmodified          the incumbent
    shift        tied-map block above, matrix state     ours
    transformer  causal attention + RoPE                THE CEILING

The ceiling arm is not optional. Today the graded battery produced four voided runs
because no arm cleared the gate and I twice misread that as a broken task. Rule: if the
transformer does not exceed 0.90 at a cell, that cell is UNREADABLE and every other arm's
number at it is discarded, whatever it says.

## Tasks

1. MQAR (single hop) — PARITY CHECK. k1 v1 k2 v2 ... query k_i -> v_i.
   Grid: {8, 16, 32, 64} kv pairs x {256, 512, 1024} sequence length.
2. MULTI-HOP CHAIN — THE DISCRIMINATOR. Edges shuffled, query start node + depth.
   Grid: depth {1,2,3,4} x distractor ratio {0, 0.5, 1.0, 2.0}, GRADED.
   ratio=0 IS A SHORTCUT CONTROL ONLY (unique-sink solves it 100% with no composition).
   All claim-bearing cells use ratio >= 0.5.

DESIGN RULES, each one paid for today:
  - Depth and interference are SEPARATE axes. Today I varied depth at fixed edge count,
    which silently varied distractors, and read the artefact as a finding.
  - Distractors are a RATIO, not whole edges. Whole-edge granularity put the
    discriminative band BETWEEN two settings: 0 distractors -> everything saturates,
    1 distractor -> everything floors.
  - LEFT-pad; the query must be the last position. Right-padding made the model answer
    from a PAD token in `reason_battery`, a bug this repo had already documented once.
  - PAD id outside the entity range. It collided with entity 0.
  - `dist` must take enough distinct values to split, or report "not measured" rather
    than emitting a quotable nan.

## Scale and budget

d=256, 2 layers, head_dim 128 (NOT 41 — the reference family runs 128-256 and GDN-2's own
config is d_k=d_v=128, 262,144 state floats; my dh=41 in mx1/mx2 was far below the tested
grid). Matched parameter count across arms via binary search on width, as bench5m does.

GPU: `fla` is Triton-only; verified working on a Tesla T4 (torch 2.10, triton 3.6, fla
0.5.2, GatedDeltaNet forward+backward finite). GPU quota is limited.

3 SEEDS MINIMUM. Every threshold below is set above the measured seed spread, not guessed.

## Registered predictions

P1 PARITY ON SINGLE HOP. On MQAR, |shift - gdn| < 0.05 at every cell whose ceiling clears
0.90. The shift changes composition, not storage, so it should neither help nor hurt here.
  FALSIFIER: if shift LOSES to gdn on MQAR by >0.05, the tie has cost storage capacity and
  the design is a regression regardless of what multi-hop shows.

P2 ADVANTAGE ON MULTI-HOP. At depth >= 2 AND DISTRACTOR RATIO >= 0.5, shift - gdn >= +0.20.
  This is the whole claim. compose_probe says untied lands at chance past hop 1, so the
  gap should be large, not marginal. A small gap refutes the mechanism story even if
  positive.

  AMENDED 2026-09-14 AFTER A CPU SMOKE FOUND THE ORIGINAL CELL WORTHLESS. P2 was registered
  at "zero distractors". Measured: at ratio=0.0 the chain contains ONLY path edges, so the
  answer is always the unique node that never appears as a SOURCE. That heuristic is correct
  100.0% of the time and needs no composition at all:

      depth=2 ratio=0.0 : unique-sink heuristic correct 100.0%
      depth=2 ratio=0.5 : 0.0%
      depth=2 ratio=1.0 : 0.0%
      depth=2 ratio=2.0 : 0.0%

  The smoke duly showed the UNTIED control - which compose_probe proves cannot compose past
  one hop - scoring 0.987 at ratio 0.0. That was the leak, not a result. Any ratio >= 0.5
  kills the shortcut, so composition cells MUST use ratio >= 0.5 and the zero-distractor
  column is reported as a shortcut control, never as evidence for the claim.

P3 INTERFERENCE IS THE REAL LIMIT. Both arms degrade as the distractor ratio rises, and
the shift's advantage SHRINKS with it. Registered because our own data predicts it: one
distractor cost two-thirds of the accuracy. If the shift's margin holds flat under load,
that is a stronger result than P2 and I did not expect it.

P4 CEILING. transformer > 0.90 at depth <= 2 and low interference. Cells where it does
not are reported as unreadable and excluded.

## Reporting rule

Any difference inside the measured seed spread is "no effect detected", in either
direction, including when it favours us. Same rule that cut against our refractory arm in
Test B and against mx2 today.

## What this is and is not

This produces a CLAIM, on the incumbents' own benchmark. It does not produce a demo — a
model that generates readable text is the TinyStories path (verified: <10M params, fluent
multi-paragraph English, 2.14M stories, CDLA-Sharing-1.0, ~30h on one V100). Two
deliverables, two tracks; do not conflate them.

## References

- Yang et al. 2024a. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*. arXiv [2406.06484](https://arxiv.org/abs/2406.06484).
- Yang et al. 2024b. *Gated Delta Networks: Improving Mamba2 with Delta Rule*. arXiv [2412.06464](https://arxiv.org/abs/2412.06464).
- Hatamizadeh et al. 2026. *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*. arXiv [2605.22791](https://arxiv.org/abs/2605.22791).
- Siems et al. 2025. *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products*. arXiv [2502.10297](https://arxiv.org/abs/2502.10297).
