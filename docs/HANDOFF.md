# Handoff — 2026-09-22, end of session

Written to be picked up cold. Read this first, then `evidence/CLAIMS.md` (39 entries, over half
retracted — the retraction rate is the point).

**Goal and deadline:** a trained model by end of this week. Today is Tuesday.

---

## What the project can honestly claim, as of tonight

One sentence: **GRU-quality, parallel-trainable.**

| claim | status |
|---|---|
| ~8× fewer tokens than a parameter-matched transformer to a given loss | holds, 2 seeds, 5M params, ctx 128 |
| beats a GRU | **falsified.** A GRU with pre-norm and residuals ties us on final loss (2.190/2.193 vs 2.189/2.176) |
| solves state tracking where others cannot | **half retracted.** A GRU also solves S₃ at 1.000; the transformer gets 0.727. The complexity argument was always about attention |
| the chunk kernel is a real advantage over a sequential RNN | holds — GRU/Saryu step time 1.36× → 3.70× from ctx 128 to 2048, monotone |
| the matrix memory and gate lower bound help | **falsified** — a 2× efficiency penalty on tokens-to-loss |
| the 3:1 hybrid helps | no effect at ctx 128, and that is a fault in the experiment: at that length there is nothing to retrieve |

Three of four registered predictions in the efficiency experiment failed. The headline is
**recurrence vs attention in a small-data regime**, which is what the literature predicts and is
therefore the least surprising reading of the data rather than the most.

---

## THE OPEN DECISION, and it gates the training launch

We found the matrix memory was missing **controls, not capacity**, by reading Qwen3-Next and
Kimi Linear's source (not their papers):

- no output gate — both shipped models apply a gated RMSNorm to the delta-rule output
- every head starts at the **same** decay timescale; theirs start on a ladder from a randomised
  per-head `A_log`

Measured on an attention-mimicry surrogate (`evidence/results/delta_gate_decay.txt`, 2 seeds,
seed spread 0.034):

```
base         0.3816      alog alone   0.3822  (nothing)
ogate alone  0.3337      ogate+alog   0.2280  (40% cut, 4.5x the spread)
```

It is an **interaction**. That also explains why the three hand-installed timescale ladders in
`results/gate_timescales.txt` all collapsed — every one was the decay half with no gate.

**Both fixes are now in the model** (`e54f4f9` output gate, on by default; `4bad3f7` A_log decay,
opt-in via `mem_alog`). **But they are not what was measured.** The surrogate reached 0.228 with
the decay clamped at −3; the model clamps at `log(ALPHA_FLOOR) = −1`. The tighter floor is the
exact constraint the diagnosis says is limiting us.

**So, before launching training:**

1. Settle the floor. `ALPHA_FLOOR` is `exp(-1)`; the field uses `exp(-5)` (FLA's `lower_bound`
   default, GLM5-Next ships `linear_lower_bound = -5.0`). Ours is 5× tighter in log space. It is
   a one-line change plus a rerun of the chunk-vs-sequential exactness tests.
2. Re-run the efficiency arm with the corrected memory. The 2× penalty was measured on the
   **miswired** version; whether the fixed one earns its cost is untested.
3. Only then decide whether the shipped model carries the memory at all.

Training two sessions around a half-corrected component is the expensive mistake.

---

## Ready and unlaunched

- **GPU path proven.** First ever GPU run (`evidence/results/gpu_probe.txt`): on a T4 the
  transformer/Saryu ratio is 1.21–1.63× at batch 32, **better** than the 2.0–2.9× on CPU. The
  registered pessimistic prediction was wrong in our favour. Batch size is worth ~2× and is free.
- **Selected config:** d=512, 8 layers, ~23.7M params, B=32, T=512, chunk 32 → 430M tokens per
  9-hour session, 9.76 GB peak. Two sessions ≈ 860M tokens, ~36 tokens/param.
- **The real constraint is memory, not speed** — peak is 2–3× a transformer's, and we OOM where
  it fits. That is a kernel problem and it now caps model size.
- **Checkpoint/resume fixed and verified** (`5f9e124`). It had been restoring model, step and
  AdamW state but **not the batch generator**, so session two would have re-trained on exactly
  session one's batches — silently, with a healthy-looking loss curve. Muon's momentum was also
  never saved. Split moved to the standard enwik8 90/5/5, so bpc is now citable.
- **Kaggle tooling works.** `scripts/build_kaggle.py` bundles the repo and runs an entry script
  verbatim; auth verified; the probe ran end to end.
- **Multiple Kaggle accounts** exist, which dissolves the "one model or one comparison" tradeoff —
  run Saryu, a matched transformer, and the hybrid in parallel on separate quotas.

Still to build: arm support in `scripts/train.py` for the transformer and hybrid arms.

---

## Logged, deliberately not attempted this week

- **Kernel work.** `_recall_chunk` forms `exp(la−ref)` and `exp(−(la−ref))` separately and then
  masks; the surviving entries are bounded by 1 but the factors are materialised for every
  position first. FLA anchors at the midpoint of a 16-token sub-chunk instead. Also `kt_f` is
  avoidable even per-channel — the standard `k·exp(la_last − la_j)` is ≤ 1 elementwise. Together
  these are what would let the floor widen.
- **Distillation.** The cheap route does not apply to us: weight inheritance from a teacher buys
  **nothing** (0.384 vs 0.383, fully overlapping), plausibly because we L2-normalise q/k and the
  delta rule uses k as an erase direction. Conversion would need full end-to-end training. Note
  the corrected cell reaches 0.228, below the 0.25 threshold — worth revisiting after the floor.
- **Mechanising the "still running" guard.** Twice in one hour a running job was read as finished
  and reported wrongly. `MODE=table` rebuilds results from `runs/`; the table should refuse to
  print an arm whose job has not exited.

---

## Method notes that earned their place today

1. **Read the source, not the paper.** Papers describe the delta rule; the gated norm is in the
   code. Two "faithful" implementations in this repo's history were missing components their own
   references require.
2. **Verify second-hand readings.** Three of four claims about Qwen/Kimi were wrong when checked
   directly — including operation order, which was not cosmetic: gate-then-norm scored **1.035**,
   worse than a fitted linear map, and would have gone on record as "the output gate is harmful".
3. **Every automated READ needs a noise floor.** Four variants of one mistake today: a mean over a
   bimodal outcome manufactured a result; a threshold above the effect size hid one; a gap inside
   the seed spread was called a win; and the guard added to prevent it had a degenerate default at
   one seed.
4. **Check for a running package manager before touching its files.** Tonight's `claude` CLI
   repair: an interrupted `npm install` had left the shims renamed and the binary a 500-byte stub.
   I created a hardlink between two paths npm hardlinks from its own cache, while an install was
   live — inflating the binary from 146 MB to 207 MB of garbage and corrupting the cache entry.
   `Get-CimInstance Win32_Process` should have been the first command, not the tenth. Repaired
   with cache verify plus a clean reinstall; `claude --version` is 2.1.280.
