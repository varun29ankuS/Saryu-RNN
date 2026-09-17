# Saryu

**A recurrent language model whose state is moved by Householder reflections.**
**Open architecture research, built in India.**

[![Website](https://img.shields.io/badge/website-saryu-0f6d82.svg)](https://varun29ankuS.github.io/Saryu-RNN/)
[![Paper A](https://img.shields.io/badge/paper%20A-architecture-e08a1e.svg)](paper/main.pdf)
[![Paper B](https://img.shields.io/badge/paper%20B-quantised%20failure-8a6fd4.svg)](paper/paperB.pdf)
[![Weights: v0.1](https://img.shields.io/badge/weights-v0.1-orange.svg)](https://github.com/varun29ankuS/Saryu-RNN/releases/tag/v0.1)
[![Tests](https://github.com/varun29ankuS/Saryu-RNN/actions/workflows/tests.yml/badge.svg)](https://github.com/varun29ankuS/Saryu-RNN/actions/workflows/tests.yml)
[![Made in India](https://img.shields.io/badge/made%20in-India-ff9933.svg)](#about-this-project)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)

[Website](https://varun29ankuS.github.io/Saryu-RNN/) ·
[What the research found](#what-the-research-found) ·
[Why it is built this way](#why-it-is-built-this-way) ·
[Results](#results-character-level-enwik8) ·
[Quickstart](#quickstart) ·
[Paper A](paper/main.pdf) ·
[Paper B](paper/paperB.pdf) ·
[Evidence](evidence/) ·
[Status](#status-and-next-steps)

Each layer keeps one small vector state per head. Every token moves that state with two
input-dependent Householder transforms, then a gate mixes in new content:

    h_t = (1 − g_t) · H_2 H_1 h_{t−1} + g_t c_t        H_i = I − β_i u_i u_iᵀ,   β_i = 1 − cos θ_i ∈ [0, 2]

The state has a fixed size, so writing a token costs the same whether the context is ten characters
or ten million. The 25M model carries 11,904 numbers of state in total.

## About this project

Independent open research, built in India.

The question is a frontier one even though the scale is not: *what can a fixed-size recurrent state
actually represent, and what does it do when it cannot?* The trained models here are small — 5M and
25M parameters, character-level — and are research instruments rather than a product; the
[results table](#results-character-level-enwik8) says plainly what they do and do not show. The work
is meant to be judged on its evidence: design choices are settled by experiments committed before
they run and reported whichever way they come out, including when that means retracting a number
already published here, which has happened twice.

## What the research found

Two technical reports. [Paper A](paper/main.pdf) is the architecture; [Paper B](paper/paperB.pdf) is
what the same transport does when it fails, which turned out to be the more interesting half.

**Failure is quantised.** Train this transport on the word problem of a finite group and it either
solves it exactly at any length, or it lands on `1/|N|` for a normal subgroup `N` — it has learned
the quotient `G/N`, gets the coset right, and guesses inside it. Across Q₈, S₄ and A₅, twelve runs
land on a rung of their own group, worst deviation 0.021. The lattice is a property of the group,
and the model never sees it ([`evidence/word_problem.py`](evidence/word_problem.py)).

**The errors name the subgroup, not just its size.** Q₈ has three *different* normal subgroups of
order 4, so accuracy alone cannot tell them apart — all three give the same 0.250. Coset consistency
picks exactly one, and scores the other two at the predicted 0.5
([`evidence/which_subgroup.py`](evidence/which_subgroup.py)).

**Binding capacity is a character.** For an orthogonal map, the expected overlap of a vector with
its image is the normalised trace, `E[v·gv] = tr(g)/d`. A single Householder reflection has
`tr = d−2` — the *least* hiding non-trivial element of O(d) — and k of them give overlap
`(1−2/d)^k`. That is why one reflection cannot bind, and why it gets worse as heads get wider
([`evidence/trace_law.py`](evidence/trace_law.py)).

**...but more reflections do not buy this architecture a memory — we checked, and it fails
backwards.** The obvious consequence was that raising `n_h` should turn tracking heads into storage
heads. Swept in a trained model, the trace follows the law closely (0.818, 0.611, 0.364, 0.122 at
`n_h` = 2, 4, 8, 16) while recall falls monotonically, 0.113 → 0.004, ending below chance
([`evidence/nh_sweep.py`](evidence/nh_sweep.py)). A bind/unbind store recovers a value by applying
its key's reflections a second time, and assumes the stored pair is untouched in between. A
recurrence does neither: it never applies a query's inverse, and every later token transports the
*whole* state. More reflections are therefore a faster scrambler, not more capacity. The trace law
bounds what a product of reflections *could* store; it does not say this recurrence can reach it.

**A trained transport can be read as a representation.** Its character norm `⟨χ,χ⟩`, computed from
traces alone with no group table and no labels, identifies which quotient it learned. Over sixteen
seeds it agrees with the independent error-based reading on all seven runs that are genuine
homomorphisms ([`evidence/character_table.py`](evidence/character_table.py)).

Every derived number in Paper B — the bounds, the rung sets, the attainable character norms — is
recomputed from the group definitions by
[`evidence/verify_paperB.py`](evidence/verify_paperB.py), which also re-reads the measured tables
from the raw logs rather than trusting a transcription.

## Why it is built this way

Every choice below exists for a stated reason, and most were settled by a small experiment in
[`evidence/`](evidence/) before the language model was trained. [`paper/main.pdf`](paper/main.pdf)
has the full arguments.

**Recurrent, not attention.** A transformer re-reads a cache that grows with the context. A
recurrent state does not grow, so generation cost and memory per token are constant. The question a
recurrent design has to answer is what that fixed state can represent; the choices below are that
answer.

**Reflections, not a diagonal recurrence.** Diagonal (element-wise) transitions commute, so a pure
diagonal transport ends in the same state for every ordering of the same tokens. It can only
compute functions of the *multiset* of inputs. On the word problem of S₃ that caps accuracy at a
provable 0.385; two commuting transports sit on that bound (0.388, 0.389) while a non-commuting one
goes above it (0.779) ([`evidence/ceiling.py`](evidence/ceiling.py),
[`results/v2.txt`](evidence/results/v2.txt)). Two reflections about different mirrors do not
commute, and each costs only a dot product and a vector update.

**β = 1 − cos θ, not a sigmoid and not a fixed 2.** The transform scales the component along *u*
by 1 − β = cos θ. This form keeps β in [0, 2] for every input, reaches the exact reflection β = 2
(a sigmoid only approaches it), and has zero slope there, so a reflection is a stable place to
sit. β stays learnable because a fixed reflection can never forget along a direction and fixes the
sign of the determinant for every token ([`evidence/parity.py`](evidence/parity.py)). In a toy
sweep the sigmoid solved 0 of 16 state-tracking runs and the cosine form 9 of 16
([`evidence/beta_activation.py`](evidence/beta_activation.py)). The trained 25M model uses the
whole range: near-exact reflections in its first layer, soft contractions in its last.

**A convex, scalar gate.** Each transform has spectral norm at most one and the write is a convex
mix, so the state is bounded by its inputs for every sequence (a one-line proof in the paper). The
gate is one scalar per head because a scalar commutes with the reflections, and that is what makes
the exact parallel kernel possible; a per-dimension gate breaks it
([`evidence/chunkwise.py`](evidence/chunkwise.py)). The gate is capped at 0.9 so the kernel's
rescaling factors stay below e^8.1.

**An exact parallel kernel, not an approximation.** Training processes chunks of 8 tokens with one
unit lower-triangular solve per chunk and no matrix inverse. It matches the step-by-step
recurrence to float precision, and [`tests/`](tests/) check that on every run.

**A small vector state, tuned for tracking rather than storage.** A vector per head is cheap and
suits tracking *where* a sequence is. At `n_h = 2` it holds almost no independent facts, and we
first assumed the fix was a separate matrix memory
([`experimental/memory/`](experimental/memory/), still untrained). The trace law says otherwise:
the shortfall is not the vector state but the *number of reflections*, since a single reflection
hides a value least of any non-trivial orthogonal map. Storage needs `n_h ≈ d_h/2`, which is the
same mechanism at a different setting rather than a second memory
([`evidence/trace_law.py`](evidence/trace_law.py)). That predicts a heterogeneous layer — tracking
heads at `n_h = 2` beside storage heads at `n_h ≈ d_h/2` — which is
[not yet built](#status-and-next-steps).

**The block around the transport.** Normalisation, a width-4 causal convolution, per-head
RMSNorm, a SiLU output gate and a SwiGLU feed-forward surround the recurrence. This surrounding
anatomy matters a lot: a GRU placed in the same block comes close to Saryu at 5M (table below). So
comparisons keep the block fixed and change only the transport.

**Small, cheap and falsifiable first.** Everything so far runs on one T4 GPU or a laptop CPU, on
character-level enwik8, so each design question is settled in hours before anything is scaled.
Predictions are committed before a run and reported whichever way they come out; the same-order
control in [`evidence/same_order.py`](evidence/same_order.py) is recorded as falsified, because one
of its 28 scored failures missed the registered tolerance, and it says so.

## Results (character-level enwik8)

Bits per character on the last 128 characters of 64 evaluation windows. Every window *ends* at the
same position, so each context length scores identical text with more or less of it in front.
Every model was trained at context 128. Lower is better.

| model | params | train steps | ctx 128 | ctx 512 | ctx 2048 | ctx 8192 |
|---|---|---|---|---|---|---|
| **Saryu 25M** | 25.2M | 10,000 | **1.535** | **1.433** | **1.433** | **1.433** |
| Saryu 5M (released checkpoint, value embedding) | 5.0M | 5,000 | 1.671 | 1.586 | 1.586 | 1.586 |

Reading it honestly:
- **The model reads about 512 characters and no more.** Past that the recurrent state entering the
  scored text is identical to six decimals in every layer and the predictions do not change
  ([`evidence/results/effective_context.txt`](evidence/results/effective_context.txt)). Context 128
  is worse only because the scored characters then have nothing in front of them.
- An earlier version of this table claimed the model kept improving out to context 8192. That was
  an artefact of drawing window positions separately per length, so each column scored different
  text. These numbers score identical text at every length; they are better than the old ones
  everywhere, over a much shorter span than was claimed.
- Scale helps: 25M beats 5M by about 0.15 bpc at every length.
- **Superseded:** the GRU and transformer rows published here earlier used the old evaluation and
  are not comparable with these numbers. They will be rerun, with several seeds, before any
  comparison is quoted again.
- These are small models on one corpus, with our own 95/5 character split, so they are not
  comparable with published enwik8 numbers and say nothing yet about LLM scale.

## Quickstart

Requires Python 3.10+.

    pip install -r requirements.txt
    gh release download v0.1 --repo varun29ankuS/Saryu-RNN -D checkpoints
    # or without the GitHub CLI:
    mkdir -p checkpoints && for f in saryu_25m.pt saryu_v4b_last.pt; do curl -L -o checkpoints/$f https://github.com/varun29ankuS/Saryu-RNN/releases/download/v0.1/$f; done

    python -m pytest tests
    python scripts/talk.py "The history of India begins with "
    python scripts/serve.py      # web UI at http://127.0.0.1:8471 that shows the state while it writes

Training needs the corpus:

    mkdir -p corpus && curl -L http://mattmahoney.net/dc/enwik8.zip -o corpus/enwik8.zip && unzip corpus/enwik8.zip -d corpus
    CHARS=400000 STEPS=5 TARGET_PARAMS=300000 EVAL_LENS=128 SAVE= python scripts/train.py   # smoke run
    python scripts/train.py                                                                  # the 25M recipe

`saryu/model.py` reproduces the original training code's logits bit-for-bit on both checkpoints.

## Layout

```
saryu/model.py          the model: kernel, block, LM, checkpoint loader
scripts/train.py        the 25M training recipe
scripts/talk.py         text completion from the 5M checkpoint
scripts/serve.py        streaming web UI that shows the state while it writes (--selftest)
tests/                  kernel exactness, norm preservation, order sensitivity, checkpoint loading
evidence/               the small experiments behind each design choice, with their result logs
  word_problem.py         group word problems: the rung structure of trained transports
  which_subgroup.py       Q_8, where three normal subgroups share one rung and the errors must choose
  character_table.py      reading a trained transport as a representation, from traces alone
  trace_law.py            binding capacity is a character: overlap ~ (1-2/d)^k
  verify_paperB.py        recomputes every derived number in paper B from the group definitions
experimental/memory/    the matrix-memory extension (not trained yet; see its README)
paper/main.pdf          report A: the architecture, the kernel, the trained models
paper/paperB.pdf        report B: what the transport does when it cannot solve a group
checkpoints/            saryu_25m.pt, saryu_v4b_last.pt  (release v0.1, not in git)
corpus/                 enwik8                            (not in git)
```

## Status and next steps

- **Works:** the reflection recurrence at 5M and 25M parameters, the exact kernel, the demos, and
  the group-theoretic results in [Paper B](paper/paperB.pdf).
- **Next:** matched comparisons against other recurrent and attention models at larger scale, with
  several seeds and a second corpus — the single biggest gap, and the reason no competitive claim is
  made here. Also a fused GPU kernel; the current one is plain PyTorch, though chunk size alone
  already buys 2.3–5.4× and is exact at any size.
- **Open:** the heterogeneous layer the trace law predicts — tracking heads at `n_h = 2` beside
  storage heads at `n_h ≈ d_h/2`. `n_h` is now a constructor argument, so this is testable; whether
  a *trained* model uses the capacity is being measured in
  [`evidence/nh_sweep.py`](evidence/nh_sweep.py).
- **Known limits:** the trained models stop using context at about 512 characters; two of the four
  rows of Paper B's character table have no qualifying run behind them; A₅'s rung coincides with
  chance, so those runs confirm the lattice law without demonstrating it.

## Citation

```bibtex
@misc{sharma2026saryu,
  author = {Varun Sharma},
  title  = {Saryu: a recurrent language model whose state is carried by Householder reflections},
  year   = {2026},
  note   = {Technical report A, draft v1},
  url    = {https://github.com/varun29ankuS/Saryu-RNN}
}

@misc{sharma2026quantised,
  author = {Varun Sharma},
  title  = {Quantised failure: trained reflection transports collapse onto exact group quotients},
  year   = {2026},
  note   = {Technical report B, draft v1},
  url    = {https://github.com/varun29ankuS/Saryu-RNN}
}
```

## License

Apache License 2.0; see [`LICENSE`](LICENSE). The weights in the v0.1 release are under the same
license.

## References

- Grazzi et al. 2024. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*. arXiv [2411.12537](https://arxiv.org/abs/2411.12537).
- Siems et al. 2025. *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products*. arXiv [2502.10297](https://arxiv.org/abs/2502.10297).
- Yang et al. 2024a. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*. arXiv [2406.06484](https://arxiv.org/abs/2406.06484).
- Marathe et al. 2026. *Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models*. arXiv [2609.12303](https://arxiv.org/abs/2609.12303).
