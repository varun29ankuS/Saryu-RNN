# Saryu

**A recurrent language model whose state is moved by Householder reflections.**

[![Tests](https://github.com/varun29ankuS/saryu/actions/workflows/tests.yml/badge.svg)](https://github.com/varun29ankuS/saryu/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Weights: v0.1](https://img.shields.io/badge/weights-v0.1-orange.svg)](https://github.com/varun29ankuS/saryu/releases/tag/v0.1)
[![Paper: draft v1](https://img.shields.io/badge/paper-draft%20v1-lightgrey.svg)](paper/main.pdf)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)

[Why it is built this way](#why-it-is-built-this-way) ·
[Results](#results-character-level-enwik8) ·
[Quickstart](#quickstart) ·
[Paper](paper/main.pdf) ·
[Evidence](evidence/) ·
[Status](#status-and-next-steps)

Each layer keeps one small vector state per head. Every token moves that state with two
input-dependent Householder transforms, then a gate mixes in new content:

    h_t = (1 − g_t) · H_2 H_1 h_{t−1} + g_t c_t        H_i = I − β_i u_i u_iᵀ,   β_i = 1 − cos θ_i ∈ [0, 2]

The state has a fixed size, so writing a token costs the same whether the context is ten characters
or ten million. The 25M model carries 11,904 numbers of state in total.

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

**A small vector state, with facts left to a separate memory.** A vector per head is cheap and
suits tracking *where* a sequence is. It cannot hold many independent facts, so that job is
planned for a matrix memory addressed by the tracked state ([`experimental/memory/`](experimental/memory/)),
which does not train yet.

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

Bits per character on the last 128 characters of 64 random evaluation windows of each context
length (the windows are drawn separately for each length, so columns score different characters).
Every model was trained at context 128. Lower is better.

| model | params | train steps | ctx 128 | ctx 512 | ctx 2048 | ctx 8192 |
|---|---|---|---|---|---|---|
| **Saryu 25M** | 25.2M | 10,000 | **1.643** | **1.516** | **1.562** | **1.467** |
| Saryu 5M (released checkpoint, value embedding) | 5.0M | 5,000 | 1.782 | 1.671 | 1.706 | 1.606 |
| Saryu 5M (base recipe, seed 0) | 5.1M | 5,000 | 1.787 | 1.671 | 1.708 | 1.607 |
| GRU, same block anatomy, 5M | 4.9M | 5,000 | 1.811 | 1.679 | 1.726 | 1.633 |
| Transformer, 5M | 5.0M | 12,000 | 1.740 | 5.569 | 5.531 | 5.509 |

Reading it honestly:
- Saryu scores better with context it was never trained on (128 → 8192 improves every model size).
- At 5M a GRU in the identical block is close behind (the Saryu run in that same head-to-head, a
  different seed, is 1.793 / 1.681 / 1.730 / 1.618). The transport is not yet a clear win over a
  strong recurrent baseline; matched comparisons at larger scale are the next step.
- The transformer is better at its training length and cannot extrapolate past it.
- These are small models on one corpus, with our own 95/5 character split, so they are not
  comparable with published enwik8 numbers and say nothing yet about LLM scale.

## Quickstart

Requires Python 3.10+.

    pip install -r requirements.txt
    gh release download v0.1 --repo varun29ankuS/saryu -D checkpoints
    # or without the GitHub CLI:
    mkdir -p checkpoints && for f in saryu_25m.pt saryu_v4b_last.pt; do curl -L -o checkpoints/$f https://github.com/varun29ankuS/saryu/releases/download/v0.1/$f; done

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
experimental/memory/    the matrix-memory extension (not trained yet; see its README)
paper/                  technical report, draft v1: the architecture, the kernel, the trained models
checkpoints/            saryu_25m.pt, saryu_v4b_last.pt  (release v0.1, not in git)
corpus/                 enwik8                            (not in git)
```

## Status and next steps

- **Works:** the reflection recurrence at 5M and 25M parameters, the exact kernel, the demos.
- **Next:** matched comparisons against other recurrent and attention models at larger scale, with
  several seeds and a second corpus; a fused GPU kernel (the current one is plain PyTorch).
- **Planned:** the matrix memory for facts, which first needs a tracker that changes state only
  when a token should change it.

## Citation

```bibtex
@misc{sharma2026saryu,
  author = {Varun Sharma},
  title  = {Saryu: a recurrent language model whose state is carried by Householder reflections},
  year   = {2026},
  note   = {Technical report, draft v1},
  url    = {https://github.com/varun29ankuS/saryu}
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
