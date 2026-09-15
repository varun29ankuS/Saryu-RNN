# Saryu

A recurrent language model whose state is carried by **Householder reflections**.

Each layer keeps one small vector state per head. Every token applies two reflections to that
state and then a gate mixes in new content:

    h_t = a_t · (H_2 H_1) h_{t-1} + b_t        H_i = I − β_i u_i u_iᵀ,   β_i = 1 − cos θ_i ∈ [0, 2]

At β = 2 a reflection preserves the state's norm, and reflections do not commute, so the state
depends on the *order* of what was read — something a commutative (diagonal) transport provably
cannot represent. The state has a fixed size, so generation costs the same per token however long
the context is, and training uses an exact chunk-parallel kernel instead of a loop over time.

## Results (character-level enwik8)

Bits per character on the last 128 characters of 64 random evaluation windows of each context
length (the windows are drawn separately for each length, so columns score different characters).
Lower is better.

| model | params | train steps | ctx 128 | ctx 512 | ctx 2048 | ctx 8192 |
|---|---|---|---|---|---|---|
| **Saryu v3, 25M** | 25.2M | 10,000 | **1.643** | **1.516** | **1.562** | **1.467** |
| Saryu v3, 5M | 5.1M | 5,000 | 1.787 | 1.671 | 1.708 | 1.607 |
| GRU, same block anatomy, 5M | 4.9M | 5,000 | 1.811 | 1.679 | 1.726 | 1.633 |
| Transformer, 5M | 5.0M | 12,000 | 1.740 | 5.569 | 5.531 | 5.509 |

Honest reading:
- Saryu improves with longer context it was never trained on (trained at 128; 8192 is better).
- At 5M a GRU given the identical block anatomy is close behind (the Saryu row in that same
  head-to-head, a different seed, is 1.793 / 1.681 / 1.730 / 1.618). The transport is not yet
  a clear win over a strong recurrent baseline.
- The transformer is better at its training length but cannot extrapolate past it.
- These are small models on one corpus. Nothing here is evidence at LLM scale.

## Layout

```
saryu/model.py          the model: kernel, block, LM, checkpoint loader
scripts/train.py        the 25M training recipe (enwik8, Kaggle T4)
scripts/talk.py         text completion from the 5M checkpoint
scripts/serve.py        streaming web UI that shows the state while it writes (--selftest)
tests/                  kernel exactness, norm preservation, order sensitivity, checkpoint loading
evidence/               the small experiments that fixed each design choice
experimental/memory/    a matrix-memory extension (not trained yet; see its README)
paper/                  technical report, draft v1: the architecture (no comparisons yet)
checkpoints/            saryu_25m.pt, saryu_v4b_last.pt  (GitHub Release v0.1, not in git)
corpus/                 enwik8                            (not in git)
```

## Running

Requires Python 3.10+ and PyTorch. Download the weights and, for training, the corpus:

    gh release download v0.1 --repo varun29ankuS/saryu -D checkpoints
    mkdir -p corpus && curl -L http://mattmahoney.net/dc/enwik8.zip -o corpus/enwik8.zip && unzip corpus/enwik8.zip -d corpus

Then:

    python -m pytest tests
    python scripts/talk.py "The history of India begins with "
    python scripts/serve.py --selftest && python scripts/serve.py      # http://127.0.0.1:8471
    CHARS=400000 STEPS=5 TARGET_PARAMS=300000 EVAL_LENS=128 SAVE= python scripts/train.py   # smoke

`saryu/model.py` reproduces the original training code's logits bit-for-bit on both checkpoints.

## Status

The trained model is the reflection recurrence above. A proposed extension — a matrix memory
addressed by the reflection-tracked state — lives in `experimental/memory/` and does not train
yet. The design reasoning and the trained models are written up in `paper/main.pdf`.

## License

Apache License 2.0; see `LICENSE`. The weights in the v0.1 release are under the same license.

## References

- Grazzi et al. 2024. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*. arXiv [2411.12537](https://arxiv.org/abs/2411.12537).
- Siems et al. 2025. *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products*. arXiv [2502.10297](https://arxiv.org/abs/2502.10297).
- Yang et al. 2024a. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*. arXiv [2406.06484](https://arxiv.org/abs/2406.06484).
- Marathe et al. 2026. *Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models*. arXiv [2609.12303](https://arxiv.org/abs/2609.12303).
