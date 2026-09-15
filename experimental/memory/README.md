# Experimental: a matrix memory on top of the reflection-tracked state

Not part of the trained model. This folder holds a proposed extension and the experiments around
it. Run everything from inside this folder (the scripts import each other by name).

## The idea

- **Matrix memory** (`parity_block.py`): a delta-rule store per head (`S <- decay*S + (v - S k) k^T`,
  beta < 1, no reflections in the store), with a fused-kernel path through flash-linear-attention.
- **Tied map**: keys and values come from one projection, the key being the previous step's value,
  so the store maps each item to the next and `n_read` repeated reads walk several links.
- **SaryuBlock** (`saryu_block.py`): the reflection tracker (beta = 2) plus the memory, with facts
  addressed by content bound to state (phase binding) and time (rotation).

## Status

- **SaryuBlock does not train yet.** The tracker reflects on every token, so a fact is stored
  under the state at write time while a later query carries a different state; addresses never
  match (recall fell from 0.793 to 0.035 when state binding was switched on). The proposed fix, a
  gated tracker that changes state only when a token should, is not built.
- **P1b (do extra reads substitute for layers?) is unresolved.** See `PREDICTIONS.md`: the chain
  task used for v1–v4 had a query-free shortcut; `recall_tasks.make_chain(dmode='chain')` fixes it.
- The transformer baseline needs lr = 3e-4 to solve 4-pair recall; on the chain task it has not
  yet been made to work.

## Checks

    python trace_logic.py              # 14 links of the tied block, float64
    python saryu_block.py              # SaryuBlock invariants
    python recall_tasks.py             # task generators, including the shortcut audit
    STEPS=60 python cpu_precheck.py    # pre-GPU gate
    python verify_fixes.py             # the 2026-09-15 audit fixes

## GPU runs

`kaggle/p1/build.py` bundles `gpu_run.py` and its modules into one Kaggle script
(`python build.py`, then `kaggle kernels push -p .`). `kaggle/triage/` is the same bundle built
with `MODE=ceiling_triage`. Each folder's `kernel-metadata.json` names the target kernel.

## References

- Yang et al. 2024a. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*. arXiv [2406.06484](https://arxiv.org/abs/2406.06484).
- Yang et al. 2024b. *Gated Delta Networks: Improving Mamba2 with Delta Rule*. arXiv [2412.06464](https://arxiv.org/abs/2412.06464).
- Hatamizadeh et al. 2026. *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*. arXiv [2605.22791](https://arxiv.org/abs/2605.22791).
- Huang et al. 2026. *MDN: Parallelizing Stepwise Momentum for Delta Linear Attention*. arXiv [2605.05838](https://arxiv.org/abs/2605.05838).
- Behrouz et al. 2025. *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*. arXiv [2504.13173](https://arxiv.org/abs/2504.13173).
- Gu & Dao 2023. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv [2312.00752](https://arxiv.org/abs/2312.00752).
- Grazzi et al. 2024. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*. arXiv [2411.12537](https://arxiv.org/abs/2411.12537).
- Siems et al. 2025. *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products*. arXiv [2502.10297](https://arxiv.org/abs/2502.10297).
- Arora et al. 2023. *Zoology: Measuring and Improving Recall in Efficient Language Models*. arXiv [2312.04927](https://arxiv.org/abs/2312.04927).
- Merrill et al. 2024. *The Illusion of State in State-Space Models*. arXiv [2404.08819](https://arxiv.org/abs/2404.08819).
- Sarrof et al. 2024. *The Expressive Capacity of State Space Models: A Formal Language Perspective*. arXiv [2405.17394](https://arxiv.org/abs/2405.17394).
- DeepSeek-AI 2024. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. arXiv [2405.04434](https://arxiv.org/abs/2405.04434).
