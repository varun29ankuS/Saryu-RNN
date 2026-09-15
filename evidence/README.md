# Evidence: the measurements behind the model's design

Each script here settled one design choice in `saryu/model.py`. They are small, self-contained,
and run on a CPU. Result logs from when they were run are in `results/`.

| design choice in the model | script | result log |
|---|---|---|
| Order-sensitive (non-commuting) transport beats any commutative one on a task with a proven ceiling | `ceiling.py` | `results/v2.txt` |
| Initialise beta at 2 (a reflection); starting at 1 (a projection) never recovers | `beta_act.py` | `results/beta_act.txt` |
| beta = 1 - cos(theta): bounded in [0, 2], and a reflection is a stable point | `beta_activation.py` | — |
| An odd number of pure reflections cannot represent groups with both parities; beta must stay learnable | `parity.py` | — |
| Random high-dimensional init starts every token near an involution | `init_angle.py` | `results/init_angle.txt` |
| No normalisation on the recurrent state: the trivial summand supplies resets for free | `flipflop.py` | `results/flipflop.txt` |
| A linear readout exposes a failing transport that an MLP readout hides | `missing.py` | `results/missing.txt` |
| Optional table-free kernel auxiliary loss | `kernel_pen.py` | `results/kernel_pen.txt` |
| Failures land on exact quotients of the group (1/\|N\| accuracies) | `lattice.py` | `results/lattice.txt` |
| The chunk-parallel kernel equals the step-by-step recurrence | `chunkwise.py` | — (also `tests/test_model.py`) |

`transport/` is the original single-layer transport package (`layer.py`, the diagnostics in
`diagnose.py`, and `benchmark.py`) with its own README. The trained model supersedes its layer.

Scope: these are toy-scale results (small groups, small dimensions). The language-model results
are in the top-level README.

## References

- Grazzi et al. 2024. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*. arXiv [2411.12537](https://arxiv.org/abs/2411.12537).
- Siems et al. 2025. *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products*. arXiv [2502.10297](https://arxiv.org/abs/2502.10297).
- Yang et al. 2024a. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*. arXiv [2406.06484](https://arxiv.org/abs/2406.06484).
