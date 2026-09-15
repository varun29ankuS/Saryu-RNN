# Saryu

A transport layer for sequence models, plus tools to measure what algebraic structure a
model actually learned.

## What this is, and what it isn't

**It is** a set of design choices, each fixed by a measurement rather than a guess, and
three diagnostics that tell you *which* structure a transport learned instead of just how
well it scored.

**It is not** a benchmarked architecture. It has never run on real text, never on a GPU, and
we never reproduced DeltaProduct's published numbers. Treat it as a hypothesis with measured
components.

## The finding it rests on

Models trained on group word problems **do not degrade gradually**. They land on an exact
quotient of the target group, and their accuracy is exactly **1/|N|**, where N is the kernel.
On S₄ that means runs cluster on 1.000, 0.250, 0.083, 0.042 and nothing in between.

Two independent checks:

- **Coset consistency** jumps to ~1.000 at exactly one normal subgroup and sits at chance
  below it, so the kernel is identifiable without ambiguity.
- **Cross-group exclusivity**: Q₈ produces a 0.500 plateau that S₄'s lattice forbids, and S₄
  produces 0.083 that Q₈'s forbids. Same code, same objective — the reachable accuracies are
  set by the group's algebra, which the model has no access to.

*Caveat:* the cross-group evidence is confounded by group order (|Q₈| = 8 vs |S₄| = 24). The
control that would remove it — two groups of the same order with different lattices — has
failed three times for unrelated reasons. This is the weakest part of the claim.

## Whether a quotient is failure depends on the task

Same group, same sequences, three different targets:

| target | classes | solved | acquires a faithful irrep? |
|---|---|---|---|
| full element | 24 | 2/4 | yes (max3dim ≈ 1.0) |
| V₄-coset | 6 | 2/4 | **no** (≈ 0.04) |
| A₄-coset | 2 | 3/4 | **no** (≈ 0.04) |

The quotient tasks reach 1.000 while leaving the faithful representations at zero. They take
exactly the structure the target needs. The *same* learned representation is a failure under
one target and a success under another — so the lattice is an enumeration of legal
compressions, not a catalogue of defects.

It is also **recursive**: the V₄ task's own failures land on S₃'s rungs (0.320, 0.336 ≈ 1/3),
and the A₄ task's on Z₂'s (0.495 ≈ 1/2).

## Design choices and the evidence for each

| choice | evidence |
|---|---|
| β initialised at **2.0** | Starting at 1.0 (a projection) gives **0/4** and β never moves. Every failing variant — `2·sigmoid`, `1·sigmoid`, unconstrained — starts at ≤ 1.0. Only init at 2.0 works. |
| β **learnable per token** | Fixed β = 2 is also 0/4: n_h pure reflections force det(T) = (−1)^n_h for every token, and a group with both parities can't be represented. |
| β range **(0, 2)** | Capped below 1 there are no reflections; β drifts down to 0.33–0.60 and non-abelian structure never forms. |
| **correlated** Householder init | Independent draws are near-orthogonal in high dim, and orthogonal reflections commute. The commutator ratio falls 0.912 (d=4) → 0.017 (d=256). At fla's defaults the two Householders commute to within 0.4%, so training starts almost abelian. |
| **linear** readout | 0.556 vs 0.059 at 8× training length. An MLP decodes a degenerate state and hides the transport's failure. |
| **no** state normalisation | Permutation-like representations carry a trivial summand that supplies resets for free (0.946). Project it out and the reset dies (0.307) unless you add an affine write term (0.996). RMSNorm on the recurrent state would remove it. |
| optional **kernel** aux loss | 5/6 vs 3/6 baseline, and table-free. For a homomorphism, faithful ⟺ only the identity maps to I, so |G|−1 constraints against a fixed reference beat |G|² mutual ones. |

## Diagnostics

```python
from saryu.diagnose import character_norm, group_law_error, coset_consistency

character_norm(layer, vocab)        # needs nothing at all
group_law_error(layer, table)       # needs the Cayley table
coset_consistency(pred, true, ...)  # needs the table + candidate subgroups
```

`character_norm` is ⟨χ,χ⟩ = mean of tr(T_g)², which equals the sum of squared irrep
multiplicities. For a 4-dim representation of S₄: **2.00 iff faithful**, 3 or 5 for ker V₄,
8 for ker A₄, 16 for trivial. Solved runs read 1.997–2.003; failed runs read 4.386–5.495.

It catches what group-law error cannot. A run with a respectable group law of 0.0964 — a
good homomorphism onto the *wrong* quotient — reads 5.495 here. Group-law error is blind to
the kernel by construction, because every quotient satisfies every relation of its parent.

**⟨χ,χ⟩ is an excellent diagnostic and a worthless objective.** Minimising it scores 2/6
against a 3/6 baseline: the bound holds only *on* the representation manifold and the penalty
pushes models off it. One run finished at 1.993 — the perfect faithful value — with accuracy
0.056.

## What we learned about objectives, the hard way

Ten were tried. Eight were gamed or harmful:

| objective | outcome |
|---|---|
| group-law penalty alone | collapse (all transports equal) satisfies it |
| margin hinge on transports | harmful — fights the algebra |
| Hopfield separation | achieved its own goal, drove group-law error up 35× |
| commutator-Hopfield | same, 67× |
| min(braid, commute) | takes the cheaper relation → abelian ceiling |
| character norm (table-free) | leaves the representation manifold |
| witness equations | T_g = cI satisfies them; **similar** to I, not **congruent** to it |
| **braid relations** | **works, table-free, 3/4** |
| **irrep multiplicity** | **works, 5/6 — but needs the character table** |
| **soft kernel count** | **works, table-free, 5/6** |

The pattern is about *shape*, not content: **things to satisfy at a fixed target hold;
scalars to minimise get gamed.** Any objective admitting a degenerate global transformation
will find it.

## Files

- `layer.py` — `SaryuTransport`, `SaryuLayer`
- `diagnose.py` — the measurement tools
- `benchmark.py` — Saryu vs the default parameterisation on S₄

## Honest status

The theory and diagnostics are solid on toys with |G| ≤ 60 and d ≤ 8. The architecture is
an assembly of separately-measured choices whose *composition* is being tested now. Nothing
here has met real data.

## References

- Siems et al. 2025. *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products*. arXiv [2502.10297](https://arxiv.org/abs/2502.10297).
- Grazzi et al. 2024. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*. arXiv [2411.12537](https://arxiv.org/abs/2411.12537).
