# A way out of the memory problem: what geometry and topology offer

Exploration note, 2026-09-17. Prompted by `evidence/results/nh_sweep_shortgap.txt`, where raising
`n_h` made recall monotonically worse. This surveys geometry and topology for structures that could
give Saryu a memory, and ranks them by whether they attack the actual failure.

## 1. The failure, stated properly

Saryu's recurrence is **parallel transport along a path**. Positions are the base, the head state
`h ∈ R^{d_h}` is the fibre, and each token supplies a transport `T_t ∈ O(d_h)`. The state at time `t`
is the holonomy of the path applied to what was written earlier.

Two measured failures (`n_h = 2`, `d_h = 16`, chance 0.016):

| pairs → | 1 | 2 | 4 |
|---|---|---|---|
| **gap 0** | 1.000 | 0.711 | 0.199 |
| **gap 4** | 1.000 | — | 0.113 |
| **gap 64** | 0.012 | — | 0.012 |

- **Capacity.** One fact is solid, four are nearly gone, even at zero distance.
- **Persistence.** Any number of facts dies by gap 64.

And the mechanism behind them:

- **No unbind.** A bind/unbind store recovers a value by applying the key's operator a second time.
  The recurrence never applies a query's inverse.
- **The state is re-transported.** Every later token acts on the *whole* state, so a stored pair is
  hit by `T_{s+1} … T_t`.

### The tension is a theorem, not a tuning problem

> Order-sensitivity **is** non-trivial holonomy. Exact retrieval **is** the ability to undo the
> transport. A flat connection has trivial holonomy around contractible loops — path-independent,
> and therefore undoable. A curved one is path-dependent, and therefore not.

Saryu is good at word problems *precisely because* its connection is curved: the holonomy around a
loop is the group element, which is the whole point of Paper B. The same curvature is what destroys
a stored pair. **One homogeneous state cannot be both**, which is exactly why raising `n_h` — which
raises curvature everywhere, including where the memory would live — made recall worse rather than
better.

A second, independent obstruction: reflections are orthogonal, so the transport is
**measure-preserving on a bounded state**. By **Poincaré recurrence** the state returns arbitrarily
close to earlier states, so no slot stays permanently distinguishable. The gate's contraction
(`g > 0`) is what breaks measure preservation — which is also why the trained model chose `β = 1.56`
rather than a pure reflection when it needed to hold something.

**So the state must split into a curved part that tracks and a flat part that remembers.** Every
candidate below is judged on whether it delivers that split.

## 2. Geometry

### 2.1 Reducible holonomy — carve an invariant subspace  ★ best first move

Constrain every reflection direction to a tracking subspace `W ⊂ R^{d_h}`, i.e. `u = P_W û`. Then
each `T_t` acts as the **identity on `W^⊥`**: the holonomy group reduces to `O(W) × I`, the
connection is flat on `W^⊥`, and anything written there is never scrambled. Tracking keeps its
curvature inside `W`; memory sits in the flat complement.

- Cost: tracking loses `dim W^⊥` directions. A per-head split is the knob.
- This is the *correct* version of the "heterogeneous heads" idea Paper A proposed. That version
  varied `n_h` and failed; this one varies **where the reflections are allowed to point**, which is
  the quantity that actually controls what gets scrambled.
- Prior art: **subspace carving in order-p tensor memories** (arXiv [2606.11391]) uses orthogonal
  projections with `P_i P_j = 0` so bindings occupy non-interfering subspaces, with unbinding by
  projection. Same idea, arrived at from tensor memories rather than from holonomy.
- Implementation in Saryu: one line in `SaryuV3Block.forward` — project `v_proj(z)` onto `W` before
  normalising. Directly testable against the table in §1.

### 2.2 Symplectic structure — invertibility for free  ★ attacks the other failure

Symplectic maps (`Sp(2n)`, SympNets) are **exactly invertible**, and the inverse is available
analytically at the same cost as the forward map. Phase space splits into conjugate pairs `(q, p)`,
which is a natural key/value pairing with the canonical form as the retrieval operator.

- Attacks "no unbind" rather than "scrambling", so it is complementary to §2.1.
- Cheaper version for Saryu: reflections are **involutions** — `H(u)² = I` — so the unbind operator
  already exists. What is missing is a read path that *applies the query's transport again* instead
  of only pushing the state forward. That is an architecture change, not new mathematics.

### 2.3 Central extensions — the Heisenberg group  ★ most elegant

The Heisenberg group has law

    (x₁, y₁, z₁) · (x₂, y₂, z₂) = (x₁+x₂, y₁+y₂, z₁+z₂+⟨x₁, y₂⟩)

The central coordinate `z` accumulates the **ordered bilinear form** of the path. That is binding,
built into the group law rather than bolted on — and the centre is invariant under conjugation, so
it is *not* scrambled by subsequent transports. It is a flat direction by construction.

This also explains the failure in one line: **an abelian state cannot bind, because binding is a
commutator.** `[x, y] = z ≠ 0` is precisely the structure a key–value store needs, and Saryu's
vector state has no centre to put it in.

- Cost: a full `z` is `d²` (an outer-product memory in disguise). Restricting to a low-rank or
  bivector part is the practical version.
- Honest reading: this says the matrix memory was *right*, and tells us exactly what form it must
  take — a conjugation-invariant accumulator, not another transported vector.

### 2.4 Clifford / geometric algebra — the native language of reflections

Cartan–Dieudonné is a Clifford statement, and the Pin group is *generated by reflections*, so Saryu
already lives here. The geometric product `ab = a·b + a∧b` splits into a symmetric part and an
antisymmetric wedge; the **wedge is order-sensitive binding**. Storing memory in the bivector grade
`Λ²V` (dimension `d(d−1)/2`) gives an antisymmetric outer-product memory.

- Prior art: Clifford Group Equivariant NNs (arXiv [2305.11141]), CliffordNet (arXiv [2601.06793]),
  Microsoft's CliffordLayers. Tensor-product representations embed into Clifford algebra, where
  roles and fillers need only be **orthogonal** for exact recovery, reducing memory from `d^(p+1)`
  to `C(d, p+1)`.
- Caveat: under the Pin action, bivectors transform too — so this needs §2.1's invariance to be
  protected. Clifford supplies the *binding operator*; it does not by itself supply flatness.

### 2.5 Hyperbolic geometry — capacity, not mechanism

Negative curvature gives exponential volume growth: a depth-`D` binary tree embeds in `O(D)`
hyperbolic dimensions against `O(2^D)` Euclidean, so exponentially many near-orthogonal items fit at
bounded radius (Poincaré ball, Ganea et al. arXiv [1805.09112]).

**But capacity was never the blocker.** `recall_by_construction` already reaches 4/4 pairs at
`k ≈ d/2` in flat space; the failure is scrambling and retrieval. Hyperbolic geometry would multiply
capacity *after* the mechanism works, so it is a later multiplier, not the fix. Worth saying plainly
because it is the most tempting-sounding option.

### 2.6 Grassmannian / Stiefel

Memory as a set of **subspaces**, retrieved by projection rather than by inner product. A structured
version of §2.1, with geodesics available on `Gr(k, n)` if the memory itself needs to move smoothly.

## 3. Topology

### 3.1 Characteristic classes — the obstruction has a name

The transports define a cocycle over the sequence. A globally consistent memory frame exists iff
that cocycle is a **coboundary**; the obstruction lives in `H¹`. This turns "can this architecture
remember?" into a measurable quantity rather than a vibe, and suggests a diagnostic: measure how far
the learned transports sit from a coboundary. Non-zero curvature ⇒ non-trivial class ⇒ no global
frame, which is §1's tension in cohomological form.

### 3.2 Sheaves — a principled objective

A cellular sheaf puts a vector space at each position and restriction maps on edges; the kernel of
the sheaf Laplacian is the space of **global sections** — data consistent across the entire
sequence. That is a precise formalisation of "what can be remembered", and it comes with a training
objective: maximise the dimension of the global-section space on the memory subbundle.

- Prior art: Neural Sheaf Diffusion (Bodnar et al.), Sheaf NNs with connection Laplacians (Barbero
  et al., PMLR v196), Cooperative Sheaf NNs (ICLR 2026). `arXiv 2605.06395` connects Hilbert bundles
  and cellular sheaves directly.

### 3.3 Braid groups and topological protection — the right long-term intuition

In topological quantum memory, information is stored non-locally and the resulting transformation
depends only on the **order** of exchanges, not on the path — so local noise cannot corrupt it.
That is exactly the property a stored pair needs against intervening tokens. Recent work recasts
learning as braid programming (arXiv [2608.15829]), claiming noise immunity and
catastrophic-forgetting resistance by construction.

- Practically distant, but it is the cleanest statement of the goal: **memory should be a
  topological invariant of the token sequence, not a point in a transported vector space.**

### 3.4 Covering spaces, winding number

A state on a covering space remembers which homotopy class of path it took — memory *of the path*,
not of content. Useful for counters and matched delimiters; not for key–value recall.

### 3.5 Persistent homology

Records shape of data across scales. No natural fit for key–value retrieval; listed for
completeness and set aside.

## 4. Ranking

| candidate | fixes | implementable now | verdict |
|---|---|---|---|
| Carved invariant subspace (§2.1) | scrambling | one line | **do first** |
| Inverse read path (§2.2) | no unbind | small change | **do first** |
| Central extension / bivector store (§2.3, §2.4) | both, principled | a real build | **next** |
| Sheaf objective (§3.2) | design language + objective | medium | useful framing |
| Cohomological diagnostic (§3.1) | measurement | small | cheap, do alongside |
| Hyperbolic (§2.5) | capacity only | medium | later multiplier |
| Braid / topological (§3.3) | the ideal | far | direction, not a plan |

## 5. The proposed experiment

Both tier-1 items are testable against the table in §1 without changing the training recipe:

1. Carve `W` (tracking) and `W^⊥` (memory) per head; reflections constrained to `W`. Predict recall
   at 4 pairs / gap 4 rises above `0.113`, and — the sharper prediction — that recall at gap 64
   stops collapsing, because the memory subspace is no longer transported.
2. Add a read path applying the query token's own transport to the state before the readout
   (reflections are involutions, so this *is* the unbind).

Registered falsifier: if carving leaves recall within noise of `0.113` at 4 pairs, then scrambling is
not the binding constraint and the problem is the write rule, not the geometry.

---

**Sources.** Subspace carving arXiv [2606.11391] · Clifford Group Equivariant NNs arXiv
[2305.11141] · CliffordNet arXiv [2601.06793] · Hyperbolic NNs arXiv [1805.09112] · Sheaf NNs with
connection Laplacians PMLR v196 · Hilbert bundles and cellular sheaves arXiv [2605.06395] · braid
programming arXiv [2608.15829] · SympNets (Jin et al.) · Kitaev, anyons in an exactly solved model
cond-mat/0506438.

---

# Addendum: the topic I missed — path signatures

Everything above treats the problem as *geometry inside a fixed state*: which subspace, which
operator, which curvature. That was the wrong level of description, and the carve experiment
(`evidence/results/carve_test.txt`) is what exposed it — carving helped exactly as predicted and
still hit a ceiling of 0.43.

The right object is **K. T. Chen's signature of a path**, the foundation of Lyons' rough path theory.

## The hierarchy

The signature of a path is its sequence of iterated integrals,

    S(X) = ( 1, ∫dX, ∫∫dX dX, ∫∫∫dX dX dX, … )

and the levels mean something concrete:

| level | object | what it remembers |
|---|---|---|
| 1 | increments `∫dX` | the **sum** of what was seen — order-blind |
| 2 | `∫∫dX dX`; its antisymmetric part is the **Lévy area** | ordered **pairwise** interaction |
| p | order-p tensor | ordered p-way interaction |

**Binding is a level-2 object.** A key–value association is an ordered pair, and the smallest
structure that can hold one is the second iterated integral — equivalently an outer product, whose
antisymmetric part is the signed area.

**Saryu's state is level 1.** It is a transported vector, `h_t = a_t T_t h_{t-1} + b_t`, a sum of
transported writes. No rearrangement of a level-1 state produces a level-2 quantity, because the
tensor order of the state bounds the order of interaction it can represent. That is a statement
about *order*, not about geometry within an order — which is why every fix that stayed at level 1
failed or capped:

- raising `n_h` — still level 1, and it only scrambled faster (`nh_sweep_shortgap.txt`)
- carving a subspace — still level 1, ceiling 0.43 (`carve_test.txt`)
- `outer-equiv` and `bivector` reaching 1.000 in `memory_mechanisms.txt` — those are **level-2
  objects**, which is precisely why they worked

So the Heisenberg centre, the Clifford bivector and the outer-product store were not three
alternatives. They are one object seen from three sides: the **level-2 term of the signature**. The
centre of the Heisenberg group *is* the Lévy area.

## Why this fits Saryu specifically

**Chen's identity**: the signature of a concatenation is the tensor product of the signatures,
`S(X * Y) = S(X) ⊗ S(Y)`. Signature accumulation is therefore **associative**, so it admits an exact
chunk-parallel scan — the same property Saryu's kernel is built on. Moving to level 2 does not cost
the architecture its defining feature.

**Hambly–Lyons uniqueness**: the full signature determines the path up to tree-like equivalence. A
signature is a *faithful* memory; truncating at level N makes it lossy in a controlled way. So the
real capacity knob is **N, the truncation level** — not `n_h`, and not the carve width.

**The state space is a group.** Truncated signatures are the grouplike elements of the truncated
tensor algebra `T^N(R^d)` — the free nilpotent Lie group of step N, of dimension `d + d(d−1)/2` at
step 2. The state stays a group element, which is the thing Paper B shows this architecture is
unusually good at tracking.

**Sub-Riemannian geometry gives the cost.** The step-2 free nilpotent group is a Carnot group; by
Chow–Rashevskii the vertical (memory) direction is generated by brackets of horizontal motion, so
memory is *written by moving* rather than through a separate write port. The ball–box theorem says
that vertical direction scales as distance², i.e. writing to memory is quadratically more expensive
than moving along it — a real prediction about write efficiency.

## Where this already exists in ML

- **Neural CDEs / Neural RDEs** (Kidger et al.) — continuous-time RNNs driven by a rough path, with
  log-signatures as the driving features.
- **Deep signature transforms**, signature kernels, `RoughPy`.
- **Incremental Signature Contributions** (arXiv 2602.11805) reorganises the signature via Chen's
  identity into a time-indexed representation usable inside standard sequential models.
- **Tensor-to-tensor models with fast iterated-sum features** (arXiv 2506.06041) — iterated sums are
  the discrete signature.
- **RunningTensor** (arXiv 2609.12814), already in this project's literature index, states the
  hierarchy outright: linear attention and SSMs keep a *second-order* (matrix) state, which bounds
  the order of interactions, and generalising to higher order improves **multi-query associative
  recall** — the task we fail — while keeping both recurrent and parallel forms.

That last one is worth sitting with. The field's linear-attention baselines are already at level 2.
Saryu is at level 1. Our recall results are what the hierarchy predicts.

## What this changes

The earlier ranking put carving first and the matrix memory last. The signature reading reverses it:
carving is a level-1 palliative, and it bought 5× at gap 16 against a 0.43 ceiling, which is exactly
what a level-1 fix should do. **A second-order term in the state is not an optional add-on; it is the
minimum structure that can bind at all.**

The honest restatement of Saryu's position: it is an unusually good **level-1** model — Paper B shows
it composes group elements exactly, which is what level 1 does well — and level-1 models cannot
associate. The open design question is the cheapest level-2 term that preserves the exact scan, with
the antisymmetric (Lévy area / bivector) part at `d(d−1)/2` the natural first candidate, since
`memory_mechanisms.txt` has it at 1.000 for every gap and pair count tested.

---

# Addendum II: the geometry we were looking at was the wrong geometry

Everything above — holonomy, curvature, signatures, tensor order — is geometry of the **state**:
what a fixed-size vector can hold. Three results since then say that question is already answered
and is not the one that matters.

| fact | source |
|---|---|
| a level-2 store recalls 1.000 at every gap and pair count | `memory_mechanisms.txt` |
| carve, bind-write and a level-2 term are **indistinguishable** across seeds | `multiseed.txt` |
| the model converges to `top1 = 1/n_pairs` with `in_set ≈ 0.9` | `set_membership.txt` |
| without a curriculum, 4 pairs never leaves chance at any learning rate | `recall_curriculum.txt` |

Representability is fine. **The failure is in the learning dynamics**, and its geometry is the
geometry of the loss landscape under the task's symmetry group — not of the state.

## 1. The Birkhoff polytope, and why 1/n is not a coincidence

For a sequence with `n` pairs, what the model has learned is an `n × n` assignment: given query key
`i`, how much mass on value `j`? That matrix lives in (or near) the **Birkhoff polytope** `B_n`, the
doubly stochastic matrices, whose vertices are the `n!` permutation matrices.

    correct binding   a VERTEX of B_n — one value per key
    set membership    the BARYCENTRE — every entry 1/n

The barycentre is the unique point **fixed by the whole symmetry group `S_n`** that relabels pairs.
Our task is exactly symmetric under that relabelling: pairs are drawn i.i.d. and nothing
distinguishes them. So the loss is `S_n`-invariant, and the barycentre is *automatically a critical
point of it*.

That reframes the headline number. `top1 = 1/n` is not "the accuracy of a weak model"; it is **the
accuracy of the symmetric critical point**. Gradient descent from a symmetric initialisation has no
reason to leave it, because the gradient in every symmetry-breaking direction vanishes there by
symmetry. Escaping is a **bifurcation**, not a descent step — which is precisely why no architecture
change moved it, and why every arm landed on the same number.

## 2. Equivariant bifurcation theory says which escapes are even possible

When a symmetric equilibrium loses stability, the branches that appear are not arbitrary. By the
**Equivariant Branching Lemma** (Golubitsky–Stewart–Schaeffer), generic bifurcating branches lie in
the fixed-point subspaces `Fix(H)` of **isotropy subgroups** `H ≤ S_n` with `dim Fix(H) = 1`. The
reachable symmetry-broken solutions are therefore classified by the **isotropy lattice** of the
group acting on parameter space.

This is the same machinery as Paper B, pointed at a different object:

| | group | lattice | what it quantises |
|---|---|---|---|
| Paper B | the task's group `G` | normal subgroups `N ⊴ G` | which **quotient** a failed transport represents |
| here | the symmetry group `S_n` of the task | isotropy subgroups `H ≤ S_n` | which **partial binding** training can reach |

Paper B found failure quantised by a subgroup lattice. If this reading is right, *learning* is
quantised by a subgroup lattice too — partial solutions should bind some pairs and leave the rest
at the barycentre, corresponding to a Young-subgroup `S_k × S_{n−k}` isotropy, rather than degrading
smoothly. That is a sharp, falsifiable prediction about the **shape of the assignment matrix**, not
about accuracy.

## 3. Why it is hard, quantitatively: the information exponent

Independently of symmetry, binding is a **degree-2** function of the input (it needs the product
key × value), while set membership is **degree-1** (a sum of what was written). For SGD on
multi-index models, the sample complexity is governed by the **information exponent / leap
complexity** (Ben Arous–Gheissari–Jagannath; Abbe et al.): a leap-`k` function needs `n ≳ d^{k−1}`
or worse, and the learning curve shows a long plateau followed by a sharp escape — *exactly* the
shape in `trace_base.txt` (flat to ~step 1000, then a climb).

So the two readings agree and are complementary: **symmetry says the barycentre is a critical point;
the information exponent says how long the escape takes.** And it explains the curriculum result
(`recall_curriculum.txt`, 9× improvement): a curriculum that starts at one pair *breaks the
symmetry by hand* — with `n = 1` there is no `S_n` to be stuck in — and then grows it. That is
symmetry breaking delivered through the data rather than the architecture, which is why it was the
one intervention that ever worked.

## 4. Two smaller geometric points, both real

**The parameter space is not the sphere.** `H(u) = I − β uuᵀ` depends on `u` only through `uuᵀ`, so
the true parameter space per reflection is **real projective space** `RP^{d−1}`, not `S^{d−1}`. It
is a `Z₂` quotient, `π₁(RP^{d−1}) = Z₂`, so with `N` reflections the landscape carries `2^N` exact
discrete symmetries. Equivalent minima are separated by this group, and paths between them need not
be contractible — a concrete source of the saddle structure, and a reason plain Euclidean intuition
about the landscape misleads here.

**Information geometry, not Euclidean geometry.** Gradient descent follows the Euclidean gradient;
the loss lives on a statistical manifold with the **Fisher** metric. If the binding direction has
small Fisher information at the barycentre, the Euclidean step is nearly orthogonal to the direction
that matters. That is the principled version of "try a better optimiser": a **natural gradient** or
a spectral preconditioner (Muon-style) rescales exactly those stiff directions. Worth noting that
this project already compared AdamW against Muon and found them tied — but that was on enwik8 loss,
where no symmetry-breaking bifurcation is in play, so it does not speak to this.

## 5. What to measure, in order

1. **Is the converged model at the barycentre?** `evidence/assignment_matrix.py` reports the full
   `n × n` assignment, its distance to the barycentre, row entropies, and row spread. It
   distinguishes three failures that all give `1/n` accuracy: the barycentre (symmetric critical
   point), a rank-one value bias (rows identical but not uniform), and a partial vertex (some rows
   peaked). Only the first supports the symmetry story.
2. **Does breaking the symmetry escape it?** The curriculum already says yes, partially. The sharper
   version is to break it *at initialisation* or with a small `S_n`-breaking auxiliary term, and
   see whether the plateau shortens.
3. **Is it conditioning?** A natural-gradient or Muon-style preconditioner, on *this* task rather
   than on language modelling.
4. **Is the partial solution lattice-shaped?** If training is stopped mid-escape, does the
   assignment matrix show `k` bound pairs and `n−k` flat ones — a Young subgroup — rather than a
   uniform partial improvement?

The honest summary of where this leaves the project: we spent the session asking what the state can
represent, and answered it. The question that actually governs the measured behaviour is **why
gradient descent never leaves the symmetric point**, and that is a different branch of mathematics —
equivariant bifurcation and the geometry of symmetric loss landscapes — which none of the earlier
notes touched.

## 6. Measured — and it corrects both the prediction above and an earlier claim

`evidence/results/assignment_matrix.txt`, base model, 4 pairs, converged:

              val0   val1   val2   val3    row entropy
      key0   0.290  0.286  0.252  0.172    1.367 / 1.386
      key1   0.290  0.286  0.253  0.172    1.367
      key2   0.290  0.288  0.251  0.171    1.366
      key3   0.291  0.286  0.251  0.172    1.367

      mean |row - mean row|  0.0004      diagonal mass  0.250

**Not the barycentre.** Section 1 predicted every cell at 1/n; the rows are instead identical to
four decimals but *positionally biased*, from 0.290 on the earliest pair's value down to 0.172 on
the latest. The matrix is **rank one**: `M = 1 pᵀ`.

That distinction matters, and it sharpens the arithmetic. For any rank-one assignment the diagonal
mass is `(1/n) Σ_j p_j = 1/n` **identically, for every `p`**. So:

> `top1 = 1/n` is the signature of a readout with ZERO key-dependence. It is not evidence of
> uniform guessing among the remembered set.

This corrects the earlier reading in `set_membership.txt`, which said the model "knows the set and
guesses uniformly among them". Half of that is right and half is not. `in_set ≈ 0.89` confirms the
mass does sit on the sequence's own values, so set membership is real. But within that set the
distribution is *not* uniform — it is a fixed positional prior — and the reason accuracy lands on
`1/n` is the rank-one structure, not uniformity.

It also re-reads the primacy pattern (`p0` best, `p3` worst) from the traces: that is not a memory
gradient at all. It is this single positional prior, the same for every query, seen one row at a
time.

**Where that leaves the symmetry story.** Weaker than section 1 claimed, and more specific. The
model *has* left the fully symmetric point — it broke the symmetry among values, which is easy,
because position distinguishes them. What it has not done is develop any coupling between the key
and the value: the key input is effectively ignored by the readout. So the bifurcation that never
happens is precisely the one that would make rows differ, and the rank-one manifold
`{1 pᵀ}` — not the single barycentre point — is the attractor to characterise. Its codimension, and
the curvature of the loss transverse to it, is the thing to compute next.

## 7. The symmetry story does not survive contact with the gradient

Section 1 argued that `top1 = 1/n` is the symmetric critical point of an `S_n`-invariant loss, so
escaping it is a **bifurcation** rather than a descent step. That is falsifiable, and
`evidence/binding_gradient.py` falsifies it.

Define `B` = diagonal minus off-diagonal mass of the assignment matrix, so `B = 0` for any rank-one
readout and raising `B` *is* building key-dependence. Measure the cosine between the descent
direction `−∇L` and the binding direction `∇B`:

    at init          B = −0.0006    |∇L| = 10.4    |∇B| = 2.97    cos(−∇L, ∇B) = +0.80
    after training   B = +0.0095    |∇L| =  3.04   |∇B| = 0.98    cos(−∇L, ∇B) = +0.75

**The gradient is 0.75-aligned with binding, before and after training.** A symmetric critical point
would show a cosine near zero — the defining property of a bifurcation is that the symmetry-breaking
direction is invisible to the gradient. It is not invisible here; it is one of the most aligned
directions available. So sections 1 and 2 are wrong about this model, and the Birkhoff-barycentre
reading should be treated as a hypothesis that was tested and failed, not as an explanation.

Two further facts rule out the softer versions:

- **`n = 2`, 6000 steps, three learning rates.** Rows `0.502 / 0.498` and `0.499 / 0.501`, row
  spread `0.0001`–`0.0009`, entropy at its maximum. Exact barycentre with a two-element symmetry
  group. If `S_n` symmetry were the obstruction, `n = 2` is where it should break first.
- **The loss does improve** — cross-entropy 4.16 → 2.95 over the same 3000 steps. Training is
  working; it is buying the value set and the positional prior, and not buying binding.

So the accurate statement is narrower and stranger than the symmetry story: *the direction that
would create key-dependence is loss-improving and well-aligned with the gradient, and training still
does not go there.* That is consistent with stiffness — the right direction taken in vanishingly
small steps — which is what `evidence/binding_long.py` tests, at 2 pairs, for 20,000 steps, against
both AdamW and an orthogonalised-momentum (Muon) preconditioner that rescales exactly such
directions.

If B stays at ~0 for both, stiffness is wrong too, and what remains is the least exotic explanation
of all: **this readout cannot express key-dependent retrieval**, so no optimiser and no amount of
training will produce it. That would be a statement about the architecture, and it would be the
correct one — reached, it should be said, only after three geometric hypotheses were tried and
discarded.

---

# Addendum III: one matched filter, one bit

The probe settled what was missing, and it is not what any of the geometry predicted.

    tap          emb   L0_mix  L0_ffn  L1_mix  L1_ffn  final   model
    2 pairs     0.024   0.032   0.023   0.998   0.998   0.998   0.993
    4 pairs     0.026   0.034   0.036   0.250   0.245   0.248   0.245

A linear probe reads anything linearly present. At 4 pairs it reads **0.250 = 1/4** — exactly the
model's own accuracy, exactly set-guessing. The association is not hidden behind a weak readout. It
is **not in the state**. And binding is localised: at 2 pairs it appears in *layer 1's recurrence*
and nowhere earlier, going 0.032 → 0.998 at that one site.

## The argument

1. The recurrence is a linear update on one vector, so the state is a **superposition** of every
   write, each transported by whatever followed it:  `h = Σᵢ Aᵢ cᵢ`.

2. Retrieval means **selecting** component `q` from that sum.

3. At read time exactly two things are available: the query token's own transport `T_q`, and a fixed
   linear readout `W`. So the output is

       W · T_q · Σᵢ Aᵢ cᵢ  =  Σᵢ (W T_q Aᵢ) cᵢ

4. Correct retrieval therefore requires `W T_q Aᵢ ≈ 0` for every `i ≠ q`: **one linear operator must
   annihilate every non-matching item.** That is a single matched filter.

5. A single matched filter extracts **one** component. This is not a tuning constant — it follows
   from having one vector and one query-dependent operator.

6. Count what each task needs, *given* the model already knows the value set (measured: in_set ≈ 0.9):

   | pairs | assignment entropy `log₂(n!)` | set-guessing |
   |---|---|---|
   | 2 | **1.00 bits** | 0.500 |
   | 3 | 2.58 bits | 0.333 |
   | 4 | 4.58 bits | 0.250 |
   | 8 | 15.30 bits | 0.125 |

   `n = 2` needs exactly **one bit** — "this one or the other one". One matched filter delivers
   precisely that. `n = 4` needs 4.58 bits and gets none of them.

That is why 4 pairs lands *exactly* on set-guessing rather than degrading partway: the architecture
does not have a small amount of binding, it has one filter's worth, and one filter's worth is one
bit. Two pairs sits exactly at the structural limit — which is why it looked like a triumph and
generalised to nothing.

## Why every fix failed, in one line each

- **more reflections (`n_h`)** — gives the filter more ways to be *shaped*; does not give a second
  filter. Failed three times.
- **carving a memory subspace** — protects the superposition from being scrambled; the superposition
  was never the problem, selecting from it is. Within seed noise.
- **a level-2 term bolted on the output** — adds an outer product to the *readout*, while the
  bottleneck is that the recurrence writes one vector. Within seed noise.
- **the optimiser** — Muon multiplies the escape rate ~10×, so it reaches the one-bit solution far
  sooner. It cannot manufacture a second bit. Perfect at 2 pairs, nothing at 4.

Each was aimed at capacity, geometry, or dynamics. The limit is **arity**: how many things can be
selected at read time, which is one.

## The prediction, registered before the run

If the limit is one matched filter ≈ one bit, then **n = 3 already fails**, collapsing to ~1/3
(0.333), because 2.58 bits is already past the budget. If instead capacity degrades smoothly with
`d/n`, n = 3 should land near 0.7, comfortably between the 1.000 at n = 2 and the 0.250 at n = 4.

A cliff at n = 3 supports the arity account. A graceful slope refutes it and points back at
capacity.

## What would actually fix it

Not a bigger state, and not a better optimiser. A **read-time comparison against every stored key
at once** — which is what attention's `softmax(qᵀK)V` is, and what a matrix state gives you: `S q`
compares `q` against all stored keys in one operation because the keys occupy separate rows rather
than being summed into one vector. The relevant difference between a vector state and a matrix state
is not how much they hold; it is that a matrix state supports **n simultaneous matched filters** and
a vector state supports one.

That reframes the level-2 result correctly. The failed experiment added the outer product to the
*output*; the argument says it has to be in the *state*, where the recurrence writes — so that
retrieval is a contraction against separately addressable rows rather than a single filter applied
to a sum.

---

# Addendum IV: the state is a sketch, and stored items are orbits

Addendum III is **retracted**. It argued that a vector state supports exactly one matched filter and
therefore one bit. Holographic Reduced Representations have one vector and one query-dependent
operator and retrieve one of eight items at d = 64 with accuracy 0.892 (`retrieval_phase_transition.txt`).
The limit was never "a vector state"; it was *our* binding operator, which at n_h = 2 sits within
0.96 of the identity by our own published trace law.

## What the equation actually permits

Unrolling the recurrence:

    h_T = sum_t M_{t->T} b_t ,   M_{t->T} = prod_{s>t} a_s T_s

Three consequences, all derivations rather than measurements:

1. **The state is a linear functional of the past — a sketch, not a store.** Sketching theory says
   such an object answers aggregate queries cheaply and point queries only under sparsity or with
   O(n) decoding. Every escape route this project has found is one of those two. That is not a
   coincidence; they are the only ones available.

2. **The query's transport is a common factor.** The query key sits at position T, so `a_T T_T`
   appears in `M_{t->T}` for *every* t and can be factored out. A common rotation reorients all
   stored items together and cannot select one. The entire query-dependent selection capacity
   reduces to `W diag(g_q) T_q`, and at n_h = 2, `T_q` is nearly the identity — so selection rests
   almost entirely on the **diagonal** output gate `silu(og(z))`.

3. **Nothing derived from an item's own key is ever applied to that item's write.** `M_{i->T}`
   depends only on tokens *after* i. The key enters additively inside `b_i` and never as an
   operator. HRR stores `k (*) v`; DeltaNet stores `v k^T`; we store a sum. This is the missing
   term, stated exactly.

## Items are orbits, not points

Because `M_{i->T}` is a product of rotations fixed by whatever tokens happen to follow item i, the
same (key, value) pair lands somewhere different in every sequence. It does not occupy a point; it
sweeps an **orbit**. And beta is measured at a median of 2.000 in trained checkpoints
(`trained_geometry.txt`), making every transport exactly norm-preserving, so that orbit lies on a
**sphere — a closed manifold**, not a contracting one.

This is the precise form of the open/closed question, and it has a quantitative theory behind it:
manifold capacity (Gardner; Chung, Lee & Sompolinsky) shows linear separability collapses as
manifold *radius* grows. The right question is not how many vectors fit in `d_h` dimensions
(exponentially many) but how many **orbits** can be separated.

It also reads the flat n_h sweep, which nothing else does. More reflections bind harder *and* smear
the orbit wider, since there is more rotation between write and read. A trade with an optimum
rather than a knob that should have helped monotonically — and nh = 2, 8, 16, 32 all measured
0.29-0.39 while the hand-built ceiling moved 0.347 -> 0.973.

Measured in `evidence/orbit_manifold.py`; the isolation is exact because the state is linear in the
writes, so changing one item's value changes the state by exactly `M_i (b(v) - b(v'))`.

## The standing against all of this

Eighteen architectural interventions have now failed to move 4-pair recall off 0.32: n_h, depth,
heads, optimiser, gate cap, carving, level-2 signatures, routing, write sparsity, orthogonal axes,
beta_init (the delta-rule regime), multi-axis key-bound writes, query key offset, nonlinear state
(tanh and top-k), and state sparsity.

And two independent hand-built constructions predicted ~0.97 for configurations that trained to
~0.31. **Constructions in this project have no demonstrated predictive value for what gradient
descent finds**, which undercuts the method used to justify most of this document.

arXiv 2609.16183 (Sep 2026) reports the same wall in rank-1 delta-rule and diagonal cells, concludes
the failure is **optimisation rather than expressivity**, and breaks it with a distance curriculum
on the *unchanged* architecture (0.021 -> 1.000, lock-in 1/10 -> 7/10 seeds). It also finds the
short causal convolution dominates, with the architectural gap collapsing from +0.32 to +0.03 once
both families have one: "no class claim survives."

If that transfers, then this entire document is a catalogue of the wrong axis, and the geometry was
never the thing to fix.
