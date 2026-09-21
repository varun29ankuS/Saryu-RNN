# What this architecture says about building a learning machine

A note, not a result. Every claim below is tagged **MEASURED** (we ran it), **DERIVED** (it follows
from the algebra), or **CONJECTURE** (it does not yet follow from anything). The tags are the point
of the document: this is the kind of writeup where this project has overclaimed before, and
`evidence/CLAIMS.md` records thirteen retractions to prove it.

---

## 1. What the architecture is, after today

Two operations, and the central lesson of the last 170 runs is that they are **different
operations**:

    transport   h_t = (1-g) H_2 H_1 h_{t-1} + g c_t     where am I      order, position, relation
    store       S <- S(I - b k k^T) + b v k^T ,  o = Sq  what do I know  selection, facts

**DERIVED.** Unrolling the recurrence gives `h_T = a_T T_T [Σ_t M_{t→T−1} b_t] + b_T`. The query's
transport is a *common factor* over everything stored: it reorients all items together, so it cannot
pick one out of a sum. Selection needs an item axis, and `Sq = Σ_i v_i (k_i·q)` has one.

**MEASURED** (`evidence/delta_reference.py`, `runs/dref-*`). A faithful DeltaNet solves 4-pair
MQAR at 1.000 on three seeds; the shipped transport-plus-gate sits at 0.32. Twenty-one
interventions that changed the transport, the write, or the training order all returned ~0.32,
because none of them touched the read.

**RETRACTED.** This paragraph previously cited 0.984 from `results/matrix_decision.txt` together
with a matrix-versus-vector comparison. Neither had a run behind it. The logged runs of that
experiment scored 0.133–0.164, because the cell was missing five components the reference
implementation has. The conclusion above is re-earned from new logged runs; the *comparison* is
not, and is withdrawn.

The design principle worth keeping: **a recurrence that conflates tracking with storage gets
neither.** Saryu had strong machinery for the first and none for the second.

---

## 2. What is measured about the way it learns

**Failure is a coarser valid theory, not noise.** MEASURED, `evidence/word_problem.py`,
`which_subgroup.py`. Train the transport on a finite group's word problem and it either solves it
exactly at any length or lands on `1/|N|` for a normal subgroup `N` — it has learned the quotient
`G/N`, gets the coset right, and guesses inside it. Twelve runs across Q₈, S₄ and A₅ land on a rung
of their own group, worst deviation 0.021. On Q₈, where three different normal subgroups share the
same accuracy, coset consistency picks exactly one.

This is the most interesting property in the project. Under insufficient capacity the model does not
degrade into noise or memorisation — it degrades into a **structurally valid abstraction** and
remains correct at that coarser level, at any sequence length.

**The learned theory can be read out without labels.** MEASURED, `evidence/character_table.py`. The
character norm `⟨χ,χ⟩`, computed from traces alone with no group table and no labels, identifies
which quotient the model settled on. Over sixteen seeds it agrees with the independent error-based
reading on all seven runs that are genuine homomorphisms. You can ask a trained transport what
theory it currently holds and check the answer.

**The memory hierarchy appears on its own, and dies when installed by hand.** MEASURED,
`results/trained_geometry.txt` item 5. Gate retention `1/g` per head in the trained 25M model:
layers 0–1 sit at 11–17 tokens, layer 2 reaches 86 and 132, layer 3 spans 15–49. Three separate
attempts to install this ladder by hand — fixed timescales, a frozen gate bias, a gate ceiling —
all collapsed to 2–3 tokens within 1000 steps (`gate_timescales.txt`). Where we intervened it died;
where we left it alone it appeared.

**Organisation beats size at a fixed budget.** ~~MEASURED~~ **RETRACTED.** This rested on a
crossover table in `results/matrix_decision.txt` with no run behind it — no experiment in this
project's history has ever paired a matrix or vector cell with 8 pairs. The construction in
`evidence/arity_test.py` still stands as a construction, and this project's own record says
constructions here have no demonstrated predictive value for trainability. Separating size from
factorisation remains an interesting claim and an untested one.

---

## 3. The bridge to cheap intelligence — and why it might not hold

**CONJECTURE.** Learning from few examples means building abstractions rather than storing
instances. Quotient formation is abstraction formation. If capacity pressure on this transport
reliably produces the coarsest consistent theory, and the character norm lets us read which theory
that is, then the architecture would give three things a learning machine wants: it abstracts under
pressure instead of memorising, its abstraction is verifiable from the inside, and it stays exactly
correct at the level it has actually learned.

**Why it might not transfer, stated plainly.** The mechanism is specific. The transport is an
orthogonal representation of a group, so under capacity pressure the only *consistent* solutions are
homomorphisms, and a homomorphism that is not injective factors through a quotient. The lattice of
normal subgroups is what makes the failure discrete, and it is a property of the group, not of the
model. **Language is not a group.** It has no normal subgroups and no quotient lattice, so there is
no argument yet that anything analogous happens on text. Everything in section 2 was measured on
synthetic algebra, at 5M and 25M parameters, on one corpus.

The honest statement is: we have a mechanism that turns capacity pressure into valid abstraction in
a setting where "valid abstraction" is mathematically defined. Whether that setting is a model of
anything else is unknown.

---

## 4. What has not been measured at all

Nothing in this repository measures sample efficiency. Not one file mentions data efficiency,
learn-once evaluation, or intelligence per parameter; the 170 runs all measured recall capacity on a
synthetic task, which is a capability axis, not an efficiency axis. The stated goal of the project —
train on one CPU in days, 100–1000× fewer parameters, learn from reading something once — has had no
experiment run against it. Fixing the read buys parity with published linear-attention models on
recall; it does not by itself move sample efficiency at all.

**Experiment 1, pre-registered.** The smallest honest learn-once measurement: fixed small token
budget, matched parameter count, Saryu against a transformer and a GRU in the same block, loss and
downstream recall as a function of tokens seen rather than steps.

**Falsifier.** If the transformer reaches the same loss in the same or fewer tokens at equal
parameters, the efficiency claim for this architecture is dead as stated, and the case for Saryu
rests on the group-theoretic results and the constant-cost inference alone — not on learning more
from less.

And that fallback is weaker than it was. DeepSeek-V4.1-Flash reports 890 bytes per token of KV
cache and near-constant decode FLOPs out to a million tokens, so "constant-cost inference" is no
longer a property only a recurrence can offer — it is being engineered into attention, hard, by
teams with very large resources. What a fixed state still has is that it does not grow *at all*,
which matters where memory is bounded and nowhere else. If the efficiency claim falls too, what is
left is the group-theoretic work and an on-device argument
([`evidence/results/literature_kv_compression.txt`](../evidence/results/literature_kv_compression.txt)).

---

## Related

- [`evidence/CLAIMS.md`](../evidence/CLAIMS.md) — every claim made here and what caught the wrong ones
- [`evidence/results/matrix_decision.txt`](../evidence/results/matrix_decision.txt) — the architecture decision
- [`evidence/results/trained_geometry.txt`](../evidence/results/trained_geometry.txt) — measurements on the shipped checkpoint
- [`docs/memory_geometry.md`](memory_geometry.md) — the geometry the memory line explored
