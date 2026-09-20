# Noise as a resource: the thread, and where each strand stands

This project has demanded **exact reads** everywhere. Every memory mechanism here retrieves a vector
and uses it directly. Four separate lines of enquiry on 2026-09-20 converged on the same objection:
the systems that actually achieve high density do the opposite — they accept a *noisy* readout and
pay for it with a correction step. This note tracks that idea and the status of each strand, so it
does not get rediscovered a fifth time.

Claims are tagged **MEASURED**, **ESTABLISHED** (someone else's result, checked), or **UNTESTED**.

---

## The four arrivals

**1. Flash storage.** ESTABLISHED. Density went from 1 bit per cell to 5 — SLC to PLC — by
distinguishing 32 voltage levels where there were 2. QLC's raw bit error rate would be unusable as a
storage device. It works because error correction got strong enough to absorb it: Reed-Solomon, then
BCH, then LDPC. The density was bought by *degrading the physical channel and fixing it downstream*.

**2. Holographic storage.** ESTABLISHED. A hologram's reconstruction is intrinsically noisy —
speckle, finite aperture, crosstalk between angle-multiplexed pages. Holographic data-storage
systems are built around error correction. Noise is not a defect of the implementation; it is what
distributed storage *is*. Which is why Plate's Holographic Reduced Representations are "Reduced":
lossy by construction, and the literature pairs them with a **cleanup** step against a codebook.

**3. Compressed sensing.** ESTABLISHED, see `evidence/results/literature_compressed_sensing.txt`.
Ganguli & Sompolinsky (2010) and Charles, Yap & Rozell (2014) show a recurrent state can hold a
recoverable history *longer than its own dimension* when the input is sparse. Recovery is L1
minimisation — a denoising procedure. The information is present in a form that looks like noise
until something reconstructs it.

**4. Diffusion.** ESTABLISHED, and the connection is structural rather than superficial. A diffusion
model learns to map noisy samples back onto the data manifold; a cleanup memory maps a noisy
retrieval onto the nearest legal codeword. Both are projection onto a learned manifold. Denoising is
how you *learn* a manifold, which is the same object this project went looking for in
`docs/memory_geometry.md`.

---

## Strand 1 — cleanup on the read.  UNDER TEST

`evidence/hrr_cleanup.py`, running.

The vector cell in `cell_shootout.py` unbinds and hands the result straight to a linear layer. The
vector-symbolic tradition treats unbinding as returning a *noisy* vector and recovers capacity with
a cleanup step. We omitted it, then recorded "the vector route is strictly dominated" as a verdict
on the architecture. Falsifier registered: gains under +0.05 at both 4 and 8 pairs closes it.

Honest cost, stated before the result: soft cleanup is a softmax over a codebook. It is a softmax
over a **constant-size set**, so per-token cost stays constant in context — but any gain is partly
credit to a softmax and the writeup must say so.

## Strand 2 — noise injected during training.  UNTESTED

The flash analogy runs the other way too. QLC does not merely tolerate noise at read time; the whole
system is *designed around* the error rate it will face. We have never trained this model under
injected state noise.

The specific reason it might matter here: our state is a sum of transported items, so retrieval is
inherently corrupted by crosstalk from everything else stored. Training under exactness lets the
model learn codes that are brittle to exactly that crosstalk. Injecting noise into the recurrent
state during training would penalise codes whose items sit too close together — which is precisely
what capacity requires.

NOT YET CHECKED against the literature. Noise injection in RNNs is old (Jim et al. 1996 and after)
and denoising autoencoders are standard, so the prior should be that this is known. Check before
treating it as a direction — this project has three retractions today from not doing that.

## Strand 3 — the gate is a noise problem too.  MEASURED, fix identified

`evidence/results/literature_gate_lower_bound.txt`. Not obviously part of this thread, but it is the
same shape. Our three attempts to install long retention all enforced the bound by **clamping**, and
`torch.minimum` has zero gradient wherever it binds — the constraint stops teaching the instant it
applies. HGRN's additive reparameterisation keeps it differentiable throughout, which is what their
"mitigates saturated gates" claim is about.

A hard constraint is an infinitely sharp one. A soft, always-differentiable one is the same move as
accepting a noisy channel: give up exactness, keep the gradient. Clamps have now cost this project
two experiments — this one and the flip-flop test, recorded void because its clamp never fired.

---

## What would close the thread

If Strand 1 fails its falsifier, cleanup is not the missing piece and the matrix decision stands
untouched. If it succeeds, then `matrix_decision.txt`'s "strictly dominated" was measured on a cell
missing part of its own memory, and that sentence needs rewriting rather than defending.

Either way Strand 2 is independent of the outcome and still untested, and it is cheap.

## Related

- [`evidence/results/literature_compressed_sensing.txt`](../evidence/results/literature_compressed_sensing.txt)
- [`evidence/results/literature_tomography.txt`](../evidence/results/literature_tomography.txt) — where the holographic framing was falsified
- [`evidence/results/literature_gate_lower_bound.txt`](../evidence/results/literature_gate_lower_bound.txt)
- [`docs/learning_machine.md`](learning_machine.md)
