"""DO REFLECTIONS AND A LATENT BOTTLENECK COMPOSE? Pure numerics, no training, seconds.

Two proposed improvements, both of which already exist in fla - so the question is not whether to
invent them but whether they SURVIVE the tied map, which is the only thing that is ours.

  REFLECTIONS  fla's gated_deltaproduct takes num_householder (default 2): "Generalized version of
               GatedDoubleDeltaNet that supports arbitrary number of householder transformations."
               Saryu's ORIGINAL block was exactly this - n_h=2 reflections, beta = 1 - cos(theta).
               A Householder product prod_i (I - b_i u_i u_i^T) is near-orthogonal. Our tied map
               needs S to be a TRANSITION OPERATOR so S^d walks d hops. Do they coexist?
  LATENT       fla's mla.py is DeepSeek MLA (2405.04434), kv_lora_rank=512 - a low-rank bottleneck
               on the cache. For us: project the state through rank r. Composition needs S^d to
               stay meaningful, and a rank-r bottleneck may destroy it. At what r does it break?

Same discipline as compose_probe and sharpness_sweep: measure the mechanism in isolation BEFORE
any training, because a null after training cannot tell "mechanism absent" from "did not learn".

REFERENCES
  [Siems et al. 2025] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products, arXiv 2502.10297
  [DeepSeek-AI 2024] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model, arXiv 2405.04434
"""
import numpy as np

D, N, TRIALS = 64, 8, 200


def nearest(x, table):
    return int(np.argmax(table @ x / (np.linalg.norm(table, axis=1) * np.linalg.norm(x) + 1e-12)))


def walk(S, x, h):
    for _ in range(h):
        x = S @ x
        n = np.linalg.norm(x)
        if n < 1e-10:
            return None
        x = x / n
    return x


def build(rng, nref=1, rank=None, n=N, d=D):
    """TIED chain written with `nref` Householder reflections per step, optional rank-r state."""
    phi = rng.normal(size=(n, d)); phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    S = np.zeros((d, d))
    for t in range(n - 1):
        k, v = phi[t], phi[t + 1]
        #   CORRECTED 2026-09-14. The first version applied arbitrary rotations with directions
        #   phi[(t+2+r) % n] - FUTURE percepts, wrapping around the chain. Non-causal, and not
        #   what DeltaProduct does. It scored 0.000 at ONE hop, below chance, which is impossible
        #   for a construction that merely adds a rotation to a working write - that impossibility
        #   is what exposed it.
        #   fla's gated_deltaproduct expands the SEQUENCE: num_householder k/v pairs per token,
        #   each applied as its own delta write. So nref writes per step, all causal, all from the
        #   percept stream. At nref=1 this must reproduce the baseline exactly.
        for r in range(nref):
            kr = phi[max(0, t - r)]                     # r-th key: this token's lag-r percept
            S = S - np.outer(S @ kr, kr) + np.outer(v, kr)
        if rank is not None:                            # LATENT: squeeze the state through rank r
            U, sv, Vt = np.linalg.svd(S, full_matrices=False)
            sv[rank:] = 0.0
            S = (U * sv) @ Vt
    return S, phi


def score(nref=1, rank=None, trials=TRIALS):
    acc = {h: 0 for h in (1, 2, 3, 7)}
    for _ in range(trials):
        rng = np.random.default_rng()
        S, phi = build(rng, nref, rank)
        for h in acc:
            if h >= N:
                continue
            got = walk(S, phi[0], h)
            acc[h] += (got is not None and nearest(got, phi) == h)
    return {h: acc[h] / trials for h in acc}


if __name__ == '__main__':
    print('REFLECTIONS AND LATENT, ON TOP OF THE TIED MAP   d=%d, chains of %d, %d trials' % (D, N, TRIALS))
    print('chance = 1/%d = %.3f. Baseline: tied with 1 reflection is the `shifted` row, known 1.000.' % (N, 1.0 / N))
    print()
    print('  %-34s %8s %8s %8s %8s' % ('construction', '1 hop', '2 hops', '3 hops', 'depth 7'))
    print('  ' + '-' * 70)
    base = score(nref=1)
    print('  %-34s %8.3f %8.3f %8.3f %8.3f' % ('tied, 1 reflection (baseline)', base[1], base[2], base[3], base[7]))
    for nref in (2, 3, 4):
        r = score(nref=nref)
        print('  %-34s %8.3f %8.3f %8.3f %8.3f' % ('tied, %d reflections' % nref, r[1], r[2], r[3], r[7]))
    print()
    print('  LATENT BOTTLENECK - state squeezed to rank r each step (d=%d, so r=%d is lossless):' % (D, D))
    for rank in (64, 32, 16, 8, 4):
        r = score(nref=1, rank=rank)
        print('  %-34s %8.3f %8.3f %8.3f %8.3f' % ('tied, rank-%d state' % rank, r[1], r[2], r[3], r[7]))
    print()
    print('  READ: 1 hop is retrieval, 2+ is COMPOSITION - the only thing that is ours.')
    print('  If extra reflections hold composition, DeltaProduct-style multi-reflection is free to adopt.')
    print('  If a rank-r state holds it down to small r, an MLA-style latent buys memory for nothing.')
    print('  Either that is measured here, or it is an opinion - and opinions have cost us all day.')
