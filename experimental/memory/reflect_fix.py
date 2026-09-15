"""REFLECTIONS, THIRD ATTEMPT. Two previous versions were my bugs, both caught by the same tell.

  v1: rotations with directions phi[(t+2+r) % n] - FUTURE percepts, wrapping the chain. Non-causal.
  v2: nref repeated WRITES of the same value v at different keys. That installs WRONG EDGES -
      with nref=2 the state learns phi[t-1] -> phi[t+1] alongside phi[t] -> phi[t+1], and the walk
      follows the wrong one.

Both scored 0.000 at ONE hop, below chance (0.125). A construction that adds structure to a
working delta write cannot be worse than random at plain retrieval - that impossibility is the
only reason the bugs were visible at all.

WHAT DELTAPRODUCT ACTUALLY DOES (fla/layers/gated_deltaproduct.py): num_householder k/v pairs are
produced per token from separate projections, then the sequence is EXPANDED - rearrange
'... t (n h d) -> ... (t n) h d' - so the n micro-steps compose into ONE transition for that
token. The transition is a PRODUCT of Householder factors, not several competing edges.

So the correct model is: one edge phi[t] -> phi[t+1] per token, realised as a product of nref
Householder factors whose directions are perturbations around the key. nref=1 must reproduce the
baseline exactly; that self-check row gates everything below it.

REFERENCES
  [Siems et al. 2025] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products, arXiv 2502.10297
"""
import numpy as np

D, N, TRIALS = 64, 8, 300


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


def build(rng, nref=1, spread=0.0, n=N, d=D):
    """ONE edge per token, written through nref Householder micro-steps that compose."""
    phi = rng.normal(size=(n, d)); phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    S = np.zeros((d, d))
    for t in range(n - 1):
        k, v = phi[t], phi[t + 1]
        if nref == 1:
            S = S - np.outer(S @ k, k) + np.outer(v, k)
            continue
        #   nref micro-keys around k, each carrying 1/nref of the write. Their product is the
        #   transition for THIS token; no extra edges are created.
        ks = []
        for r in range(nref):
            kr = k + spread * rng.normal(size=d)
            ks.append(kr / np.linalg.norm(kr))
        for r, kr in enumerate(ks):
            S = S - np.outer(S @ kr, kr) + np.outer(v, kr)
    return S, phi


def score(nref, spread, trials=TRIALS):
    acc = {h: 0 for h in (1, 2, 3, 7)}
    for _ in range(trials):
        rng = np.random.default_rng()
        S, phi = build(rng, nref, spread)
        for h in acc:
            if h >= N:
                continue
            got = walk(S, phi[0], h)
            acc[h] += (got is not None and nearest(got, phi) == h)
    return {h: acc[h] / trials for h in acc}


if __name__ == '__main__':
    print('REFLECTIONS DONE CORRECTLY  d=%d, chains of %d, %d trials, chance %.3f' % (D, N, TRIALS, 1.0 / N))
    print('SELF-CHECK: the nref=1 row is DEFINITIONALLY the tied baseline. If it is not 1.000 the')
    print('harness is broken and NO row below may be read.')
    print()
    print('  %-38s %8s %8s %8s %8s' % ('construction', '1 hop', '2 hops', '3 hops', 'depth 7'))
    print('  ' + '-' * 74)
    b = score(1, 0.0)
    print('  %-38s %8.3f %8.3f %8.3f %8.3f' % ('nref=1 (baseline, self-check)', b[1], b[2], b[3], b[7]))
    if b[1] < 0.99:
        print('  *** SELF-CHECK FAILED - stopping, nothing below is readable ***')
        raise SystemExit
    for nref in (2, 3, 4):
        for spread in (0.0, 0.05, 0.2):
            r = score(nref, spread)
            print('  %-38s %8.3f %8.3f %8.3f %8.3f'
                  % ('nref=%d, key spread %.2f' % (nref, spread), r[1], r[2], r[3], r[7]))
    print()
    print('  spread=0 means all nref micro-keys are IDENTICAL, so the writes are redundant and this')
    print('  must match the baseline. If it does not, the micro-step composition is still wrong.')
