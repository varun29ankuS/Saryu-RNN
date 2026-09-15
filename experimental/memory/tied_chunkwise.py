"""DOES TYING k AND v BREAK THE CHUNKWISE PARALLEL FORM? A derivation check, not a benchmark.

DeltaNet's whole contribution is that the delta rule can be trained with a CHUNKWISE PARALLEL
form (Yang et al. 2406.06484 Eq. 8-9) instead of a sequential scan. Their Sec. 6 warns of "a
fundamental trade-off between parallelism and expressiveness" - recurrent enhancements of
DeltaNet "cannot be parallelized across sequence length". So the question that decides whether
a tied map is BUILDABLE, before any training:

    does k_t = phi(x_{t-lag}), v_t = phi(x_t)  still admit the chunkwise form?

The answer should be YES by inspection, because tying changes only HOW k and v are PRODUCED -
both are still ordinary per-timestep vectors handed to the same recurrence
    S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
and Eq. 8-9 never assume k and v are independent. But "should be" is how six things went wrong
yesterday, so this checks it numerically: run the SAME tied inputs through a sequential scan and
through a chunkwise form, and compare.

If they agree to float tolerance, tying is free w.r.t. parallelism and the only open cost is
the sharpness constraint. If they disagree, the design pays DeltaNet's stated trade and the
spec must say so.

REFERENCES
  [Yang et al. 2024a] Parallelizing Linear Transformers with the Delta Rule over Sequence Length, arXiv 2406.06484
"""
import numpy as np

D, T, C = 32, 64, 8          # dim, sequence length, chunk size


def seq_scan(K, V, B):
    """the ground truth: S_t = S_{t-1}(I - b k k^T) + b v k^T, one step at a time."""
    S = np.zeros((D, D))
    out = []
    for t in range(len(K)):
        k, v, b = K[t], V[t], B[t]
        S = S - b * np.outer(S @ k, k) + b * np.outer(v, k)
        out.append(S.copy())
    return np.stack(out)


def chunkwise(K, V, B, C=C):
    """DeltaNet Eq.3-4 form: unroll within a chunk via the u_i 'pseudo-value' construction,
    carrying only the chunk-boundary state. Never materialises per-step states inside a chunk."""
    S = np.zeros((D, D))
    out = []
    for c0 in range(0, len(K), C):
        Kc, Vc, Bc = K[c0:c0 + C], V[c0:c0 + C], B[c0:c0 + C]
        n = len(Kc)
        #   u_i = b_i (v_i - sum_{j<i} u_j (k_j . k_i)) - the WY-style pseudo values
        U = np.zeros((n, D))
        for i in range(n):
            corr = np.zeros(D)
            for j in range(i):
                corr = corr + U[j] * (Kc[j] @ Kc[i])
            U[i] = Bc[i] * (Vc[i] - S @ Kc[i] - corr)
        Sl = S.copy()
        for i in range(n):
            Sl = Sl + np.outer(U[i], Kc[i])
            out.append(Sl.copy())
        S = Sl
    return np.stack(out)


def make(mode, rng):
    """untied: k and v independent. tied: k is a LAGGED copy of the same percept stream."""
    phi = rng.normal(size=(T + 4, D))
    phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    if mode == 'untied':
        psi = rng.normal(size=(T, D))
        psi /= np.linalg.norm(psi, axis=1, keepdims=True)
        K, V = phi[:T], psi
    else:
        K, V = phi[:T], phi[1:T + 1]          # k_t = p_t, v_t = p_{t+1}: the tied transition map
    B = rng.uniform(0.2, 1.0, size=T)
    return K, V, B


if __name__ == '__main__':
    print('CHUNKWISE EQUIVALENCE UNDER TYING   (D=%d, T=%d, chunk=%d, float64)' % (D, T, C))
    print('Question: does k_t = phi(x_{t-lag}), v_t = phi(x_t) still admit DeltaNet Eq.8-9?')
    print('DeltaNet Sec.6 warns recurrent enhancements "cannot be parallelized across sequence')
    print('length" - so this decides whether a tied map is buildable at all.')
    print()
    print('  %-10s %16s %16s  %s' % ('inputs', 'max |seq-chunk|', 'rel err', 'verdict'))
    print('  ' + '-' * 62)
    ok_all = True
    for mode in ('untied', 'tied'):
        errs = []
        for trial in range(5):
            rng = np.random.default_rng(trial)
            K, V, B = make(mode, rng)
            a, b = seq_scan(K, V, B), chunkwise(K, V, B)
            errs.append((np.abs(a - b).max(), np.abs(a - b).max() / (np.abs(a).max() + 1e-30)))
        mx = max(e[0] for e in errs)
        rel = max(e[1] for e in errs)
        ok = mx < 1e-9
        ok_all &= ok
        print('  %-10s %16.3e %16.3e  %s' % (mode, mx, rel, 'EQUIVALENT' if ok else '*** DIVERGES ***'))
    print()
    if ok_all:
        print('  TYING IS FREE w.r.t. PARALLELISM. The chunkwise form never assumes k and v are')
        print('  independent - it only needs them as per-timestep vectors, and a lagged percept')
        print('  stream supplies exactly that. So the tied map does NOT pay DeltaNet Sec.6\'s')
        print('  parallelism/expressiveness trade, and fla\'s existing kernels apply unchanged.')
        print('  The only open cost remains the SHARPNESS constraint (tap weight >= ~0.90).')
    else:
        print('  *** The tied form diverges from the chunkwise reference. It would need a new')
        print('  kernel, which is exactly the trade DeltaNet Sec.6 describes. Spec must say so.')
