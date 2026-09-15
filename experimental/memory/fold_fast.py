"""THE ORIGAMI QUESTION, made cheap enough to actually finish.

Same fork as fold_scaling.py, which was too slow: it scanned rank upward from 1, 120 trials each,
with an SVD at every timestep. This bisects on rank, uses d=64 (the cliff was IDENTICAL at d=64 and
d=128, so the ambient dimension demonstrably does not set it), and 60 trials - locating a threshold,
not measuring an effect size.

    rank ~ log(N)   the fold is REAL. 1000 items cost ~10 dimensions.
    rank ~ N        rank is CAPACITY renamed. No origami.

Our own capacity work (a dh x dh state holds ~dh associations) predicts LINEAR.
The origami intuition predicts SUBLINEAR. They disagree, which is why it is worth measuring.

ANCHORS, checked first: at rank = d the answer MUST be 1.000, at rank 1 it MUST be near chance.
If those two ends do not hold, the speedup broke something and no row below is readable.
"""
import numpy as np

D, TRIALS = 64, 60


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


def acc_at(n, rank, hops=2, trials=TRIALS, d=D):
    ok = 0
    for _ in range(trials):
        rng = np.random.default_rng()
        phi = rng.normal(size=(n, d)); phi /= np.linalg.norm(phi, axis=1, keepdims=True)
        S = np.zeros((d, d))
        for t in range(n - 1):
            k, v = phi[t], phi[t + 1]
            S = S - np.outer(S @ k, k) + np.outer(v, k)
            if rank < d:
                U, sv, Vt = np.linalg.svd(S, full_matrices=False)
                sv[rank:] = 0.0
                S = (U * sv) @ Vt
        got = walk(S, phi[0], hops)
        ok += (got is not None and nearest(got, phi) == hops)
    return ok / trials


def min_rank(n, target=0.95, hops=2, d=D):
    """bisect for the smallest rank clearing `target` (accuracy is monotone in rank)"""
    if acc_at(n, d, hops) < target:
        return None
    lo, hi = 1, d
    while lo < hi:
        mid = (lo + hi) // 2
        if acc_at(n, mid, hops) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


if __name__ == '__main__':
    print('ORIGAMI: does required rank grow like log(N) or like N?   d=%d, %d trials, 2-hop, thr 0.95'
          % (D, TRIALS))
    print()
    print('  ANCHORS (if these fail, nothing below is readable):')
    a_full = acc_at(8, D)
    a_one = acc_at(8, 1)
    print('    N=8 rank=%d (lossless): %.3f   %s' % (D, a_full, 'OK' if a_full > 0.95 else '*** BROKEN ***'))
    print('    N=8 rank=1            : %.3f   %s' % (a_one, 'OK' if a_one < 0.40 else '*** BROKEN ***'))
    if not (a_full > 0.95 and a_one < 0.40):
        raise SystemExit('  anchors failed - stopping')
    print()
    print('  %-6s %-10s %-10s %-12s' % ('N', 'min rank', 'rank/N', 'rank/log2(N)'))
    print('  ' + '-' * 44)
    rows = []
    for n in (4, 6, 8, 12, 16, 24, 32):
        r = min_rank(n)
        if r is None:
            print('  %-6d %-10s' % (n, 'none'))
            continue
        rows.append((n, r))
        print('  %-6d %-10d %-10.2f %-12.2f' % (n, r, r / n, r / np.log2(n)), flush=True)
    print()
    if len(rows) >= 3:
        ns = np.array([a for a, _ in rows], float)
        rs = np.array([b for _, b in rows], float)
        lin = np.polyfit(ns, rs, 1); log = np.polyfit(np.log2(ns), rs, 1)
        rl = float(np.sum((rs - np.polyval(lin, ns)) ** 2))
        rg = float(np.sum((rs - np.polyval(log, np.log2(ns))) ** 2))
        print('  residual  linear %.2f   log %.2f' % (rl, rg))
        print('  rank ~ %.2f*N + %.1f      rank ~ %.2f*log2(N) + %.1f' % (lin[0], lin[1], log[0], log[1]))
        print()
        if rl < rg:
            print('  LINEAR. Rank is capacity renamed - 1000 items would need ~%d dims. No fold.'
                  % max(1, int(lin[0] * 1000 + lin[1])))
        else:
            print('  LOGARITHMIC. The fold is real - 1000 items need ~%d dims.'
                  % max(1, int(log[0] * np.log2(1000) + log[1])))
