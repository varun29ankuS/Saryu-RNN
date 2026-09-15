"""THE ORIGAMI QUESTION: does the required rank grow like log(N) or like N?

The latent cliff sat at rank ~7 for BOTH d=64 and d=128 - an ABSOLUTE rank, not a fraction of the
ambient dimension. That is the Johnson-Lindenstrauss signature: the dimension needed to keep N
points separable depends on N, not on the size of the space they started in. "Fold the page, keep
the map readable."

Which makes the real question a fork:

    rank ~ log(N)   the fold WINS. A store of 1000 items needs ~10 dimensions, not 1000.
    rank ~ N        rank is just CAPACITY renamed and there is no origami here.

Our own capacity work (a dh x dh state holds ~dh associations) predicts LINEAR. The origami
intuition predicts SUBLINEAR. They disagree, so measure it.

Method: build a tied chain of N nodes, squeeze the state to rank r each step, and find the
smallest r whose 2-hop composition still holds at >= 0.95. Sweep N. Read the growth.
"""
import numpy as np

D_DEFAULT, TRIALS = 128, 120


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


def build(rng, n, d, rank):
    phi = rng.normal(size=(n, d)); phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    S = np.zeros((d, d))
    for t in range(n - 1):
        k, v = phi[t], phi[t + 1]
        S = S - np.outer(S @ k, k) + np.outer(v, k)
        if rank is not None and rank < d:
            U, sv, Vt = np.linalg.svd(S, full_matrices=False)
            sv[rank:] = 0.0
            S = (U * sv) @ Vt
    return S, phi


def acc_at(n, d, rank, hops=2, trials=TRIALS):
    ok = 0
    for _ in range(trials):
        rng = np.random.default_rng()
        S, phi = build(rng, n, d, rank)
        got = walk(S, phi[0], hops)
        ok += (got is not None and nearest(got, phi) == hops)
    return ok / trials


def min_rank(n, d, target=0.95, hops=2):
    """smallest rank whose `hops`-hop accuracy still clears `target`"""
    lo, hi, best = 1, d, None
    for r in range(1, min(d, 4 * n + 8) + 1):
        if acc_at(n, d, r, hops) >= target:
            return r
    return None


if __name__ == '__main__':
    print('DOES THE FOLD SCALE LIKE log(N) OR LIKE N?   d=%d, %d trials, 2-hop, threshold 0.95'
          % (D_DEFAULT, TRIALS))
    print('  N = nodes in the chain (so N-1 stored edges)')
    print()
    print('  %-6s %-12s %-12s %-12s %s' % ('N', 'min rank', 'rank/N', 'rank/log2(N)', 'reading'))
    print('  ' + '-' * 66)
    rows = []
    for n in (4, 6, 8, 12, 16, 24, 32):
        r = min_rank(n, D_DEFAULT)
        if r is None:
            print('  %-6d %-12s' % (n, 'none <= %d' % D_DEFAULT))
            continue
        rows.append((n, r))
        print('  %-6d %-12d %-12.2f %-12.2f' % (n, r, r / n, r / np.log2(n)))
    print()
    if len(rows) >= 3:
        ns = np.array([a for a, _ in rows], float)
        rs = np.array([b for _, b in rows], float)
        lin = np.polyfit(ns, rs, 1)
        log = np.polyfit(np.log2(ns), rs, 1)
        res_lin = float(np.sum((rs - np.polyval(lin, ns)) ** 2))
        res_log = float(np.sum((rs - np.polyval(log, np.log2(ns))) ** 2))
        print('  fit residuals:  linear in N  %.2f      logarithmic in N  %.2f' % (res_lin, res_log))
        print('  slope: rank ~ %.2f*N + %.1f     or     rank ~ %.2f*log2(N) + %.1f'
              % (lin[0], lin[1], log[0], log[1]))
        print()
        if res_lin < res_log:
            print('  -> LINEAR wins. Rank is capacity renamed; there is no origami win. A store of')
            print('     1000 items needs ~%d dimensions, which is no better than listing them.' % int(lin[0] * 1000))
        else:
            print('  -> LOGARITHMIC wins. The fold is real: a store of 1000 items needs ~%d dimensions.'
                  % int(log[0] * np.log2(1000) + log[1]))
    print()
    print('  CAVEAT: this measures the LINEAR-ALGEBRAIC rank of the state, which is metric')
    print('  embedding (Johnson-Lindenstrauss), not topology in the strict sense. Homology and')
    print('  genus deliberately discard distance; the whole point here is to KEEP it.')
