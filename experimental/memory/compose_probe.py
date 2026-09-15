"""CAN THE STATE COMPOSE AT ALL? No training, no model, float64, seconds. (2026-09-10)

Chains sit at the guessing floor at every depth while single-hop binding is near perfect - source 0.995,
exposures 0.984. That pattern has a structural candidate. The state is S = sum v_i k_i^T with k and v coming
from SEPARATE, UNRELATED projections (block_v3.py:172-173). Reading gives S k_a = v_a. To take a second hop the
value must go back in AS A KEY - but a value is not a legal key, so S composed with S means nothing. The state
is a LOOKUP TABLE, and lookup tables do not compose.

What composition needs is a TRANSITION OPERATOR on ONE space: value = the next node's key, through the SAME map.
    S = sum phi(x_{t}) phi(x_{t-1})^T   ->   S phi(a) = phi(b),  S^2 phi(a) = phi(c),  depth d is S^d
Note that naive TYING (v = k) is the wrong fix and gives S = sum k k^T, which maps every node to itself. The
fix is a SHIFT, not a tie.

This measures the mechanism's own property in isolation, before any task touches it - the rule that has now paid
four times. Three constructions, the same chain, the same delta rule, read once and read twice:

  untied      k and v from independent random maps           what the block does today
  tied        v = k                                          the obvious fix, predicted to fail
  shifted     v_t = k_{t+1}, one shared map                  the transition operator

usage: python compose_probe.py
"""
import numpy as np

D, N, TRIALS = 64, 8, 200                     # dk=64, chains of 8 nodes


def build(mode, rng, n=N, d=D):
    """write a chain n0 -> n1 -> ... into a delta-rule state, return (S, phi_of_nodes)"""
    phi = rng.normal(size=(n, d)) / np.sqrt(d)          # one embedding per node, in KEY space
    psi = rng.normal(size=(n, d)) / np.sqrt(d)          # an independent VALUE space
    S = np.zeros((d, d))
    for t in range(n - 1):
        k = phi[t]
        v = {'untied': psi[t + 1], 'tied': phi[t], 'shifted': phi[t + 1]}[mode]
        # the delta rule at beta=1: exact replacement along k. Keys here are near-orthogonal, so this is
        # effectively S += v k^T, but run the real update so nothing is assumed.
        S = S - np.outer(S @ k, k) / (k @ k) + np.outer(v, k) / (k @ k)
    return S, phi, psi


def hop(S, x, k=1):
    for _ in range(k):
        x = S @ x
    return x


def nearest(x, table):
    return int(np.argmax(table @ x / (np.linalg.norm(table, axis=1) * np.linalg.norm(x) + 1e-12)))


def load_test(nchain, depth, d=D, trials=60, cleanup=False):
    """the shifted operator with MANY chains written into ONE state. Composition working on a single chain says
    nothing about whether it survives interference: every extra edge adds an outer product and reading is only
    as clean as the keys are separable. This is the capacity-versus-depth trade the real task actually runs."""
    ok = 0
    for _ in range(trials):
        rng = np.random.default_rng()
        n = nchain * (depth + 1)
        phi = rng.normal(size=(n, d)) / np.sqrt(d)
        S = np.zeros((d, d))
        edges = [(c * (depth + 1) + i, c * (depth + 1) + i + 1)
                 for c in range(nchain) for i in range(depth)]
        rng.shuffle(edges)
        for a, b in edges:
            k = phi[a]
            S = S - np.outer(S @ k, k) / (k @ k) + np.outer(phi[b], k) / (k @ k)
        c = int(rng.integers(0, nchain))
        x = phi[c * (depth + 1)]
        for _ in range(depth):
            x = S @ x
            if cleanup:
                # RE-PERCEIVE between hops. A raw repeated multiply carries every hop's interference into the
                # next one, so error compounds. Snapping the retrieved vector back onto a real percept is error
                # correction, and it is what 'recall the page, then re-read it' does that S^d does not.
                x = phi[nearest(x, phi)]
        ok += (nearest(x, phi) == c * (depth + 1) + depth)
    return ok / trials


if __name__ == '__main__':
    print('CAN THE STATE COMPOSE?  chains of %d nodes, d=%d, %d trials, float64\n' % (N, D, TRIALS))
    print('  %-10s %12s %12s %12s %12s'
          % ('construction', '1 hop', '2 hops', '3 hops', 'depth 7'))
    print('  ' + '-' * 62)
    for mode in ('untied', 'tied', 'shifted'):
        acc = {h: 0 for h in (1, 2, 3, 7)}
        for _ in range(TRIALS):
            rng = np.random.default_rng()
            S, phi, psi = build(mode, rng)
            table = psi if mode == 'untied' else phi
            for h in acc:
                if h >= N:
                    continue
                got = hop(S, phi[0], h)
                acc[h] += (nearest(got, table) == h)
        print('  %-10s %12s %12s %12s %12s'
              % (mode, *['%.3f' % (acc[h] / TRIALS) for h in (1, 2, 3, 7)]))
    print('\n  read: fraction of trials where walking h hops from node 0 lands on node h.')
    print('  1 hop is retrieval and every construction should do it. 2+ hops is COMPOSITION.')
    print('  STATED PRECISELY, because the strong version is false: this shows that with UNRELATED key and')
    print('  value maps the state itself cannot be walked twice - untied multi-hop lands at chance, 1/8.')
    print('  It does NOT show that no trained W_k, W_v could ever compose. A 2-layer model can build a')
    print('  second hop ACROSS LAYERS - roughly one layer per hop - so untied reaches depth ~NL and no')
    print('  further. The shifted operator reaches ANY depth by repeated application of ONE state, which is')
    print('  what a depth loop would walk. That is the difference: composition by DEPTH versus by ALGEBRA.')

    print('')
    print('  SHIFTED UNDER LOAD - many chains written into ONE state, walked the full depth, d=%d' % D)
    print('  %-10s %s' % ('chains', '  '.join('%7s' % ('d=%d' % dd) for dd in (2, 3, 4, 6))))
    print('  ' + '-' * 44)
    for nc in (1, 4, 8, 16, 32):
        print('  %-10d %s' % (nc, '  '.join('%7.3f' % load_test(nc, dd) for dd in (2, 3, 4, 6))))
    print('')
    print('  edges stored = chains x depth, and the task runs 8 chains, so that is the row that matters.')
    print('  1/chains is the guessing floor for each column. Composition working on ONE chain proves nothing')
    print('  about a loaded state: this is where interference, not algebra, decides.')
