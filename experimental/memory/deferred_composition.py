"""DEFERRED COMPOSITION: store a graph in arbitrary order, then walk it at query time.

I twice mis-framed this, so state the distinction precisely:

  STREAMING STATE TRACKING (Grazzi et al. 2411.12537, and everything they prove):
      h_t = A(x_t) h_{t-1}
      The operator comes from the CURRENT TOKEN. Transitions must arrive IN ORDER; the state
      accumulates a running product. You CANNOT ask about a path you were not shown in sequence.
      Their Theorem 3 is about implementing an FSA whose transitions are the INPUT.

  DEFERRED COMPOSITION (what a tied/transport state does):
      S accumulates EDGES in any order; queries walk S^d at READ time.
      Arrival order is irrelevant - the graph is in the state, and composition happens later.

These are different capabilities. Their framework does not address the second, and their own
A.4.1 sets the split up without exploiting it: d=1 vector states suffice for tracking, while a
matrix state is "very important for associative recall".

This measures the regime where only deferred composition can work: edges SHUFFLED, then a d-hop
query. Streaming tracking is structurally unable to answer - there is no order to accumulate.

  untied  (every fla layer)  stores edges as a LOOKUP TABLE. S.S is meaningless -> chance past 1 hop.
  tied    (ours)             stores edges as a TRANSITION OPERATOR -> S^d walks d hops.

The control that makes it honest: a SEQUENTIAL reader that sees the same shuffled edges and
accumulates them in arrival order, i.e. what streaming tracking would compute. If that lands at
chance while tied does not, the two mechanisms are genuinely different, not relabelled.

REFERENCES
  [Grazzi et al. 2024] Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues, arXiv 2411.12537
"""
import numpy as np

D, TRIALS = 64, 300


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


def trial(rng, nchain, depth, mode, d=D):
    """nchain independent chains of `depth` edges each, ALL edges shuffled together."""
    n = nchain * (depth + 1)
    phi = rng.normal(size=(n, d)); phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    psi = rng.normal(size=(n, d)); psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    edges = [(c * (depth + 1) + i, c * (depth + 1) + i + 1)
             for c in range(nchain) for i in range(depth)]
    rng.shuffle(edges)                                   # ARRIVAL ORDER IS SCRAMBLED
    S = np.zeros((d, d))
    for a, b in edges:
        if mode == 'untied':
            k, v = phi[a], psi[b]                        # independent maps: a lookup table
        else:
            k, v = phi[a], phi[b]                        # one map: a transition operator
        S = S - np.outer(S @ k, k) + np.outer(v, k)
    c = int(rng.integers(0, nchain))
    start = c * (depth + 1)
    tab = psi if mode == 'untied' else phi
    got = walk(S, phi[start], depth)
    return got is not None and nearest(got, tab) == start + depth


def streaming(rng, nchain, depth, d=D):
    """CONTROL: accumulate the shuffled edges IN ARRIVAL ORDER, as streaming tracking would.
    There is no correct order to accumulate, so this should fail - and that is the point."""
    n = nchain * (depth + 1)
    phi = rng.normal(size=(n, d)); phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    edges = [(c * (depth + 1) + i, c * (depth + 1) + i + 1)
             for c in range(nchain) for i in range(depth)]
    rng.shuffle(edges)
    h = None
    c = int(rng.integers(0, nchain))
    start = c * (depth + 1)
    h = phi[start].copy()
    for a, b in edges:                                   # apply each edge as a transition, in order
        A = np.eye(d) - np.outer(phi[a], phi[a]) + np.outer(phi[b], phi[a])
        h = A @ h
        nh = np.linalg.norm(h)
        if nh < 1e-10:
            return False
        h = h / nh
    return nearest(h, phi) == start + depth


if __name__ == '__main__':
    print('DEFERRED COMPOSITION: edges arrive SHUFFLED, query walks the graph afterwards')
    print('d=%d, %d trials. chance = 1/(nchain*(depth+1))' % (D, TRIALS))
    print()
    print('  %-8s %-7s %-9s %-12s %-12s %-12s' % ('chains', 'depth', 'chance', 'TIED (ours)', 'untied', 'streaming'))
    print('  ' + '-' * 66)
    for nchain, depth in ((1, 2), (1, 4), (4, 2), (4, 3), (8, 2), (8, 3), (16, 2)):
        ch = 1.0 / (nchain * (depth + 1))
        t = sum(trial(np.random.default_rng(), nchain, depth, 'tied') for _ in range(TRIALS)) / TRIALS
        u = sum(trial(np.random.default_rng(), nchain, depth, 'untied') for _ in range(TRIALS)) / TRIALS
        s = sum(streaming(np.random.default_rng(), nchain, depth) for _ in range(TRIALS)) / TRIALS
        print('  %-8d %-7d %-9.3f %-12.3f %-12.3f %-12.3f' % (nchain, depth, ch, t, u, s), flush=True)
    print()
    print('  READ: if TIED holds while untied AND streaming sit at chance, then deferred composition')
    print('  over a shuffled store is a capability neither a lookup table nor streaming state-tracking')
    print('  has. That is a DIFFERENT axis from Grazzi et al., not a smaller version of it.')
