"""WHY "REFLECTIONS ARE NOT ADDITIVE" WAS WRONG, demonstrated rather than asserted.

WHAT I ORIGINALLY MEASURED (reflect_fix.py): does adding reflections help WALK A STORED CHAIN?
Result: nref=2/3/4 at key spread 0.20 gave 2-hop 0.957 / 0.943 / 0.937 against a 1.000 baseline -
slightly worse. I concluded "compatible but not additive".

WHY THAT WAS THE WRONG TEST. In my construction the extra reflections did REDUNDANT WORK - they
re-wrote the same edge with jittered keys. Of course that does not help: nothing new is being
expressed. But reflections in Grazzi Theorem 3 / Fig 3 are not for writing more edges. They are
for building a RICHER SINGLE TRANSITION. A permutation of k elements is a product of at most k-1
Householder swaps, and crucially ONE reflection CANNOT express an even permutation at all.

So the right question is not "do extra reflections improve chain-walking accuracy" but "what can
k reflections represent that k-1 cannot". That is expressivity, and it is measurable exactly.

THE DETERMINANT ARGUMENT, which settles it in one line:
    a Householder I - 2vv^T is a reflection, det = -1
    a product of k reflections has det = (-1)^k
    an even permutation has det = +1
therefore ONE reflection can never be an even permutation, and a 3-cycle (even) needs at least 2.
Expressivity is strictly additive in the number of reflections. My earlier probe could not see
this because chain-walking never asked for an even permutation.

REFERENCES
  [Grazzi et al. 2024] Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues, arXiv 2411.12537
  [Siems et al. 2025] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products, arXiv 2502.10297
"""
import numpy as np
from itertools import permutations

np.set_printoptions(precision=3, suppress=True)


def house(v):
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    return np.eye(len(v)) - 2.0 * np.outer(v, v)


def perm_matrix(p):
    n = len(p)
    P = np.zeros((n, n))
    for i, j in enumerate(p):
        P[j, i] = 1.0
    return P


def min_reflections(P, tol=1e-8, cap=8):
    """smallest k such that P is a product of k Householder reflections (greedy peel, exact here)"""
    n = P.shape[0]
    M, used = P.copy(), 0
    for i in range(n):
        e = np.zeros(n); e[i] = 1.0
        w = M[:, i] - e
        if np.linalg.norm(w) > tol:
            M = house(w) @ M
            used += 1
    return used if np.allclose(M, np.eye(n), atol=1e-7) else None


print('CAN ONE REFLECTION BE AN EVEN PERMUTATION?   det(reflection) = -1, det(even perm) = +1')
print()
print('  %-22s %-8s %-10s %-14s %s' % ('permutation of (0,1,2)', 'parity', 'det', 'min reflections', 'one enough?'))
print('  ' + '-' * 74)
for p in permutations(range(3)):
    P = perm_matrix(p)
    det = float(np.linalg.det(P))
    inv = sum(1 for i in range(3) for j in range(i + 1, 3) if p[i] > p[j])
    k = min_reflections(P)
    print('  %-22s %-8s %-10.1f %-14s %s'
          % (str(p), 'even' if inv % 2 == 0 else 'odd', det, k if k is not None else '?',
             'YES' if k == 1 else 'NO'))
print()
print('  Every EVEN permutation needs >= 2 reflections. One reflection cannot reach them at all.')
print('  That is additive expressivity, and chain-walking never asked for it.')

print()
print('HOW MANY REFLECTIONS DOES A k-CYCLE NEED?   Theorem 3 predicts at most k-1')
print()
print('  %-10s %-12s %-16s %s' % ('cycle', 'parity', 'min reflections', 'k-1 bound holds?'))
print('  ' + '-' * 58)
for k in range(2, 8):
    p = tuple(list(range(1, k)) + [0]) + tuple(range(k, 8))
    P = perm_matrix(p)
    inv = sum(1 for i in range(8) for j in range(i + 1, 8) if p[i] > p[j])
    r = min_reflections(P)
    print('  %-10s %-12s %-16s %s' % ('%d-cycle' % k, 'even' if inv % 2 == 0 else 'odd', r,
                                      'yes' if r is not None and r <= k - 1 else 'NO'))

print()
print('WHAT A SINGLE GENERALISED HOUSEHOLDER CAN DO  (I - beta v v^T, the DeltaNet update)')
d = 5
rng = np.random.default_rng(0)
v = rng.normal(size=d); v /= np.linalg.norm(v)
for beta in (1.0, 2.0):
    A = np.eye(d) - beta * np.outer(v, v)
    ev = np.sort(np.linalg.eigvals(A).real)
    print('  beta=%.1f  eigenvalues %s  det %+.3f  orthogonal? %s'
          % (beta, np.round(ev, 3), float(np.linalg.det(A)),
             np.allclose(A @ A.T, np.eye(d), atol=1e-9)))
print('  Only beta=2 is ORTHOGONAL (a true reflection). beta<2 shrinks one direction and is not')
print('  norm-preserving - it cannot be any permutation at all. DeltaNet default beta in (0,1)')
print('  therefore cannot represent even a single swap exactly.')

print()
print('THE CORRECTION:')
print('  reflections ARE additive - in EXPRESSIVITY, which is what they are for.')
print('  My earlier probe measured chain-walking accuracy, where extra reflections wrote redundant')
print('  edges and so could only add noise. It never posed a task requiring an even permutation,')
print('  so the one thing reflections buy was invisible to it. Conclusion withdrawn, on evidence.')
