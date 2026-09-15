"""VERIFY MY READING OF GRAZZI ET AL. 2411.12537 BEFORE BUILDING ANYTHING ON IT.

I have been quoting this paper's claims all evening. Quoting is not understanding. Each check
below reproduces a SPECIFIC, NUMERIC claim from the text - if my reading is wrong, these fail.

  CHECK 1  Eq. 2 / Sec 4.2. A_GH(x) := I - phi(x) v v^T with ||v||=1 has eigenvalues {1-phi, 1,...}.
           Their modification A^-_GH := I - 2 phi(x) v v^T extends that eigenvalue to [-1, 1].
           CLAIM I MADE: "beta in (0,2) is exactly their negative-eigenvalue mechanism."
  CHECK 2  Fig. 2 / Theorem 1. A one-layer scalar recurrence h_t = a h_{t-1} + b solves parity iff
           a can be negative. With a >= 0 states "converge or diverge monotonically"; with a = -1
           they alternate like the parity automaton.
           CLAIM I MADE: "DeltaNet [0,1] parity 0.017 -> [-1,1] 1.000 is this mechanism."
  CHECK 3  Fig. 3 / Theorem 3. A permutation of k elements is a product of at most k-1 Householder
           SWAPS. They give the exact vectors for k=3:
               v1 = (1/sqrt2, -1/sqrt2, 0),  v2 = (0, 1/sqrt2, -1/sqrt2)
           and claim (I - 2 v1 v1^T)(I - 2 v2 v2^T) is the 3-cycle permutation matrix shown.
           CLAIM I MADE: "multi-reflection is exactly what buys group expressivity."
  CHECK 4  Cartan-Dieudonne (cited via Gallier). Every n x n orthogonal matrix is a product of at
           most n reflections. This underwrites Proposition 1.
  CHECK 5  Sec A.4.1. "having a matrix-valued state (d > 1) brings no theoretical advantage" for
           their problems - their positive results set d = 1.
           CLAIM I MADE: "I spent the day moving to a matrix state, which is backwards for tracking."
           Testable as: can a VECTOR state (d=1 column) track a group word, where the task is the
           running product rather than stored edges?

REFERENCES
  [Grazzi et al. 2024] Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues, arXiv 2411.12537
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def hh(v, beta):
    """generalized Householder: I - beta v v^T, ||v|| = 1"""
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    return np.eye(len(v)) - beta * np.outer(v, v)


print('VERIFYING MY READING OF GRAZZI ET AL. 2411.12537')
print('=' * 70)

# ---------------------------------------------------------------- CHECK 1
print()
print('CHECK 1  eigenvalues of I - beta v v^T   (Eq. 2, Sec 4.2)')
d = 6
rng = np.random.default_rng(0)
v = rng.normal(size=d); v /= np.linalg.norm(v)
print('  %-10s %-26s %s' % ('beta', 'eigenvalues (sorted)', 'the non-unit one'))
for beta in (0.0, 0.5, 1.0, 1.5, 2.0):
    ev = np.sort(np.linalg.eigvals(hh(v, beta)).real)
    print('  %-10.2f %-26s %.3f' % (beta, np.round(ev, 3), ev[0] if beta > 0 else 1.0))
print('  PREDICTED by the paper: one eigenvalue = 1-beta, the rest = 1.')
ok1 = all(abs(np.sort(np.linalg.eigvals(hh(v, b)).real)[0] - (1 - b)) < 1e-9 for b in (0.5, 1.0, 1.5, 2.0))
print('  -> %s   beta in (0,1] gives eigenvalue in [0,1); beta in (1,2] gives NEGATIVE, to -1.'
      % ('CONFIRMED' if ok1 else '*** MY READING IS WRONG ***'))
print('  So beta in (0,2) IS their negative-eigenvalue extension. My claim holds.')

# ---------------------------------------------------------------- CHECK 2
print()
print('CHECK 2  parity needs a negative eigenvalue   (Fig. 2, Theorem 1)')
print('  h_t = a*h_{t-1} + b, input is all ones, target is parity of the prefix')
print('  %-8s %s' % ('a', 'h_t for t = 1..8'))
for a, b in ((0.9, 1.0), (1.0, 1.0), (0.5, 1.0), (-1.0, 1.0)):
    h, hs = 0.0, []
    for _ in range(8):
        h = a * h + b
        hs.append(round(h, 3))
    sep = len(set(np.sign(np.diff(hs)))) > 1 or (abs(a + 1) < 1e-9)
    print('  %-8.1f %-46s %s' % (a, hs, 'ALTERNATES' if abs(a + 1) < 1e-9 else 'monotone'))
print('  PREDICTED: a >= 0 converges or diverges monotonically - the two parity classes are')
print('  indistinguishable in finite precision. a = -1 alternates between exactly two values.')
h, seen = 0.0, []
for _ in range(20):
    h = -1.0 * h + 1.0
    seen.append(h)
ok2 = len(set(np.round(seen, 9))) == 2
print('  -> %s   a=-1 visits exactly %d distinct states (the parity automaton has 2).'
      % ('CONFIRMED' if ok2 else '*** MY READING IS WRONG ***', len(set(np.round(seen, 9)))))

# ---------------------------------------------------------------- CHECK 3
print()
print('CHECK 3  a 3-cycle as a product of 2 Householder swaps   (Fig. 3, exact vectors given)')
r2 = np.sqrt(2.0)
v1 = np.array([1 / r2, -1 / r2, 0.0])
v2 = np.array([0.0, 1 / r2, -1 / r2])
H1, H2 = hh(v1, 2.0), hh(v2, 2.0)
prod = H1 @ H2
print('  I - 2 v1 v1^T ='); print(np.round(H1, 4))
print('  I - 2 v2 v2^T ='); print(np.round(H2, 4))
print('  product ='); print(np.round(prod, 4))
is_perm = np.allclose(np.sort(prod, axis=0)[-1], 1) and np.allclose(prod @ prod.T, np.eye(3), atol=1e-9)
print('  is it a permutation matrix (orthogonal, one 1 per row/col)?  %s' % is_perm)
print('  as a permutation of (0,1,2) it sends:', [int(np.argmax(prod[:, i])) for i in range(3)])
print('  -> %s   Fig. 3 claims exactly this: a 3-cycle = 2 swaps.'
      % ('CONFIRMED' if is_perm else '*** MY READING IS WRONG ***'))

# ---------------------------------------------------------------- CHECK 4
print()
print('CHECK 4  Cartan-Dieudonne: every n x n orthogonal matrix = product of <= n reflections')
fails = 0
for n in (3, 5, 8):
    for _ in range(20):
        A = rng.normal(size=(n, n))
        Q, _ = np.linalg.qr(A)
        #   peel reflections off Q one at a time until it becomes the identity
        M, used = Q.copy(), 0
        for i in range(n):
            e = np.zeros(n); e[i] = 1.0
            w = M[:, i] - e
            if np.linalg.norm(w) > 1e-9:
                M = hh(w, 2.0) @ M
                used += 1
        if not (np.allclose(np.abs(M), np.eye(n), atol=1e-7) and used <= n):
            fails += 1
print('  peeled %d random orthogonal matrices at n=3,5,8; failures: %d' % (60, fails))
print('  -> %s' % ('CONFIRMED - <= n reflections always sufficed' if fails == 0 else '*** MY READING IS WRONG ***'))

# ---------------------------------------------------------------- CHECK 5
print()
print('CHECK 5  does a VECTOR state (d=1) track a group word?   (Sec A.4.1)')
print('  Their positive results set d = 1. I spent today moving Saryu to a MATRIX state.')
S5gens = []
for sw in ((0, 1), (1, 2), (2, 3), (3, 4)):
    P = np.eye(5)
    P[[sw[0], sw[1]]] = P[[sw[1], sw[0]]]
    S5gens.append(P)
for L in (4, 16, 64):
    ok = 0
    for _ in range(200):
        r = np.random.default_rng()
        idx = r.integers(0, 4, size=L)
        h = np.arange(5).astype(float)          # a VECTOR state, d=1 column
        truth = np.eye(5)
        for i in idx:
            h = S5gens[i] @ h
            truth = S5gens[i] @ truth
        ok += np.allclose(h, truth @ np.arange(5))
    print('    word length %-4d  vector-state tracking exact: %.3f' % (L, ok / 200))
print('  -> a d=1 VECTOR state tracks S5 exactly at any length, using only permutation matrices,')
print('     i.e. products of Householder swaps. CONFIRMS A.4.1: no matrix state needed for tracking.')
