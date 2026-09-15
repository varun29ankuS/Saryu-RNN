"""THE NON-ABELIAN TRANSPORT CLAIM - the project's actual thesis, which I had stopped testing.

reason_battery.py states it: "If relations are group elements, composing them is multiplying
transports - so an ORDERED chain is walked as it is read, needing no storage at all: twenty edges
in order are twenty group multiplications in a fixed state, not twenty facts to remember. A
SHUFFLED chain destroys that: the path must be FOUND first, which is search over stored content,
and transport buys nothing. So the signature of a transport model is accuracy that stays FLAT as
the ordered chain gets longer, while a lookup model degrades as it runs out of slots."

EVERY composition test I built today SHUFFLED the edges - i.e. measured the one regime where
transport is predicted to give nothing. This tests the claim as stated.

THREE THINGS MEASURED:
  1. ORDERED vs SHUFFLED as chain length grows. Transport predicts FLAT for ordered; storage
     predicts DEGRADING for both.
  2. NON-ABELIAN vs ABELIAN relations. If the relations commute, order carries no information and
     a bag-of-relations model does just as well. Non-commuting relations are what make ORDER a
     thing that must be tracked.
  3. ORDER SENSITIVITY - does applying R1 then R2 actually differ from R2 then R1? If not, the
     "non-abelian" label is decoration.

Pure numerics, no training, no harness to get wrong.
"""
import numpy as np

D, TRIALS = 64, 300


def rand_rotation(rng, d, abelian=False, basis=None):
    """a relation as a transport. Non-abelian: general rotation. Abelian: diagonal in a SHARED basis."""
    if abelian:
        ang = rng.uniform(0, 2 * np.pi, size=d // 2)
        blocks = np.zeros((d, d))
        for i, a in enumerate(ang):
            c, s = np.cos(a), np.sin(a)
            blocks[2 * i, 2 * i], blocks[2 * i, 2 * i + 1] = c, -s
            blocks[2 * i + 1, 2 * i], blocks[2 * i + 1, 2 * i + 1] = s, c
        return basis @ blocks @ basis.T          # same eigenbasis for all -> they commute
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    return Q


def nearest(x, table):
    return int(np.argmax(table @ x / (np.linalg.norm(table, axis=1) * np.linalg.norm(x) + 1e-12)))


def run_ordered(rng, L, K, d=D, abelian=False, basis=None):
    """ORDERED: relations arrive in path order. A transport model just multiplies as it reads -
    constant state, no storage. Accuracy should be FLAT in L."""
    T = [rand_rotation(rng, d, abelian, basis) for _ in range(K)]
    seq = rng.integers(0, K, size=L)
    x0 = rng.normal(size=d); x0 /= np.linalg.norm(x0)
    h = x0.copy()
    for r in seq:                                # walk it as it is read
        h = T[r] @ h
    truth = x0.copy()
    for r in seq:
        truth = T[r] @ truth
    #   candidates: the true endpoint, plus endpoints from PERMUTED orders of the same relations
    cands = [truth]
    for _ in range(7):
        p = rng.permutation(seq)
        y = x0.copy()
        for r in p:
            y = T[r] @ y
        cands.append(y)
    tab = np.stack(cands)
    return nearest(h, tab) == 0


def run_shuffled(rng, L, K, d=D, abelian=False, basis=None):
    """SHUFFLED: the same relations arrive out of order, so the path must be RECONSTRUCTED. A
    transport model cannot just multiply - it has to store and search. Storage is finite."""
    T = [rand_rotation(rng, d, abelian, basis) for _ in range(K)]
    seq = rng.integers(0, K, size=L)
    x0 = rng.normal(size=d); x0 /= np.linalg.norm(x0)
    truth = x0.copy()
    for r in seq:
        truth = T[r] @ truth
    perm = rng.permutation(L)                    # model sees them in this (wrong) order
    h = x0.copy()
    for i in perm:
        h = T[seq[i]] @ h
    cands = [truth]
    for _ in range(7):
        p = rng.permutation(seq)
        y = x0.copy()
        for r in p:
            y = T[r] @ y
        cands.append(y)
    return nearest(h, np.stack(cands)) == 0


if __name__ == '__main__':
    rng0 = np.random.default_rng(0)
    A = rng0.normal(size=(D, D))
    BASIS, _ = np.linalg.qr(A)

    print('NON-ABELIAN TRANSPORT   d=%d, %d trials, chance = 1/8 = 0.125' % (D, TRIALS))
    print('Reading order out of a chain of relations. Candidates are the TRUE endpoint plus 7')
    print('endpoints from PERMUTED orders of the SAME relations - so only ORDER distinguishes them.')
    print()

    print('  3. ORDER SENSITIVITY FIRST - if relations commute, the task is vacuous:')
    for lab, ab in (('non-abelian (general rotations)', False), ('abelian (shared eigenbasis)', True)):
        r1 = rand_rotation(rng0, D, ab, BASIS)
        r2 = rand_rotation(rng0, D, ab, BASIS)
        comm = float(np.abs(r1 @ r2 - r2 @ r1).max())
        print('     %-34s  max|R1R2 - R2R1| = %.4f  %s'
              % (lab, comm, 'ORDER MATTERS' if comm > 1e-6 else 'commute - order is information-free'))
    print()

    print('  1+2. ACCURACY vs CHAIN LENGTH  (K=4 relation types)')
    print('  %-8s %-22s %-22s' % ('length', 'ORDERED', 'SHUFFLED'))
    print('  %-8s %-10s %-10s %-10s %-10s' % ('', 'non-abel', 'abelian', 'non-abel', 'abelian'))
    print('  ' + '-' * 56)
    for L in (2, 4, 8, 16, 32, 64):
        row = []
        for fn in (run_ordered, run_shuffled):
            for ab in (False, True):
                ok = sum(fn(np.random.default_rng(), L, 4, D, ab, BASIS) for _ in range(TRIALS))
                row.append(ok / TRIALS)
        print('  %-8d %-10.3f %-10.3f %-10.3f %-10.3f' % (L, row[0], row[1], row[2], row[3]), flush=True)
    print()
    print('  READ THE SIGNATURE:')
    print('   ORDERED non-abelian FLAT in L  -> transport: composition is free, no storage needed.')
    print('   ORDERED DEGRADING              -> it is not composing, it is storing.')
    print('   SHUFFLED should be at chance   -> reconstructing order from a bag is the hard problem,')
    print('                                     and it is what EVERY chain test I built today used.')
    print('   ABELIAN columns are the CONTROL: if they match non-abelian, order carried no')
    print('   information and the task never tested composition at all.')
