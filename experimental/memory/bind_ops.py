"""WHY MULTIPLICATION? Comparing binding operations, because I never justified the one in use.

full_stack binds the address as normalize(c * h) - an elementwise product. I did not choose that
after comparing anything; temporal_bind used sign vectors, where the product is the obvious code,
and it carried over into the three-way address unexamined. This tests it.

WHAT BINDING HAS TO DO, which is what rules most operations out:
  dissimilarity   bind(a,b) must be UNLIKE both a and b, or "X in state A" collides with plain X
  selectivity     bind(a,b) ~ bind(a',b') only when a~a' AND b~b'
  dimension       d x d -> d, or a three-way address costs d^3

THE OPERATIONS
  add       a + b            NOT binding - this is BUNDLING. The sum stays SIMILAR to both
                             operands, so it cannot separate. Included as a CONTROL: it must
                             fail, and if it does not, this harness cannot detect a bad binding.
  hadamard  a * b            what full_stack uses. Dimension-preserving, self-inverse for sign
                             vectors. PRESERVES COORDINATE STRUCTURE, which is the suspect below.
  conv      a (*) b          circular convolution, Plate's HRR. O(d log d) by FFT, invertible by
                             correlation. DELOCALISES: every output component mixes all inputs.
  fhrr      phase addition   complex unit-modulus binding. Exactly invertible (subtract phases),
                             commutative, associative. NOTE: the temporal rotation R^t is a
                             SPECIAL CASE of this, so time and content binding are one family.

  (outer product is the ideal binding - perfectly separating and invertible - and is excluded
   because it costs d^2 per bind and d^3 three-way. We already pay it ONCE: the matrix S is
   itself a d x d tensor-product space. This probe is about what to do INSIDE the address.)

THE HYPOTHESIS WORTH TESTING. full_stack found a real capacity tax from the state factor, and
traced it to orbit correlation: two permutations differing by one transposition share n-2 of n
slots, giving |cos| ~ 0.6 by construction. Three linear embeddings failed to fix it, and I
concluded "there isn't an embedding that removes the correlation". That conclusion is correct but
NARROW - it is about embeddings under HADAMARD. Hadamard keeps coordinates separate, so shared
slots stay shared. Convolution and phase binding smear each component across all positions, so
the shared structure should be diluted rather than preserved.

PRE-REGISTERED
  B1  `add` fails episodic separation - the bundle stays similar to bare content. Control.
  B2  hadamard reproduces full_stack: separation ~0.58-0.64, capacity tax ~0.12 at N=96.
  B3  conv and fhrr separate AT LEAST as well, and carry a SMALLER capacity tax, because they
      delocalise the shared slots that hadamard preserves. This is the claim; if it fails, the
      state tax is intrinsic to the group and no binding operation rescues it.
  B4  composition survives all of them - the address is applied identically to key and value, so
      value-is-a-legal-key holds by construction regardless of the operation.

RESULTS - FHRR WINS ON EVERY METRIC, AND HADAMARD WAS COSTING US A LOT.

  B1  EPISODIC + the dissimilarity control
        op         earlier   recent   |cos(bound, c)|
        add        0.648     0.968    0.710     BUNDLES, not a binding
        hadamard   0.644     0.990    0.108
        conv       0.709     0.994    0.105
        fhrr       0.743     0.995    0.068

      HELD, and the control earned its place: the SEPARATION column alone would NOT have caught
      `add` (0.648, essentially hadamard's 0.644). A bundle is still a distinct vector, so the
      delta rule stores and retrieves it fine when queried with the same bundle. What makes it
      useless is that it stays SIMILAR to bare content (0.710), so it collides with everything
      sharing a component. Only the dissimilarity check discriminates.

  B3  CAPACITY at matched fact count - THE RESULT THAT MATTERS
        content-only ref   0.983  0.971  0.935  0.840  0.769
        fhrr               0.982  0.968  0.937  0.867  0.766
        conv               0.980  0.952  0.892  0.786  0.668
        hadamard           0.973  0.941  0.877  0.719  0.614
        add                0.922  0.890  0.830  0.810  0.765

      FHRR MATCHES UNBOUND CONTENT AT EVERY N. The state-binding tax is ZERO with it, against
      -0.155 for hadamard at N=96. conv sits between, as predicted, confirming the mechanism is
      delocalisation: hadamard keeps coordinates separate so shared slots stay shared; conv
      smears partially; fhrr works in phase, where a shared slot has no coordinate to share.

      `add`'s capacity (0.765) matching content-only (0.769) is not a win, it is the tell: it
      scores well BECAUSE it barely binds, so the state factor is nearly ignored and the address
      is essentially bare content.

  B4  COMPOSITION    add 0.785/0.709/0.478   hadamard 0.955/0.923/0.843
                     conv 0.964/0.939/0.882  fhrr 0.976/0.956/0.917   (depths 2/3/5)
      HELD for every real binding, and fhrr leads at every depth. `add` degrades badly by depth 5.

WHAT THIS OVERTURNS. full_stack concluded the state-binding capacity tax was intrinsic, on the
evidence that three linear EMBEDDINGS could not shift the orbit correlation. The correlation part
stands - it is group-theoretic, two permutations differing by a transposition share n-2 of n slots
and no linear embedding changes that. But the CONSEQUENCE was wrong: fhrr leaves the correlation
exactly as it is and removes the capacity cost anyway. Correlated states are only expensive if the
binding operation lets the correlation reach the address.

THE UNIFICATION, which is the part worth keeping: the temporal code R^t is ALREADY FHRR binding -
a rotation is phase addition. Switching content*state from hadamard to fhrr makes the entire
address ONE operation applied three times. content, state and time stop being three mechanisms
and become three factors in a single phase sum.

NOT A CHANGE TO THE PENDING GPU RUN: ParityBlock has no binding at all - no state factor, no time
factor, the address is the bare percept. This is design input for the build AFTER it.

usage: cd experiments && python bind_ops.py
"""
from __future__ import annotations

import numpy as np

from full_stack import GENS, NSLOT, track, write, cos
from temporal_bind import unit

D = 128
TRIALS = 120


# ------------------------------------------------------------------ the operations
def bind(a, b, op):
    if op == 'add':
        return unit(a + b)
    if op == 'hadamard':
        return unit(a * b)
    if op == 'conv':
        return unit(np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))))
    if op == 'fhrr':
        h = len(a) // 2
        za, zb = a[:h] + 1j * a[h:], b[:h] + 1j * b[h:]
        za = za / np.maximum(np.abs(za), 1e-12)
        zb = zb / np.maximum(np.abs(zb), 1e-12)
        z = za * zb
        return unit(np.concatenate([np.real(z), np.imag(z)]))
    raise ValueError(op)


OPS = ('add', 'hadamard', 'conv', 'fhrr')


def addr(c, s, E, op):
    return bind(c, unit(E @ s), op)


# ------------------------------------------------------------------ B1/B2  episodic
def episodic(op, d=D, trials=TRIALS, nfill=10):
    early, recent, ctrl = [], [], []
    for _ in range(trials):
        rng = np.random.default_rng()
        E = rng.normal(size=(d, NSLOT))
        base = rng.normal(size=NSLOT)
        sA = track(base, rng.integers(0, len(GENS), size=3))[-1]
        sB = track(base, rng.integers(0, len(GENS), size=7))[-1]
        X, P, Q = unit(rng.normal(size=(3, d)))
        S = np.zeros((d, d))
        t1, t2 = 1, 2 + nfill
        for t in range(t2 + 2):
            if t == t1:
                k, v = addr(X, sA, E, op), addr(P, sA, E, op)
            elif t == t2:
                k, v = addr(X, sB, E, op), addr(Q, sB, E, op)
            else:
                f = unit(rng.normal(size=(2, d)))
                sf = track(base, rng.integers(0, len(GENS), size=2))[-1]
                k, v = addr(f[0], sf, E, op), addr(f[1], sf, E, op)
            S = write(S, k, v)
        early.append(cos(S @ addr(X, sA, E, op), addr(P, sA, E, op)))
        recent.append(cos(S @ addr(X, sB, E, op), addr(Q, sB, E, op)))
        #   DISSIMILARITY: a bound address must be UNLIKE bare content, or it is bundling
        ctrl.append(abs(cos(addr(X, sA, E, op), X)))
    return float(np.mean(early)), float(np.mean(recent)), float(np.mean(ctrl))


# ------------------------------------------------------------------ B3  capacity, matched N
def capacity(op, nfacts, d=D, trials=60):
    out = []
    for _ in range(trials):
        rng = np.random.default_rng()
        E = rng.normal(size=(d, NSLOT))
        base = rng.normal(size=NSLOT)
        orbit = [track(base, rng.integers(0, len(GENS), size=k + 1))[-1] for k in range(8)]
        S = np.zeros((d, d))
        facts = []
        for i in range(nfacts):
            c, v = unit(rng.normal(size=(2, d)))
            s = orbit[i % len(orbit)]
            k, vv = addr(c, s, E, op), addr(v, s, E, op)
            facts.append((k, vv))
            S = write(S, k, vv)
        j = rng.integers(nfacts)
        out.append(cos(S @ facts[j][0], facts[j][1]))
    return float(np.mean(out))


# ------------------------------------------------------------------ B4  composition
def compose(op, depth, d=D, trials=TRIALS, nfill=6):
    out = []
    for _ in range(trials):
        rng = np.random.default_rng()
        E = rng.normal(size=(d, NSLOT))
        base = rng.normal(size=NSLOT)
        gens = rng.integers(0, len(GENS), size=depth + nfill + 2)
        states = [base] + track(base, gens)
        cs = unit(rng.normal(size=(depth + nfill + 3, d)))
        S = np.zeros((d, d))
        order = list(range(depth + nfill))
        rng.shuffle(order)
        for t in order:
            S = write(S, addr(cs[t], states[t], E, op), addr(cs[t + 1], states[t + 1], E, op))
        x = addr(cs[0], states[0], E, op)
        ok = True
        for _ in range(depth):
            x = S @ x
            n = np.linalg.norm(x)
            if n < 1e-10:
                ok = False
                break
            x = x / n
        out.append(cos(x, addr(cs[depth], states[depth], E, op)) if ok else 0.0)
    return float(np.mean(out))


if __name__ == '__main__':
    print('BINDING OPERATIONS  d=%d  content*state only (time excluded to isolate the op)\n' % D)

    print('B1/B2  EPISODIC: one content, two tracked states. |cos vs bare content| must be LOW')
    print('  %-10s %-10s %-12s %-16s %s' % ('op', 'earlier', 'most recent', '|cos(bound, c)|', ''))
    for op in OPS:
        e, r, c = episodic(op)
        note = 'BUNDLES - not a binding' if c > 0.4 else ('separates' if e > 0.5 else 'weak')
        print('  %-10s %-10.3f %-12.3f %-16.3f %s' % (op, e, r, c, note), flush=True)

    print('\nB3  CAPACITY at matched fact count (content*state, same N writes, same matrix)')
    NS = (8, 16, 32, 64, 96)
    print('  %-10s %s' % ('op', '  '.join('N=%-5d' % n for n in NS)))
    for op in OPS:
        row = [capacity(op, n) for n in NS]
        print('  %-10s %s' % (op, '  '.join('%-7.3f' % r for r in row)), flush=True)
    print('  full_stack reference: content-only 0.983 0.971 0.935 0.840 0.769')
    print('                        hadamard+state             0.873 0.739 0.622')

    print('\nB4  COMPOSITION through the bound address')
    print('  %-10s %-10s %-10s %s' % ('op', 'depth 2', 'depth 3', 'depth 5'))
    for op in OPS:
        print('  %-10s %-10.3f %-10.3f %.3f'
              % (op, compose(op, 2), compose(op, 3), compose(op, 5)), flush=True)

    print('\n  READ: if conv/fhrr beat hadamard on B3 while holding B1 and B4, the state tax is an')
    print('  artefact of an operation that preserves coordinates, not a fact about the group, and')
    print('  full_stack\'s "no embedding fixes it" was answering the wrong question.')
