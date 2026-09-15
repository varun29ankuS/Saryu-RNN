"""THE WHOLE STACK AT ONCE: matrix + reflection-tracked state + temporal, in every combination.

Each piece has been measured ALONE and each works alone. Nothing here has been measured TOGETHER,
and they are not independent - all three are multiplicative changes to the SAME key, spending the
SAME finite address space in the same rank-d matrix. That is the question this settles.

THE DESIGN

  state tracking   s_t in R^n evolves by Householder reflections. u = (e_i - e_j)/sqrt(2) makes
                   I - 2uu^T EXACTLY the transposition (i j), so the generators generate S_n and
                   s_t is always a genuine permutation of the base vector. This is the non-abelian
                   tracker - reflections are not a separate module, they ARE beta=2 on a rank-1
                   erase, which is why det = -1 and why one reflection can never be an even
                   permutation.
  the address      A(c, s, t) = rotate( normalize( c * embed(s) ), t )
                   content, times the embedded tracked state, rotated by the clock. Flags drop
                   the state factor and/or the time factor, giving the 2x2 of bindings.
  the memory       one matrix S, delta rule. beta=1 (least squares) or beta=2 (reflection) as an
                   AXIS - see the prediction below, which I expect to come out against beta=2.

WHY value-is-a-legal-key IS NOT FREE HERE, and is asserted rather than argued: in temporal_bind
the codes were sign vectors and rotations, both norm-preserving, so k_t == v_(t-1) held exactly.
normalize(c * h) is a different animal - ||c*h|| ~ 1/sqrt(d) before normalising and the direction
depends jointly on both factors. So CHECK 0 asserts the stream is built such that the value
written at step t IS bitwise the key read at step t+1. If that fails nothing downstream means
anything, exactly as the tying bug taught.

PRE-REGISTERED

  F1  CHECK 0 holds for all four bindings - the address function is consistent by construction.
  F2  reflections track S_n EXACTLY (the tracker is not approximate), and the orbit's pairwise
      cosine is HIGH - a transposition leaves n-2 of n slots fixed, so distinct states are
      strongly correlated. I do NOT gate on that: the delta rule is least squares and decorrelating
      correlated keys is precisely what separates it from a Hebbian outer-product memory. Whether
      that is enough is the measurement, not an assumption.
  F3  episodic separation NEEDS the state factor. Content-only must fail to tell "X in state A"
      from "X in state B" - it is the same address - and content*state must recover both.
  F4  capacity: AT MATCHED SEPARABLE-FACT COUNT, binding is roughly free. Comparing content-only
      holding N facts against content*state holding N facts puts the same number of rank-1 writes
      in the same matrix; the earlier P4 compared different fact counts and could not have found
      a cost. If bound recall is materially below unbound at equal N, THAT is the binding tax.
  F5  beta=2 in the MEMORY breaks composition, while beta=2 in the TRACKER is exactly right. On a
      colliding key the erase is I - 2kk^T, which flips the sign along k: a walked edge returns
      -v rather than v. Reflections belong in the state recurrence, not in the associative store.
      If beta=2 composes anyway, that is a real finding and I want it on record that I expected
      the opposite.

RESULTS

  GATES     CHECK 0 = 0.00e+00 for all four bindings; the S_5 tracker is EXACT at 4.88e-15.
            Reflections really do track the permutation group, not an approximation of it.

  F1  HELD. The address function is consistent - value at t IS key at t+1, bitwise.
  F2  HELD on exactness, and the orbit correlation is worse than "high". Over all 7140 state
      pairs, |cos| mean 0.361, and:
            > 0.50  25.95% of pairs
            > 0.80  19.27%
            > 0.90  13.45%
            > 0.99   2.52%          ~180 pairs are effectively THE SAME state after embedding
      I first wrote here that the embedding was "the obvious place to fix this - an orthogonalised
      or learned embedding would spread the orbit". THAT CLAIM WAS COMMITTED AND IS FALSE. Tested
      across three embeddings of the same tracked states:

            embedding      orbit|cos|   pairs>0.9   cap N=32   cap N=96
            random         0.361        13.5%       0.884      0.652
            orthonormal    0.354        16.0%       0.906      0.636
            block          0.354        16.0%       0.930      0.693
            (content-only reference, no state factor)  0.935      0.769

      Orthogonalising moves the orbit correlation by 0.007 and makes the >0.9 tail WORSE. The
      reason is not a coding accident: two permutations differing by one transposition share n-2
      of n slots, so their inner product is ~(n-2)/n = 0.6 BY CONSTRUCTION, and any linear
      embedding preserves inner products up to conditioning. The correlation is a fact about the
      group, not about E.

      What does change is CAPACITY, and the tax is LOAD-DEPENDENT. A block embedding is nearly
      free at N=32 (0.930 against content-only's 0.935) and recovers about half the random
      embedding's loss at N=96 (0.652 -> 0.693 against 0.769). So: use a block embedding, expect
      no meaningful cost until the store approaches capacity, and stop looking for an embedding
      that removes the correlation - there isn't one.

      SUPERSEDED IN PART - see bind_ops.py. The caveat below turned out to be the whole story.

      What stands: the orbit correlation is group-theoretic and no linear EMBEDDING shifts it.
      What does NOT stand: the implied conclusion that the capacity tax is therefore intrinsic.
      The tax came from HADAMARD, not from the group. Binding by phase (FHRR) instead:

            binding op    N=8    N=16   N=32   N=64   N=96
            content-only  0.983  0.971  0.935  0.840  0.769   (no state factor at all)
            fhrr          0.982  0.968  0.937  0.867  0.766   tax is ZERO
            hadamard      0.973  0.941  0.877  0.719  0.614   tax is -0.155

      FHRR leaves the orbit correlation untouched and removes the cost regardless, because
      hadamard preserves coordinate structure - shared slots stay shared - while phase binding
      has no coordinate for a shared slot to occupy. FHRR also wins separation (0.743 vs 0.644),
      dissimilarity from bare content (0.068 vs 0.108) and composition at depth 5 (0.917 vs
      0.843). Every number in F3/F4/F5 above was measured with hadamard and is therefore a
      LOWER BOUND on what the stack does.

      CAVEAT: only linear/additive slot codes were tested as EMBEDDINGS. The fix turned out to be
      the binding operation instead.
  F3  HELD, but the state factor is the WEAKEST separator, and F2 says why:
            content only        -0.003   collides, same address
            content*state        0.641
            content*time         0.905
            content*state*time   0.916
  F4  WRONG FOR STATE, RIGHT FOR TIME. I registered "binding is roughly free at matched fact
      count". At N=96, against content-only's 0.769:  +time 0.741 (-0.028, near free),
      +state 0.622 (-0.147, a REAL tax), +both 0.687 (-0.082).
      So there is a binding tax and it is specifically the state factor, for the reason F2 gives:
      correlated keys interfere, and tracked permutation states are correlated.

      THE RESULT I DID NOT EXPECT, and the direct answer to "why combine them":
      content*state*time (0.687) BEATS content*state (0.622). The clock DECORRELATES keys that
      the state factor had bunched together, so adding time repairs interference that state
      binding causes. Combining is not merely compatible - it is better than the state factor
      alone, and neither piece measured separately could have shown that.

      *** THAT HEADLINE DOES NOT SURVIVE FHRR. *** The mechanism above was right and the
      conclusion drawn from it was not. Re-measured with phase binding instead of hadamard:

            binding              epi early   N=32    N=64    N=96
            content only         -0.011      0.914   0.871   0.788
            content*state         0.736      0.930   0.862   0.757
            content*time          0.910      0.935   0.843   0.766
            content*state*time    0.940      0.914   0.843   0.734

      Under hadamard, adding time to state was worth +0.065 at N=96 (0.622 -> 0.687). Under FHRR
      it is worth -0.023 (0.757 -> 0.734), i.e. NOTHING. The clock was repairing damage hadamard
      itself caused; remove the damage and there is nothing left to repair.

      NOISE FLOOR, because -0.023 is not a cost: content-only involves no binding and so must be
      identically distributed across both runs, yet it reads 0.769 there and 0.788 here. That
      ~0.02 gap is the sampling floor at 60 trials, so anything under ~0.03 here is noise. The
      honest statement is "time no longer HELPS capacity", not "time costs capacity".

      WHAT TIME ACTUALLY BUYS, and it is large and outside the noise: episodic separation,
      0.736 -> 0.940. So the clock earns its place for recovering WHEN a fact held, not for
      capacity. Under hadamard I would have kept it for the right reason by accident.

  F5  HELD - but ONLY once tested on the case the prediction actually describes, and the first
      test could not see it. compose() writes distinct random contents, so a colliding key never
      arises and beta=2 looked merely degraded there (0.915-0.941 vs 0.958-0.973). Re-run on a
      REVISITED key, which is the episodic setup:

            binding              b=1 early   b=2 early   b=1 recent   b=2 recent
            content only         -0.007      -0.636      0.996        0.739
            content*state         0.582       0.403      0.993        0.868
            content*time          0.904       0.855      0.996        0.936
            content*state*time    0.914       0.873      0.995        0.935

      -0.636 is the prediction confirmed exactly: the erase I - 2kk^T REFLECTS along k, so a
      superseded edge comes back INVERTED rather than merely gone (beta=1 gives -0.007, neutral).
      beta=2 also damages the surviving recent fact, 0.996 -> 0.739, so it hurts on both sides.

      AND BINDING RESCUES IT: at beta=2 the fully bound address gives 0.873 / 0.935, because the
      two writes land at DIFFERENT addresses and there is no collision left to reflect. The harm
      is collision-specific, and binding is what removes collisions.

CONCLUSION, and it is a design answer rather than a score: reflections belong in the TRACKER,
never in the associative store. The tracker needs them - one reflection can never be an even
permutation, which is the whole non-abelian argument. The store must not have them, because there
beta=2 turns "this fact was superseded" into "this fact is now false with confidence". Those are
different objects and the experiment separates them cleanly. Binding, meanwhile, is what makes the
store safe at beta=2 at all - which is an argument for the full stack, not against it.

STILL NOT TESTED: any of this inside a trained model, and whether a learned state embedding fixes
the orbit correlation that F4's tax comes from.

usage: cd experiments && python full_stack.py
"""
from __future__ import annotations

import itertools

import numpy as np

from temporal_bind import apply_rot, make_rot, unit

D = 128          # address dimension
NSLOT = 5        # S_5 on five slots
NFREQ = 8        # clock resolution; temporal_bind found >=7 needed
TRIALS = 120


# ------------------------------------------------------------------ the reflection tracker
def transposition_u(i, j, n=NSLOT):
    """u such that I - 2uu^T is EXACTLY the transposition (i j)."""
    u = np.zeros(n)
    u[i], u[j] = 1.0, -1.0
    return u / np.linalg.norm(u)


def reflect(x, u, beta=2.0):
    return x - beta * np.outer(u, u) @ x if x.ndim == 1 else x - beta * u[:, None] * (u @ x)


GENS = [(0, 1), (1, 2), (2, 3), (3, 4)]      # adjacent transpositions generate S_5


def track(base, gen_seq):
    """Apply a sequence of reflections. Returns the state after each step."""
    s, out = base.copy(), []
    for g in gen_seq:
        s = reflect(s, transposition_u(*GENS[g]))
        out.append(s.copy())
    return out


def true_perm(perm, gen_seq):
    p = list(perm)
    for g in gen_seq:
        i, j = GENS[g]
        p[i], p[j] = p[j], p[i]
    return p


# ------------------------------------------------------------------ the address
class Addr:
    def __init__(self, use_state, use_time, rng, d=D):
        self.use_state, self.use_time, self.d = use_state, use_time, d
        self.E = rng.normal(size=(d, NSLOT))          # state embedding, fixed
        self.ang = make_rot(d, NFREQ, rng)

    def __call__(self, c, s, t):
        x = c
        if self.use_state:
            h = unit(self.E @ s)
            x = unit(x * h)
        if self.use_time:
            x = apply_rot(x, t, self.ang)
        return unit(x)


def write(S, k, v, beta=1.0):
    return S - beta * np.outer(S @ k - v, k)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-9 or nb < 1e-9 else float(a @ b / (na * nb))


BINDINGS = [('content only', False, False), ('content*state', True, False),
            ('content*time', False, True), ('content*state*time', True, True)]


# ------------------------------------------------------------------ CHECK 0
def check0():
    print('CHECK 0  the value written at t IS the key read at t+1 (else nothing below counts)')
    ok = True
    for name, us, ut in BINDINGS:
        rng = np.random.default_rng(0)
        A = Addr(us, ut, rng)
        base = rng.normal(size=NSLOT)
        gens = rng.integers(0, len(GENS), size=10)
        states = [base] + track(base, gens)
        cs = unit(rng.normal(size=(12, D)))
        #   Build the two streams the way compose() builds them and compare ACROSS them: the value
        #   stored at step t against the key used at step t+1. The first version of this compared
        #   A(x) with A(x) - the same call twice - and would have passed at 0.0 while testing
        #   nothing, which is the LayerNorm-perturbation bug all over again.
        ks = [A(cs[t], states[t], t) for t in range(10)]
        vs = [A(cs[t + 1], states[t + 1], t + 1) for t in range(10)]
        err = max(float(np.abs(vs[t] - ks[t + 1]).max()) for t in range(9))
        good = err < 1e-12
        ok &= good
        print('  %-22s max |v_t - k_(t+1)| = %.2e   %s' % (name, err, 'OK' if good else '*** FAIL ***'))
    return ok


# ------------------------------------------------------------------ CHECK 1/2  the tracker
def check_tracker():
    print('\nCHECK 1  reflections track S_5 exactly; CHECK 2  how separated is the orbit?')
    rng = np.random.default_rng(1)
    base = rng.normal(size=NSLOT)
    worst = 0.0
    for _ in range(200):
        gens = rng.integers(0, len(GENS), size=12)
        s = track(base, gens)[-1]
        want = np.array(true_perm(base, gens))
        worst = max(worst, float(np.abs(s - want).max()))
    print('  tracker vs true permutation: max err %.2e   %s'
          % (worst, 'EXACT' if worst < 1e-12 else '*** DRIFTS ***'))

    orbit = [np.array(p) for p in itertools.permutations(base)]
    E = rng.normal(size=(D, NSLOT))
    emb = [unit(E @ s) for s in orbit]
    cc = [abs(cos(emb[i], emb[j])) for i in range(len(emb)) for j in range(i + 1, len(emb))]
    print('  orbit |cos| over all %d state pairs: mean %.3f  max %.3f  min %.3f'
          % (len(cc), np.mean(cc), np.max(cc), np.min(cc)))
    print('  HIGH by construction - a transposition fixes 3 of 5 slots. The delta rule is least')
    print('  squares, so decorrelating correlated keys is its job; F3/F4 measure whether it can.')
    return worst < 1e-12


# ------------------------------------------------------------------ F3  episodic separation
def episodic(us, ut, beta=1.0, d=D, trials=TRIALS, nfill=10):
    """One content X seen in TWO different tracked states, with different successors."""
    both, recent = [], []
    for _ in range(trials):
        rng = np.random.default_rng()
        A = Addr(us, ut, rng)
        base = rng.normal(size=NSLOT)
        sA = track(base, rng.integers(0, len(GENS), size=3))[-1]
        sB = track(base, rng.integers(0, len(GENS), size=7))[-1]
        X, P, Q = unit(rng.normal(size=(3, d)))
        S = np.zeros((d, d))
        t1, t2 = 1, 2 + nfill
        for t in range(t2 + 2):
            if t == t1:
                k, v = A(X, sA, t), A(P, sA, t + 1)
            elif t == t2:
                k, v = A(X, sB, t), A(Q, sB, t + 1)
            else:
                f = unit(rng.normal(size=(2, d)))
                sf = track(base, rng.integers(0, len(GENS), size=2))[-1]
                k, v = A(f[0], sf, t), A(f[1], sf, t + 1)
            S = write(S, k, v, beta)
        both.append(cos(S @ A(X, sA, t1), A(P, sA, t1 + 1)))
        recent.append(cos(S @ A(X, sB, t2), A(Q, sB, t2 + 1)))
    return float(np.mean(both)), float(np.mean(recent))


# ------------------------------------------------------------------ F4  capacity, MATCHED N
def capacity_matched(us, ut, nfacts, beta=1.0, d=D, trials=60):
    """N facts in the same d x d matrix, whatever the binding. The comparison the old P4 missed."""
    out = []
    for _ in range(trials):
        rng = np.random.default_rng()
        A = Addr(us, ut, rng)
        base = rng.normal(size=NSLOT)
        orbit = [track(base, rng.integers(0, len(GENS), size=k + 1))[-1] for k in range(8)]
        S = np.zeros((d, d))
        facts = []
        for i in range(nfacts):
            c = unit(rng.normal(size=d))
            s = orbit[i % len(orbit)] if us else base
            v = unit(rng.normal(size=d))
            k = A(c, s, i)
            vv = A(v, s, i + 1)
            facts.append((k, vv))
            S = write(S, k, vv, beta)
        j = rng.integers(nfacts)
        out.append(cos(S @ facts[j][0], facts[j][1]))
    return float(np.mean(out))


# ------------------------------------------------------------------ F5  composition, beta axis
def compose(us, ut, depth, beta=1.0, d=D, trials=TRIALS, nfill=6):
    out = []
    for _ in range(trials):
        rng = np.random.default_rng()
        A = Addr(us, ut, rng)
        base = rng.normal(size=NSLOT)
        gens = rng.integers(0, len(GENS), size=depth + nfill + 2)
        states = [base] + track(base, gens)
        cs = unit(rng.normal(size=(depth + nfill + 3, d)))
        S = np.zeros((d, d))
        order = list(range(depth)) + list(range(depth, depth + nfill))
        rng.shuffle(order)
        for t in order:
            S = write(S, A(cs[t], states[t], t), A(cs[t + 1], states[t + 1], t + 1), beta)
        x = A(cs[0], states[0], 0)
        for _ in range(depth):
            x = S @ x
            n = np.linalg.norm(x)
            if n < 1e-10:
                x = None
                break
            x = x / n
        out.append(0.0 if x is None else cos(x, A(cs[depth], states[depth], depth)))
    return float(np.mean(out))


if __name__ == '__main__':
    print('FULL STACK  d=%d  S_%d tracker  nfreq=%d  %d trials\n' % (D, NSLOT, NFREQ, TRIALS))
    g0 = check0()
    g1 = check_tracker()

    print('\nF3  EPISODIC: one content, two tracked states, different successors')
    print('  %-22s %-12s %-12s %s' % ('binding', 'earlier', 'most recent', ''))
    for name, us, ut in BINDINGS:
        e, r = episodic(us, ut)
        print('  %-22s %-12.3f %-12.3f %s'
              % (name, e, r, 'SEPARATES' if e > 0.5 and r > 0.5 else
                 ('collides - same address' if e < 0.3 else '')), flush=True)

    print('\nF4  CAPACITY AT MATCHED FACT COUNT (same N writes, same d x d matrix)')
    NS = (8, 16, 32, 64, 96)
    print('  %-22s %s' % ('binding', '  '.join('N=%-5d' % n for n in NS)))
    for name, us, ut in BINDINGS:
        row = [capacity_matched(us, ut, n) for n in NS]
        print('  %-22s %s' % (name, '  '.join('%-7.3f' % r for r in row)), flush=True)

    print('\nF5  COMPOSITION, and beta=2 (reflection) IN THE MEMORY as the axis')
    print('  %-22s %-10s %-10s %-10s %s' % ('binding', 'b=1 d2', 'b=1 d3', 'b=2 d2', 'b=2 d3'))
    for name, us, ut in BINDINGS:
        vals = [compose(us, ut, 2, 1.0), compose(us, ut, 3, 1.0),
                compose(us, ut, 2, 2.0), compose(us, ut, 3, 2.0)]
        print('  %-22s %-10.3f %-10.3f %-10.3f %.3f' % (name, *vals), flush=True)

    print('\n  READ: F3 says whether the state factor buys situation-dependence, F4 whether it is')
    print('  paid for in capacity at equal load, F5 whether the full address still composes and')
    print('  whether reflections belong in the memory (predicted: no) as against the tracker.')
    print('\n  gates: CHECK 0 %s   tracker exact %s' % ('OK' if g0 else 'FAIL', 'OK' if g1 else 'FAIL'))
