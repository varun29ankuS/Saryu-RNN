"""CAN THE MATRIX HOLD *WHEN* AS WELL AS *WHAT*?  Pure numpy, CPU, no training.

THE QUESTION. Today a fact is addressed by content alone: k_t = phi(x_{t-1}). Two consequences,
one wanted and one not:

  wanted    the same subject always lands at the same address, so the newest fact about it wins.
            That is SEMANTIC memory and the delta rule gives it for free (it is an overwrite).
  not       you cannot ask "what followed X the FIRST time". The second write DESTROYS the first -
            not fades, destroys, because S <- S - beta(Sk-v)k^T with beta=1 fits the new value
            exactly at that address. There is no episode, only a current belief.

Decay does NOT fix this. dec[:,t] scales the WHOLE state, making old facts fainter but not
DISTINGUISHABLE. Faint and overwritten are different failures.

So: put time in the key too. k_t = code(phi(x_{t-1}), t). Then one subject at two times occupies
two addresses and both survive. The cost, and the reason this is not free, is that the address
space is still rank d: spending it on time leaves less for content.

THE CONSTRAINT THAT MAKES THIS NON-OBVIOUS, and the reason arbitrary time tags are wrong:
tying requires the VALUE to be a legal KEY, or S^d stops walking (compose_probe: untied 0.120 at
two hops, chance). If the key carries time and the value does not, tying breaks and deferred
composition - the one mechanism here that is ours - dies with it. The code must therefore apply
to BOTH sides consistently: S maps code(p_{t-1}, t-1) -> code(p_t, t), so the output at step t
IS the legal key at step t+1 and composition survives. Both codes below are built that way; the
arms differ in WHICH code, not in whether it is applied.

  none   code(p,t) = p                 today's behaviour, the control
  rot    code(p,t) = R^t p             a ROTATION, RoPE-style, nfreq distinct angles. A group
                                       action: R^a R^b = R^(a+b), so the code telescopes and
                                       "how long ago" is a relative phase. Nearby times give
                                       CORRELATED keys -> revisits close in time should INTERFERE.
  tag    code(p,t) = p * s_t           independent random sign vector per step (Hadamard/HRR
                                       binding, an involution). Every time is orthogonal to every
                                       other, so no interference at any gap - and no notion of
                                       "recent" either.

PRE-REGISTERED PREDICTIONS, written before running:
  P1  composition survives for ALL THREE arms, because each code is applied to key and value
      alike. If `rot` or `tag` loses composition, my argument above is wrong, not the code.
  P2  `none` scores ~0 on the FIRST of two successors and ~1 on the second (overwrite, not decay).
  P3  `rot` recovers the first successor only when the gap is large; small gaps interfere. The
      interference gap should SHRINK as nfreq rises - this is the same phenomenon already measured
      in the drawing line (commit 4c7d82c: "~15-20 px interference radius at K_CLK=4, halved at
      K_CLK=7"), and finding it again here in a different medium would be real corroboration.
  P4  `tag` recovers both successors at every gap, but pays in CAPACITY versus `none`.

RESULTS, and TWO OF THE FOUR PREDICTIONS WERE WRONG:

  P1  HELD. Composition is untouched by either code - none/rot/tag all 1.000/1.000/0.999 at
      depths 1/2/3 and ~0.98 at depth 7. Applying the code to key and value alike does keep the
      value a legal key, as argued.
  P2  HELD, and starkly. Content-only addressing returns the first successor at -0.012/0.005/
      0.000/-0.010/-0.008 across gaps - that is ZERO, not a faded value. The delta rule at
      beta=1 fits the new value exactly at that address and the old one is gone. Both time codes
      recover it (rot 0.70-0.84, tag 0.75-0.90) while keeping the second at ~0.92.
  P3  WRONG, AND THE TEST WAS CONFOUNDED. I predicted small gaps interfere and large gaps
      recover; the raw numbers declined with gap, the opposite direction. Both readings are
      void: test_revisit writes a filler edge at every non-visit step, so gap=16 stores ~15 more
      facts than gap=1 and the decline is CAPACITY. `tag` uses independent per-step signs and
      cannot have phase interference at any gap, yet declined nearly as much - the tell.
      temporal_gap.py holds the write count constant and finds NO clock-distance effect at all
      (tag spread 0.007; rot nfreq=1 slope -0.001 across gaps 0..32). There is no interference
      radius here. What matters is nfreq, the clock's RESOLUTION: 0.502 at nfreq=1 rising to
      0.878 at nfreq=16, which matches `tag`'s 0.885. So the drawing line's K_CLK finding carries
      over as "more frequencies separate better", NOT as "nearby times collide".
  P4  WRONG, in our favour - but this test was too easy to fail, so trust it only that far. I
      predicted a capacity tax for spending address space on time. There is none: at n=8/32/96
      distinct edges none scores 0.967/0.845/0.563, rot 0.972/0.838/0.527, tag 0.973/0.839/0.570,
      identical within noise. READ test_capacity BEFORE BELIEVING THAT: it draws INDEPENDENT
      random keys, and a norm-preserving code applied to random vectors returns random vectors,
      just as separable. The result could not have come out any other way. It therefore rules out
      a PER-FACT tax and nothing more. The capacity question that actually bites is one content
      at MANY times - the revisit case scaled up - and that is not measured anywhere here.

CONSEQUENCE: rotation is the one to use. It ties tag once nfreq is large enough, and it is a
group action, so "how long ago" is a relative phase. The machinery is RoPE's - but this is NOT
a free ride, and saying "already fused in every kernel" would be wrong: in RoPE the phase cancels
inside q.k, whereas here the READ-OUT carries R^(t+1) and must be un-rotated before it leaves the
block, or the tied map's next key is wrong. That is a change to the block, not a pre-processing
step. Random tags win nothing and are structureless.

SCOPE - WHAT THIS DID NOT TEST. The question asked was content x STATE x time. This measures
content x time only. Putting the tracked state in as well (the BIND=3 conjunctive key in
stage_c.py, whose header still records the address as IDENTITY-ONLY) is the next probe. Because
a rotation is norm-preserving it SHOULD commute with a conjunctive address - but that is an
argument, not a measurement, and arguments of exactly that shape are 0 for 2 on this page.

usage: python experiments/temporal_bind.py
"""
from __future__ import annotations

import numpy as np

D = 64
TRIALS = 200


# ----------------------------------------------------------------- the time codes
def make_rot(d, nfreq, rng):
    """Block-diagonal 2x2 rotations. `nfreq` distinct angles assigned to the d/2 pairs cyclically,
    so nfreq is a clock resolution knob exactly like K_CLK in the drawing runs."""
    half = d // 2
    ang = rng.uniform(0.15, np.pi - 0.15, size=nfreq)
    return ang[np.arange(half) % nfreq]          # (half,) angle per coordinate pair


def apply_rot(p, t, ang):
    half = len(ang)
    a, b = p[..., :half], p[..., half:]
    c, s = np.cos(t * ang), np.sin(t * ang)
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)


def code(p, t, arm, ctx):
    if arm == 'none':
        return p
    if arm == 'rot':
        return apply_rot(p, t, ctx['ang'])
    if arm == 'tag':
        return p * ctx['sign'][t]
    raise ValueError(arm)


def make_ctx(arm, d, L, nfreq, rng):
    if arm == 'rot':
        return dict(ang=make_rot(d, nfreq, rng))
    if arm == 'tag':
        return dict(sign=rng.choice([-1.0, 1.0], size=(L + 8, d)))
    return {}


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def write(S, k, v, beta=1.0):
    return S - beta * np.outer(S @ k - v, k)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(a @ b / (na * nb))


# ----------------------------------------------------------------- P1: composition
def test_compose(arm, depth, d=D, nfreq=4, trials=TRIALS):
    """Walk a clean chain of `depth` edges, then iterate S from the start and see if we arrive."""
    out = []
    for _ in range(trials):
        rng = np.random.default_rng()
        ctx = make_ctx(arm, d, depth + 4, nfreq, rng)
        phi = unit(rng.normal(size=(depth + 2, d)))
        S = np.zeros((d, d))
        for t in range(depth):
            S = write(S, code(phi[t], t, arm, ctx), code(phi[t + 1], t + 1, arm, ctx))
        x = code(phi[0], 0, arm, ctx)
        for _ in range(depth):
            x = unit(S @ x)
        out.append(cos(x, code(phi[depth], depth, arm, ctx)))
    return float(np.mean(out))


# ------------------------------------------------- P2/P3: two successors for one node
def test_revisit(arm, gap, d=D, nfreq=4, nfill=6, trials=TRIALS):
    """Node X is visited at t1 with successor A and again at t1+gap with successor B.
    Filler edges surround both. Can we get A back, and can we still get B?"""
    first, second = [], []
    L = gap + nfill + 4
    for _ in range(trials):
        rng = np.random.default_rng()
        ctx = make_ctx(arm, d, L, nfreq, rng)
        X, A, B = unit(rng.normal(size=(3, d)))
        fill = unit(rng.normal(size=(L, 2, d)))
        t1, t2 = 1, 1 + gap
        S = np.zeros((d, d))
        for t in range(L):
            if t == t1:
                k, v = X, A
            elif t == t2:
                k, v = X, B
            else:
                k, v = fill[t]
            S = write(S, code(k, t, arm, ctx), code(v, t + 1, arm, ctx))
        #   query the SAME node at each of its two visit times
        rA = S @ code(X, t1, arm, ctx)
        rB = S @ code(X, t2, arm, ctx)
        first.append(cos(rA, code(A, t1 + 1, arm, ctx)))
        second.append(cos(rB, code(B, t2 + 1, arm, ctx)))
    return float(np.mean(first)), float(np.mean(second))


# ----------------------------------------------------------------- P4: capacity cost
def test_capacity(arm, nedge, d=D, nfreq=4, trials=80):
    """Distinct edges, no revisits. Mean recall of a randomly chosen one."""
    out = []
    for _ in range(trials):
        rng = np.random.default_rng()
        ctx = make_ctx(arm, d, nedge + 4, nfreq, rng)
        ks = unit(rng.normal(size=(nedge, d)))
        vs = unit(rng.normal(size=(nedge, d)))
        S = np.zeros((d, d))
        for t in range(nedge):
            S = write(S, code(ks[t], t, arm, ctx), code(vs[t], t + 1, arm, ctx))
        j = rng.integers(nedge)
        out.append(cos(S @ code(ks[j], j, arm, ctx), code(vs[j], j + 1, arm, ctx)))
    return float(np.mean(out))


if __name__ == '__main__':
    ARMS = ('none', 'rot', 'tag')
    print('TEMPORAL BINDING IN THE MATRIX   d=%d, %d trials' % (D, TRIALS))
    print('the code is applied to BOTH key and value, so the value stays a legal key')
    print()

    print('P1  COMPOSITION must survive the time code (else the time code is disqualified)')
    print('    %-6s %s' % ('arm', '  '.join('d=%d' % x for x in (1, 2, 3, 5, 7))))
    for arm in ARMS:
        row = [test_compose(arm, dep) for dep in (1, 2, 3, 5, 7)]
        print('    %-6s %s' % (arm, '  '.join('%.3f' % r for r in row)), flush=True)
    print()

    print('P2/P3  ONE NODE, TWO SUCCESSORS. "first" is the fact a content-only address destroys.')
    print('    %-6s %-6s %-8s %-8s  %s' % ('arm', 'gap', 'first', 'second', ''))
    for arm in ARMS:
        for gap in (1, 2, 4, 8, 16):
            f, s = test_revisit(arm, gap)
            print('    %-6s %-6d %-8.3f %-8.3f  %s'
                  % (arm, gap, f, s, 'both recovered' if f > 0.5 and s > 0.5 else
                     ('first LOST' if s > 0.5 else 'neither')), flush=True)
    print()

    print('P3  INTERFERENCE vs CLOCK RESOLUTION for `rot` - does the bad gap shrink with nfreq?')
    print('    (drawing line found the radius halving from K_CLK=4 to K_CLK=7, commit 4c7d82c)')
    print('    %-8s %s' % ('nfreq', '  '.join('g=%-5d' % g for g in (1, 2, 4, 8, 16))))
    for nf in (1, 2, 4, 7, 16):
        row = [test_revisit('rot', g, nfreq=nf)[0] for g in (1, 2, 4, 8, 16)]
        print('    %-8d %s' % (nf, '  '.join('%-7.3f' % r for r in row)), flush=True)
    print()

    print('P4  CAPACITY COST - distinct edges, no revisits. Time in the key is not free.')
    print('    %-6s %s' % ('arm', '  '.join('n=%-6d' % n for n in (8, 16, 32, 48, 64, 96))))
    for arm in ARMS:
        row = [test_capacity(arm, n) for n in (8, 16, 32, 48, 64, 96)]
        print('    %-6s %s' % (arm, '  '.join('%-8.3f' % r for r in row)), flush=True)
    print()
    print('  READING THIS: a time code EARNS its place only if it keeps P1 (composition), wins P2')
    print('  (the first successor, which content-only addressing cannot hold at all), and does not')
    print('  collapse P4. Losing any one of the three disqualifies it regardless of the other two.')
