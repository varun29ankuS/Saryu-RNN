"""DOES COMPOSITION SURVIVE THE BLOCK'S ACTUAL DYNAMICS?  Pure numpy, CPU.

Every composition number on record was measured under conditions the shipping block does not have:

    compose_probe          tied 1.000 to depth 7        beta = 1, NO decay
    deferred_composition   0.960 at depth 3, shuffled   beta = 1, NO decay
    temporal_bind P1       0.98 at depth 7              beta = 1, NO decay

ParityBlock has both. beta = bmax*sigmoid(.) is learned and generally < 1, so each write is a
PARTIAL fit rather than an exact one. And the decay is Mamba-style:

    g = -exp(A_log) * softplus(a_proj(z) + dt_bias),  dec = exp(g)
    A ~ U(0,16) at init, dt log-uniform [1e-3, 0.1]  ->  g ~ -(0 .. 1.6),  dec ~ (0.20 .. 1.00)

so at init some heads forget almost everything in three steps while others keep nearly everything.
That spread is Mamba's design and is not a bug. The question is whether the mechanism we are
claiming can live anywhere inside it.

WHY DECAY IS NOT OBVIOUSLY HARMLESS. It is applied to the WHOLE state each step, so every edge is
scaled by dec^(age). The pointer chase normalises between hops, so a uniform shrink of S cancels
and is irrelevant. What does NOT cancel is the RELATIVE attenuation: an edge written at step t has
been decayed (T-t) times, so the OLDEST edge is the faintest - and walking a chain from its start
uses the oldest edge FIRST. Decay therefore attacks exactly the hop composition depends on most.

PRE-REGISTERED, written before running:
  R1  at dec=1.0, beta=1.0 this must reproduce ~1.000 to depth 5, or the harness is wrong and
      nothing below it counts.
  R2  composition degrades as dec falls, and the degradation is WORSE at greater depth and at
      greater sequence length (more steps of ageing between write and read).
  R3  there is a critical dec below which multi-hop is at chance while ONE hop still works -
      i.e. recall survives where composition does not. If so, decay is the knob that separates
      our claim from the incumbent's, and the tied arm needs the slow-decay end of the range.
  R4  beta < 1 costs less than decay does, because a partial write still points the right way -
      it is the direction of S.k that the chase follows, and normalisation discards the length.

RESULTS.

  R1  ROUGHLY held, with a caveat worth stating: dec=1.0, beta=1.0 gives 0.909 at depth 3 and
      0.903 at depth 5, not the ~1.000 I registered. The difference is the 8 filler edges, which
      compose_probe did not have - they cost capacity in a rank-d matrix. The harness is sane;
      my prediction forgot its own setup.
  R2  HELD, in all three directions. Composition falls with dec (0.944 -> 0.094 from dec 1.0 to
      0.5 at depth 3), falls faster at greater depth (at dec=0.90: 0.938/0.907/0.870/0.753 for
      d=1/2/3/5), and falls MUCH faster with sequence length (see below).
  R3  HELD, but the right statement is about AGE, and "composition is frailer than recall" is
      the wrong way to say it. Compare the two dec=0.80 rows:
        in order   1 hop 0.460      shuffled   1 hop 0.796
      Same decay, same task, the same single edge. What differs is that writing in path order
      makes edge 0->1 the OLDEST write in the stream, aged ~10 steps (0.8^10 ~ 0.11). A ONE-hop
      query of an OLD fact dies exactly as fast as a three-hop walk does. So decay is a tax on
      AGE, not a tax on composition per se. The asymmetry is real but indirect: a walk MUST
      reach the oldest edge in its chain, whereas MQAR-style recall happens to query recent ones.
      Composition pays most because it always holds the longest-dated claim.
  R4  HELD, strongly, and it is good news. beta is nearly free: at dec=1.0, three-hop composition
      is 0.940 / 0.944 / 0.944 / 0.947 / 0.933 for beta 1.0 / 0.9 / 0.7 / 0.5 / 0.3. A partial
      write still points the right way, and the chase normalises the length away. Learned beta<1
      is not a threat to the mechanism. Decay is.

TWO THINGS I DID NOT PREDICT.

  SHUFFLED BEATS IN-ORDER, and the margin grows as decay bites: 0.944 vs 0.909 at dec=1.0, but
  0.694 vs 0.215 at dec=0.80. That looks backwards until you count ages. In-order writes all
  chain edges FIRST and then 8 fillers, so every chain edge ages 8 extra steps; shuffling spreads
  them and lowers the mean age. This is direct confirmation that the harm mechanism is relative
  attenuation BY AGE, which is what the header argued and what the decay-vs-depth table alone
  could not have distinguished from a generic "decay is bad".

  SEQUENCE LENGTH COMPOUNDS WITH DECAY. At NO decay at all, depth-3 composition runs
  0.989 / 0.943 / 0.790 / 0.422 for 0 / 8 / 24 / 56 filler edges. At dec=0.90 the same row is
  0.982 / 0.879 / 0.363 / 0.010. Read this narrowly: 59 edges in a d=64 matrix is simply the
  rank limit, which is old news and not a discovery about decay. It does NOT say long context is
  impossible - GPU head_dim is 128 (4x the state), and in real text most tokens do not write a
  novel edge. The finding is only that the two costs multiply rather than add, so a capacity
  headroom that looks adequate at dec=1 can be inadequate at dec=0.9.

CONSEQUENCE FOR THE GPU RUN - WITH THE ARITHMETIC DONE. My first draft of this paragraph said the
Mamba init puts "a large share of heads where composition provably cannot run". That cited the
worst case as if it were the typical one. At init softplus(a_proj(z) + dt_bias) = dt exactly
(that is what the inverse-softplus init makes true at a_proj(z) ~ 0), so

    dec = exp(-A*dt),   A ~ U(0,16),   dt log-uniform [1e-3, 0.1]

The MEDIAN head is A=8, dt=0.01 -> dec ~ 0.92. Integrating, roughly 46% of heads start at
dec >= 0.95, so at H=8 the chance that no head starts slow is about 0.54^8, near 1%. And the
shuffled table puts depth-3 composition at dec=0.80 at 0.694 - degraded, not dead. "Cannot run"
is fair only below dec ~ 0.5, which is the bottom tenth of the range.

Honest version: decay taxes age, composition pays the most, about half the heads start where it
works, and only one of them needs to. The init is not wrong.

What IS worth flagging: shift_arms.TiedBlock sets nn.init.constant_(self.dec.bias, 3.0), i.e.
dec ~ 0.95 - an init chosen for precisely this reason - and the Mamba spread replaced it in the
name of parity. That is the THIRD time a parity feature has worked against the mechanism, after
the key-side conv and the k-only normalisation. So the GPU script carries a dec_init FLAG,
defaulting to Mamba: offered, not forced, because forcing slow decay on the tied arm alone would
hand it an advantage the untied arm does not get, and the result would be unreadable in the other
direction.

NOT TESTED: whether gradient descent finds the slow-decay heads on its own. Everything above is
hand-set decay; a trained block learns A_log and dt_bias. That is the open question and it is not
answerable on CPU.

usage: cd experiments && python theory_vs_dynamics.py

REFERENCES
  [Gu & Dao 2023] Mamba: Linear-Time Sequence Modeling with Selective State Spaces, arXiv 2312.00752
"""
from __future__ import annotations

import numpy as np

D = 64
TRIALS = 200


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def walk(S, start, depth):
    """The pointer chase, exactly as the block does it: normalise between hops."""
    x = start.copy()
    for _ in range(depth):
        x = S @ x
        n = np.linalg.norm(x)
        if n < 1e-10:
            return None
        x = x / n
    return x


def run(depth, dec, beta, nfill, shuffled, d=D, trials=TRIALS):
    """Write a chain of `depth` edges plus `nfill` unrelated edges, then walk `depth` hops.

    `shuffled` puts the chain edges in RANDOM order in the stream - the deferred-composition
    claim. Unshuffled writes them in path order, which is the easier case.
    """
    hit1, hitd = [], []
    for _ in range(trials):
        rng = np.random.default_rng()
        phi = unit(rng.normal(size=(depth + 1, d)))
        edges = [(phi[i], phi[i + 1]) for i in range(depth)]
        fill = [(f[0], f[1]) for f in unit(rng.normal(size=(nfill, 2, d)))]
        stream = edges + fill
        order = rng.permutation(len(stream)) if shuffled else np.arange(len(stream))
        S = np.zeros((d, d))
        for idx in order:
            k, v = stream[idx]
            S = dec * S                                   # decay the WHOLE state, every step
            S = S - beta * np.outer(S @ k - v, k)         # partial delta-rule write
        r1 = walk(S, phi[0], 1)
        rd = walk(S, phi[0], depth)
        hit1.append(0.0 if r1 is None else float(r1 @ phi[1]))
        hitd.append(0.0 if rd is None else float(rd @ phi[depth]))
    return float(np.mean(hit1)), float(np.mean(hitd))


DECS = (1.0, 0.99, 0.95, 0.90, 0.80, 0.50)

if __name__ == '__main__':
    print('COMPOSITION UNDER THE BLOCK\'S REAL DYNAMICS   d=%d, %d trials' % (D, TRIALS))
    print('ParityBlock init gives dec in ~(0.20, 1.00) per head, so this whole range is in play.')
    print()

    print('R1/R2  depth-3 chain, 8 filler edges, chain written IN ORDER')
    print('  %-10s %-12s %-12s %s' % ('dec', '1 hop', '3 hops', ''))
    for dec in DECS:
        h1, hd = run(3, dec, 1.0, 8, False)
        print('  %-10.2f %-12.3f %-12.3f %s'
              % (dec, h1, hd, 'composition GONE' if hd < 0.3 <= h1 else
                 ('both gone' if h1 < 0.3 else '')), flush=True)
    print()

    print('R2/R3  same, but chain edges SHUFFLED into the stream (deferred composition)')
    print('  %-10s %-12s %-12s %s' % ('dec', '1 hop', '3 hops', ''))
    for dec in DECS:
        h1, hd = run(3, dec, 1.0, 8, True)
        print('  %-10.2f %-12.3f %-12.3f %s'
              % (dec, h1, hd, 'composition GONE' if hd < 0.3 <= h1 else
                 ('both gone' if h1 < 0.3 else '')), flush=True)
    print()

    print('R2  does more DEPTH hurt more?  (dec x depth, 8 fillers, shuffled)')
    print('  %-10s %s' % ('dec', '  '.join('d=%-6d' % x for x in (1, 2, 3, 5))))
    for dec in DECS:
        row = [run(x, dec, 1.0, 8, True)[1] for x in (1, 2, 3, 5)]
        print('  %-10.2f %s' % (dec, '  '.join('%-8.3f' % r for r in row)), flush=True)
    print()

    print('R2  does a longer SEQUENCE hurt more?  (dec x fillers, depth 3, shuffled)')
    print('  %-10s %s' % ('dec', '  '.join('fill=%-4d' % x for x in (0, 8, 24, 56))))
    for dec in DECS:
        row = [run(3, dec, 1.0, x, True)[1] for x in (0, 8, 24, 56)]
        print('  %-10.2f %s' % (dec, '  '.join('%-9.3f' % r for r in row)), flush=True)
    print()

    print('R4  partial writes: beta < 1 at dec=1.0 (isolating beta from decay), depth 3 shuffled')
    print('  %-10s %-12s %s' % ('beta', '1 hop', '3 hops'))
    for b in (1.0, 0.9, 0.7, 0.5, 0.3):
        h1, hd = run(3, 1.0, b, 8, True)
        print('  %-10.2f %-12.3f %-12.3f' % (b, h1, hd), flush=True)
    print()

    print('  WHAT WOULD CHANGE THE GPU PLAN: if composition needs dec very close to 1 while one-hop')
    print('  recall tolerates much less, then the tied arm is not competing with the incumbent on')
    print('  equal terms - it needs the slow-forgetting end of the head range, and a decay init')
    print('  spanning (0.20, 1.00) is spending most of its heads where the mechanism cannot run.')
