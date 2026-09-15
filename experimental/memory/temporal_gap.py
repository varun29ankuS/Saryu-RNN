"""CORRECTION TO temporal_bind.py's P3. The gap axis there was confounded.

WHAT I GOT WRONG, found by reading test_revisit rather than its output. That function sets
L = gap + nfill + 4 and writes a FILLER EDGE at every step that is not one of the two visits.
So gap=16 stores about fifteen more facts than gap=1. The reported decline

    rot   first-successor  0.837 (g=1) -> 0.695 (g=16)
    tag   first-successor  0.901 (g=1) -> 0.751 (g=16)

is therefore what a rank-d matrix does when you put more in it, and says NOTHING about temporal
distance. `tag` uses independent per-step signs and CANNOT have phase interference at any gap,
yet it declined by almost the same amount - which is the tell. Two variables moved; I read the
result as if one had.

I also predicted the opposite DIRECTION (small gaps interfere, large gaps recover). Had the
confound not been there, a decline would have looked like a refutation of rotation. It was not
evidence either way.

THE FIX: decouple clock time from write index. The number of writes between the two visits is
held CONSTANT at w2-1; only the clock jumps.

    clock_of(i) = i  if i < w2  else  i + gap

so the two visits to X sit w2+gap apart on the clock with the same number of intervening facts.

PREDICTIONS, registered before running:
  Q1  `tag` is FLAT in gap. Independent signs have no distance structure, so any slope here is
      the harness still leaking, not the code. This is the control that validates the design.
  Q2  `rot` at nfreq=1 RISES with gap: one angle means keys at nearby clock times are strongly
      correlated, and correlated keys collide under the delta rule.
  Q3  that rise flattens as nfreq grows, because more angles decorrelate nearby times faster.
      nfreq=16 should look like `tag`.

NOT TESTED HERE: composition. The clock jump breaks the telescoping property that made the value
a legal key, which is fine because P1 in temporal_bind.py already covered composition on an
unbroken clock, and the two questions are separate.

usage: cd experiments && python temporal_gap.py
"""
from __future__ import annotations

import numpy as np

from temporal_bind import code, cos, make_ctx, unit, write

D = 64
TRIALS = 300
W = 12          # total writes, CONSTANT across every cell
W2 = 6          # write-index of the second visit; 5 intervening facts, CONSTANT


def revisit_fixed_writes(arm, gap, d=D, nfreq=4, trials=TRIALS, w=W, w2=W2):
    """X is visited at write 0 and write w2. Intervening writes are constant; only the clock moves."""
    first, second = [], []
    for _ in range(trials):
        rng = np.random.default_rng()
        ctx = make_ctx(arm, d, w + gap + 8, nfreq, rng)
        X, A, B = unit(rng.normal(size=(3, d)))
        fill = unit(rng.normal(size=(w, 2, d)))
        clock = [i if i < w2 else i + gap for i in range(w)]
        S = np.zeros((d, d))
        for i in range(w):
            if i == 0:
                k, v = X, A
            elif i == w2:
                k, v = X, B
            else:
                k, v = fill[i]
            c = clock[i]
            S = write(S, code(k, c, arm, ctx), code(v, c + 1, arm, ctx))
        c1, c2 = clock[0], clock[w2]
        first.append(cos(S @ code(X, c1, arm, ctx), code(A, c1 + 1, arm, ctx)))
        second.append(cos(S @ code(X, c2, arm, ctx), code(B, c2 + 1, arm, ctx)))
    return float(np.mean(first)), float(np.mean(second))


GAPS = (0, 1, 2, 4, 8, 16, 32)

if __name__ == '__main__':
    print('CLOCK DISTANCE WITH WRITE COUNT HELD CONSTANT   d=%d, %d writes, %d trials'
          % (D, W, TRIALS))
    print('every cell stores the SAME number of facts; only the clock gap between the two')
    print('visits to X changes. temporal_bind.py P3 varied both at once.')
    print()
    print('  first-successor recall')
    print('  %-14s %s' % ('arm', '  '.join('g=%-6d' % g for g in GAPS)))
    print('  ' + '-' * 70)

    tag_row = [revisit_fixed_writes('tag', g)[0] for g in GAPS]
    print('  %-14s %s' % ('tag (control)', '  '.join('%-8.3f' % r for r in tag_row)), flush=True)
    none_row = [revisit_fixed_writes('none', g)[0] for g in GAPS]
    print('  %-14s %s' % ('none', '  '.join('%-8.3f' % r for r in none_row)), flush=True)
    print()
    rot_rows = {}
    for nf in (1, 2, 4, 7, 16):
        r = [revisit_fixed_writes('rot', g, nfreq=nf)[0] for g in GAPS]
        rot_rows[nf] = r
        print('  %-14s %s' % ('rot nfreq=%d' % nf, '  '.join('%-8.3f' % x for x in r)), flush=True)

    print()
    print('  VERDICTS')
    spread = max(tag_row) - min(tag_row)
    print('  Q1  tag flat?      spread %.3f  -> %s'
          % (spread, 'FLAT, design is clean' if spread < 0.05 else
             'NOT FLAT - the harness still leaks, rot rows below are unreadable'))
    r1 = rot_rows[1]
    print('  Q2  rot nfreq=1 rises with gap?  g=0 %.3f -> g=32 %.3f  (%+.3f)  -> %s'
          % (r1[0], r1[-1], r1[-1] - r1[0],
             'YES, nearby clock times collide' if r1[-1] - r1[0] > 0.05 else 'NO'))
    for nf in (1, 2, 4, 7, 16):
        r = rot_rows[nf]
        print('  Q3  nfreq=%-3d slope %+.3f   worst cell %.3f   gap-to-tag at g=0 %+.3f'
              % (nf, r[-1] - r[0], min(r), r[0] - tag_row[0]))
