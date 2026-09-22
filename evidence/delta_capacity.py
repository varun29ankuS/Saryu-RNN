"""WHICH limitation stops the delta rule matching an attention layer? Three candidates, separated.

distill_surrogate.txt measured a residual relative error of 0.38 against a softmax attention layer
(a fitted linear map gets 0.83, so the recurrence carries most of what attention does) and found
that inheriting the teacher's q/k/v/o weights buys nothing at all. Before trying to improve the
delta rule, it is worth knowing WHAT is binding, because the three plausible causes need different
fixes and improving the wrong one costs a day.

  CAPACITY.  The state is d_k x d_v per head -- 32 x 32 here. Over 256 positions a softmax layer
             can attend anywhere; a rank-limited matrix state holds on the order of d_k
             associations. Thirty-two slots for two hundred and fifty-six positions is the most
             obvious constraint in the whole setup.
  UPDATE RANK. One rank-one write per token. DeltaProduct (Siems et al., in our refs.bib) writes
             n times per token, which is the SAME primitive our transport already uses -- a
             product of Householder reflections rather than a single one -- applied to the matrix
             state instead of the vector one.
  NORMALISATION. q and k are L2-normalised, as the delta-rule family does. That discards the
             magnitude a teacher's projections encode, and is the likeliest reason inheritance
             transferred nothing.

ARMS. base reproduces the 0.38. Then each knob alone, then the two that are not mutually
exclusive together.

A CONFOUND STATED UP FRONT, because these arms are NOT parameter-matched: widening the state adds
parameters, and so does writing more than once per token. So a lower residual could be capacity,
or it could be that any model with more parameters fits better. The params column is printed for
every arm and the informative reading is whether an arm beats the trend set by the others, not
whether it beats base. An arm that improves only in proportion to its parameter count has told us
nothing about which limitation binds.

REGISTERED PREDICTIONS, before running:
  P1  CAPACITY dominates. dk_mult=4 gives the largest single improvement, because 32 slots against
      256 positions is a harder limit than anything else here. This is the prediction I would bet
      on and it is the one most likely to be wrong in an interesting way.
  P2  n_delta helps less than width at comparable parameter cost. Writing more per token raises
      the rank of the update but does NOT enlarge the state it writes into, so it should help with
      what the transition can express and not with how much can be held.
  P3  turning L2 off helps inheritance specifically, even if it does little for the from-scratch
      residual. That is the arm aimed at the failure the previous run actually found.

FALSIFIER for the whole exercise: if no arm gets meaningfully below base, the 0.38 is a property
of the delta rule itself rather than of how it is sized, and distillation into this family needs a
different student, not a bigger one.

    python evidence/delta_capacity.py
Environment: STEPS, SEEDS, D, HEADS, CTX, BS, LR, THREADS.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import AttnBlock                                    # noqa: E402
from saryu.metrics import Run                                        # noqa: E402
from distill_surrogate import (DeltaSurrogate, hidden_states, rel_err, D, HEADS,  # noqa: E402
                               CTX, BS, LR, STEPS, SEEDS)

torch.set_num_threads(int(os.environ.get('THREADS', max(1, (os.cpu_count() or 4) - 2))))

ARMS = [
    ('base',            dict()),
    ('wide2',           dict(dk_mult=2)),
    ('wide4',           dict(dk_mult=4)),
    ('delta2',          dict(n_delta=2)),
    ('delta3',          dict(n_delta=3)),
    ('noL2',            dict(l2=False)),
    ('wide2+delta2',    dict(dk_mult=2, n_delta=2)),
    ('ogate',           dict(ogate=True)),
    ('alog',            dict(alog=True)),
    ('ogate+alog',      dict(ogate=True, alog=True)),
]
ONLY = os.environ.get('ONLY')
if ONLY:
    ARMS = [a for a in ARMS if a[0] in ONLY.split(',')]


def run(name, kw, teacher, seed):
    torch.manual_seed(1000 + seed)
    student = DeltaSurrogate(D, HEADS, teacher.nkv, **kw)
    npar = sum(p.numel() for p in student.parameters())
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.0)
    log = Run(f'dc-{name}-s{seed}',
              config=dict(arm=f'delta_capacity/{name}', d=D, heads=HEADS, ctx=CTX, bs=BS,
                          steps=STEPS, seed=seed, lr=LR, params=npar, **kw))
    xv = hidden_states(BS, D, 999)
    with torch.no_grad():
        yv = teacher(xv)
    best = 1e9
    for s in range(STEPS):
        x = hidden_states(BS, D, 10_000 + s)
        with torch.no_grad():
            y = teacher(x)
        loss = nn.functional.mse_loss(student(x), y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        if (s + 1) % 100 == 0:
            with torch.no_grad():
                e = rel_err(student(xv), yv)
            best = min(best, e)
            log.log(s + 1, loss=float(loss), **{'eval/rel_err': e, 'eval/best_rel_err': best})
    log.done()
    return best, npar


def main():
    print(f'DELTA CAPACITY  d={D} heads={HEADS} ctx={CTX}, {STEPS} steps, {SEEDS} seeds')
    print('which limitation stops the delta rule matching a softmax attention layer?')
    print('baseline to beat: 0.38 residual (distill_surrogate.txt). A linear map gets 0.83.\n')
    t0 = time.time()
    res = {}
    print(f'{"arm":>14} {"params":>9} {"vs base":>9}  residual per seed')
    print('-' * 58)
    base_par = None
    for name, kw in ARMS:
        vals, npar = [], 0
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            teacher = AttnBlock(D, n_heads=HEADS)
            for p in teacher.parameters():
                p.requires_grad_(False)
            e, npar = run(name, kw, teacher, sd)
            vals.append(e)
        res[name] = (vals, npar)
        if base_par is None:
            base_par = npar
        print(f'{name:>14} {npar:>9,} {npar/base_par:>8.2f}x  '
              + ', '.join(f'{v:.4f}' for v in vals), flush=True)
    print(f'\n{time.time()-t0:.0f}s\n')

    print('READ, against the registered predictions')
    mean = {n: sum(v) / len(v) for n, (v, _) in res.items()}
    spread = {n: max(v) - min(v) for n, (v, _) in res.items()}
    floor = max(spread.values())
    if SEEDS < 2:
        print('  ONE SEED: there is no noise floor, so NOTHING below can be called an effect.')
        print('  A single seed makes every spread exactly 0.0000, which would let the check that')
        print('  exists to prevent over-reading declare every difference real. Rerun with SEEDS>=2')
        print('  before quoting any of these numbers.')
        floor = float('inf')
    else:
        print(f'  seed spread across arms: {floor:.4f} -- smaller differences are NOT real')
    if 'base' in mean:
        for n in mean:
            if n == 'base':
                continue
            gap = mean['base'] - mean[n]
            tag = ('no effect' if abs(gap) <= floor else
                   f'{"better" if gap > 0 else "WORSE"} by {abs(gap):.4f}')
            print(f'    {n:<14} {mean[n]:.4f}  ({tag}, {res[n][1]/base_par:.2f}x params)')
    wide = [n for n in mean if n.startswith('wide') and n in mean]
    dl = [n for n in mean if n.startswith('delta')]
    if wide and dl and 'base' in mean:
        bw, bd = min(wide, key=lambda n: mean[n]), min(dl, key=lambda n: mean[n])
        print(f'  P1 capacity dominates: best width {bw} {mean[bw]:.4f} vs best rank {bd} '
              f'{mean[bd]:.4f} -> {"HOLDS" if mean[bw] < mean[bd] - floor else "not shown"}')
    if all(abs(mean['base'] - mean[n]) <= floor for n in mean if n != 'base'):
        print('\n  FALSIFIER FIRED: no arm beats base beyond the seed spread. The 0.38 is a')
        print('  property of the delta rule itself, not of how it is sized, and conversion into')
        print('  this family needs a different student rather than a bigger one.')


if __name__ == '__main__':
    main()
