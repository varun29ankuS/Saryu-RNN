"""A factorial, because every one-factor test in this project has come back null.

WHY. Twenty interventions here changed exactly one thing each -- n_h, depth, beta, orthogonality,
the gate cap, write sparsity, key-bound writes, the query offset, state nonlinearity -- and all
twenty returned ~0.32. A one-factor design is structurally blind to effects that only exist in
combination, and this project's own evidence says the effects ARE combinatorial:

    write_operator (constructed, dh=128, 4 pairs)
        key-bound write alone, nh=2 ................ 0.282   (chance 0.250)
        orthogonal axes alone (additive write) ..... 0.250
        key-bound AND orthogonal AND nh = dh/4 ..... 0.980

Neither ingredient did anything alone. I measured that, wrote it down, and then spent six more
hours testing one knob at a time.

THE DESIGN. 2x2x2, with nh held FIXED at 17 in every cell so that binding varies without dragging
nh along with it (the usual confound when bind_n is switched on).

    bind    0  = no carve, plain additive write
            16 = carve + bind_write with 16 binding axes, so the write carries an operator
                 derived from the item's own key
    orth    mutually orthogonal reflection axes, making the transport a true involution
    cur     uniform (4 pairs throughout) vs slow (equal thirds at 2, 3, 4 pairs)

REGISTERED PREDICTIONS, before running:
  P1  MAIN EFFECT of curriculum. It is the only factor that has moved anything alone
      (curriculum2: 8/16 seeds above 0.5 against 0/8 uniform, Fisher p ~ 0.01).
  P2  INTERACTION between bind and orth: neither alone beats its control, both together do.
      This is the constructed result above, tested in training for the first time.
  P3  the best cell is all three on, and it beats curriculum alone.

FALSIFIER: no cell beats curriculum alone. Then the architecture factors contribute nothing even in
combination, the combinatorial reading is wrong too, and the data order is the whole story.

Solve rate over seeds is the statistic; escape is bimodal. A secondary threshold at 0.5 is reported
because curriculum2 showed the effect lives there while nothing crosses 0.8 -- both are stated in
advance so neither is chosen after the fact.

    python evidence/factorial.py
Environment: NPAIRS, GAP, STEPS, LR, OPT, D, H, NL, NHF, SEEDS, THREADS.
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                     # noqa: E402
from saryu.metrics import Run                                        # noqa: E402
from binding_long import Muon                                        # noqa: E402

TARGET = int(os.environ.get('NPAIRS', 4))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 4000))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
OPT = os.environ.get('OPT', 'muon')
D = int(os.environ.get('D', 128))
H = int(os.environ.get('H', 1))              # dh = 128, so carve=64 is legal
NL = int(os.environ.get('NL', 2))
NHF = int(os.environ.get('NHF', 17))         # nh fixed across ALL cells: bind varies, nh does not
SEEDS = int(os.environ.get('SEEDS', 2))
NENT = 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


def make(rng, n):
    ent = rng.choice(NENT, size=2 * n, replace=False)
    ks, vs = ent[:n], ent[n:]
    rest = np.setdiff1d(np.arange(NENT), ent)
    s = []
    for k, v in zip(ks, vs):
        s += [int(k), int(v)]
    s += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, n))
    s += [SEP, int(ks[i])]
    return s, int(vs[i])


def batch(rng, bs, n):
    o = [make(rng, n) for _ in range(bs)]
    return torch.tensor([a for a, _ in o]), torch.tensor([b for _, b in o])


@torch.no_grad()
def evaluate(m, rng, n, reps=4):
    m.eval()
    a = 0.
    for _ in range(reps):
        x, y = batch(rng, BS, n)
        a += float((m(x)[:, -1].argmax(-1) == y).float().mean()) / reps
    m.train()
    return a


def train_one(bind, orth, cur, seed):
    torch.manual_seed(seed)
    kw = dict(nh=NHF, H=H, orth_axes=orth)
    if bind:
        kw.update(carve=64, bind_write=True, bind_n=bind)
    m = SaryuV3LM(NENT + 2, D, NL, **kw)
    p = list(m.parameters())
    opt = (torch.optim.AdamW(p, lr=LR, weight_decay=0.01) if OPT == 'adamw' else Muon(p, lr=LR))
    rng = np.random.default_rng(1000 + seed)
    ev = np.random.default_rng(7)
    tag = f'f-b{bind}-o{int(orth)}-{cur}-s{seed}'
    log = Run(tag, config=dict(arm=f'factorial/b{bind}-o{int(orth)}-{cur}', pairs=TARGET,
                               bind_n=bind, orth_axes=orth, curriculum=cur, nh=NHF, H=H, d=D,
                               nl=NL, lr=LR, opt=OPT, steps=STEPS, seed=seed,
                               chance_top1=round(1 / TARGET, 4)))
    best = 0.0
    for s in range(STEPS):
        n = TARGET if cur == 'uniform' else min(TARGET, 2 + int(3.0 * s / STEPS))
        x, y = batch(rng, BS, n)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(p, 1.0); opt.step()
        if s == 0 or (s + 1) % 250 == 0:
            tgt = evaluate(m, ev, TARGET)
            best = max(best, tgt)
            log.log(s + 1, loss=float(ce), **{'eval/top1': tgt, 'eval/best': best,
                                              'cur/pairs': n})
    log.done()
    return best


def main():
    print(f'FACTORIAL  {TARGET} pairs, d={D} H={H} dh={D//H}, nh FIXED at {NHF} in every cell, '
          f'{STEPS} steps, {SEEDS} seeds')
    print('one-factor designs are blind to combinations; this project\'s own construction says the')
    print('effects ARE combinatorial (0.282 / 0.250 alone, 0.980 together)\n')
    print(f'{"bind":>5} {"orth":>5} {"cur":>8} {">0.8":>6} {">0.5":>6}  best per seed')
    print('-' * 62)
    t0 = time.time()
    rows = []
    for bind, orth, cur in itertools.product((0, 16), (False, True), ('uniform', 'slow')):
        bests = [train_one(bind, orth, cur, s) for s in range(SEEDS)]
        n8 = sum(b > 0.8 for b in bests)
        n5 = sum(b > 0.5 for b in bests)
        rows.append((bind, orth, cur, n8, n5, bests))
        print(f'{bind:>5} {str(orth):>5} {cur:>8} {f"{n8}/{SEEDS}":>6} {f"{n5}/{SEEDS}":>6}  '
              + ', '.join(f'{b:.3f}' for b in bests), flush=True)
    print(f'\n{time.time()-t0:.0f}s\n')
    # main effects and the interaction that P2 is about
    def mean_where(cond):
        sel = [r for r in rows if cond(r)]
        return float(np.mean([b for r in sel for b in r[5]])) if sel else float('nan')
    print('MAIN EFFECTS (mean best over all cells with that setting)')
    print(f'  bind  0 {mean_where(lambda r: r[0] == 0):.3f}   '
          f'bind 16 {mean_where(lambda r: r[0] == 16):.3f}')
    print(f'  orth  F {mean_where(lambda r: not r[1]):.3f}   '
          f'orth  T {mean_where(lambda r: r[1]):.3f}')
    print(f'  uniform {mean_where(lambda r: r[2] == "uniform"):.3f}   '
          f'slow    {mean_where(lambda r: r[2] == "slow"):.3f}')
    print('\nBIND x ORTH interaction (P2): neither alone, both together')
    for b in (0, 16):
        for o in (False, True):
            print(f'  bind {b:>2} orth {str(o):>5}: '
                  f'{mean_where(lambda r, b=b, o=o: r[0] == b and r[1] == o):.3f}')
    print('\nREAD')
    print('  a cell beats curriculum alone -> the architecture factors matter in COMBINATION and')
    print('                                   twenty one-factor nulls were a design artefact')
    print('  nothing beats curriculum      -> the data order is the whole story')


if __name__ == '__main__':
    main()
