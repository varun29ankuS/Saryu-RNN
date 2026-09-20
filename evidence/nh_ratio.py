"""The decisive test: does training at nh ~ dh/4 actually solve 4 pairs?

WHY THIS IS THE ONE THAT MATTERS. expressivity.txt hand-built the MQAR solution in the exact
shipped recurrence with nothing trained, and found that at nh = 2 NO construction reaches four
pairs: the ceiling is 0.347 at dh = 128, against 0.21-0.25 for the trained model. Training is
therefore at 60-72% of the constructible optimum rather than far below it, which says the limit is
EXPRESSIVITY and not optimisation -- reversing the reading of three earlier files.

That conclusion rests on a single hand-built construction, and failing to build a solution is not a
proof that none exists. This test puts it at risk. The construction predicts specific numbers at
dh = 128:

    nh= 2   0.347     nh= 8   0.550     nh=16   0.838     nh=32   0.973

so if expressivity is really the binding constraint, TRAINED accuracy should climb the same ladder
and nh = 32 should solve reliably where nh = 2 cannot.

NOTE ON THE HEAD WIDTH. The controlling quantity is the RATIO nh/dh, through tr(H)/dh = 1-2nh/dh
(Paper B's trace law). The default H = 8 gives dh = 16, where even nh = dh/2 only reaches 0.848 in
the construction, so the test uses H = 1 and dh = 128. Every previous nh sweep in this project
varied nh with dh held at 16 and so never moved the ratio far.

REGISTERED PREDICTIONS, before running:
  P1  solve rate rises monotonically with nh.
  P2  nh = 32 (= dh/4) solves reliably -- most or all seeds above 0.8 -- where nh = 2 solves none.
  P3  accuracy tracks the construction ladder above, within the noise of a bimodal escape.

FALSIFIER, and it is the point of the run: nh = 32 fails as badly as nh = 2. Then the construction
was misleading, expressivity is NOT the limit, optimisation is back on the table, and
expressivity.txt must be retracted.

Solve rate over seeds is the statistic, not mean accuracy: escape is bimodal and a mean over a
1.000 seed and a 0.500 seed describes nothing that happened.

    python evidence/nh_ratio.py
Environment: NHS, PAIRS, GAP, STEPS, LR, OPT, H, D, NL, SEEDS, THREADS.
"""
from __future__ import annotations

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

NHS = [int(x) for x in os.environ.get('NHS', '2,8,16,32').split(',')]
PAIRS = int(os.environ.get('PAIRS', 4))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 6000))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
OPT = os.environ.get('OPT', 'muon')
H = int(os.environ.get('H', 1))
D = int(os.environ.get('D', 128))
NL = int(os.environ.get('NL', 2))
SEEDS = int(os.environ.get('SEEDS', 4))
ORTH = os.environ.get('ORTH', '0') == '1'   # mutually orthogonal axes -> T is an involution
SOLVED = float(os.environ.get('SOLVED', 0.80))
NENT = 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 8)))
# construction ceilings from expressivity.txt at dh=128, for the record
PREDICTED = {2: 0.347, 4: 0.393, 8: 0.550, 16: 0.838, 32: 0.973, 64: 0.998}


def make(rng, n):
    ent = rng.choice(NENT, size=2 * n, replace=False)
    ks, vs = ent[:n], ent[n:]
    rest = np.setdiff1d(np.arange(NENT), ent)
    seq = []
    for k, v in zip(ks, vs):
        seq += [int(k), int(v)]
    seq += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, n))
    seq += [SEP, int(ks[i])]
    return seq, int(vs[i])


def batch(rng, bs, n):
    xs, ys = zip(*(make(rng, n) for _ in range(bs)))
    return torch.tensor(xs), torch.tensor(ys)


def train_one(nh, seed):
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT + 2, D, NL, nh=nh, H=H, orth_axes=ORTH)
    p = list(m.parameters())
    opt = (torch.optim.AdamW(p, lr=LR, weight_decay=0.01) if OPT == 'adamw' else Muon(p, lr=LR))
    rng = np.random.default_rng(1000 + seed)
    ev = np.random.default_rng(7)
    dh = D // H
    log = Run(f'nhr-p{PAIRS}-nh{nh}-dh{dh}{"-orth" if ORTH else ""}-s{seed}',
              config=dict(arm=f'nh_ratio/nh{nh}', pairs=PAIRS, nh=nh, H=H, dh=dh, d=D, nl=NL,
                          lr=LR, opt=OPT, steps=STEPS, seed=seed,
                          trace_ratio=round(1 - 2 * nh / dh, 3),
                          orth_axes=ORTH,
                          constructed_ceiling=PREDICTED.get(nh),
                          chance_top1=round(1 / PAIRS, 4)))
    best = 0.0
    for s in range(STEPS):
        x, y = batch(rng, BS, PAIRS)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(p, 1.0); opt.step()
        if s == 0 or (s + 1) % 250 == 0:
            m.eval()
            with torch.no_grad():
                acc = 0.
                for _ in range(4):
                    x2, y2 = batch(ev, BS, PAIRS)
                    acc += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train()
            best = max(best, acc)
            log.log(s + 1, loss=float(ce), **{'eval/top1': acc, 'eval/best': best})
    log.done()
    return best


def main():
    dh = D // H
    print(f'NH RATIO TEST  {PAIRS} pairs, d={D} H={H} dh={dh}, {OPT} lr {LR:g}, {STEPS} steps, '
          f'{SEEDS} seeds')
    print('the construction ceiling per nh is from expressivity.txt (hand-built, untrained)\n')
    print(f'{"nh":>4} {"tr/dh":>7} {"built":>7} {"solved":>8} {"rate":>6} {"best top1":>11}')
    print('-' * 50)
    t0 = time.time()
    for nh in NHS:
        bests = [train_one(nh, s) for s in range(SEEDS)]
        ns = sum(b > SOLVED for b in bests)
        pr = PREDICTED.get(nh)
        print(f'{nh:>4} {1-2*nh/dh:>7.3f} {(f"{pr:.3f}" if pr else "--"):>7} '
              f'{f"{ns}/{SEEDS}":>8} {ns/SEEDS:>6.2f} {max(bests):>11.3f}   '
              f'seeds {", ".join(f"{b:.3f}" for b in bests)}', flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ')
    print('  solve rate climbs with nh, nh=32 reliable -> EXPRESSIVITY confirmed; the fix is a')
    print('                                               sizing rule, nh ~ dh/4.')
    print('  nh=32 fails like nh=2                     -> the construction misled; optimisation is')
    print('                                               back and expressivity.txt is retracted.')


if __name__ == '__main__':
    main()
