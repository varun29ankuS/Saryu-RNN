"""Is binding capacity set by WRITE SPARSITY rather than by binding strength?

WHERE THIS CAME FROM. Reading the block line by line rather than reasoning about it. forward()
computes

    y = self.out(s * F.silu(self.og(z)))

so the read is an elementwise product of the state with a query-dependent vector -- a DIAGONAL
bilinear form, not a full one. (This also corrects an earlier claim in this project that there is
no bilinear read at all; there is one, it is just diagonal.)

A per-coordinate reweighting can only separate items whose writes occupy DIFFERENT coordinates.
Measured on the 25M checkpoint over real text (trained_geometry.txt):

    write c effective support   ~30 of dh=93
    output gate support         ~11 of dh=93

Every item therefore contributes to every coordinate the gate looks at, and the number of items
that can be held in near-disjoint codes is about dh/support ~ 3. The binding cliff in this project
is at 3-4.

THE PREDICTION THAT DISTINGUISHES THIS FROM EVERYTHING ELSE TRIED. If capacity is dh/support, then
making the write SPARSER should raise it, and making the binding STRONGER should not. The second
half is already measured: nh_ratio.txt swept nh from 2 to 32 -- a 16x change in binding strength,
with the hand-built ceiling moving 0.347 -> 0.973 -- and trained accuracy never left 0.31-0.34.

REGISTERED PREDICTIONS, before running, at dh = 128 with 4 pairs:
  P1  dense (write_topk=None) fails, ~0.32, reproducing every other arm in this project.
  P2  write_topk = 32 (capacity dh/k = 4) is the threshold arm: it should start to work.
  P3  write_topk = 16 and 8 (capacity 8 and 16) solve.
  P4  the ordering is monotone in dh/k, not in anything else.

FALSIFIER: every arm lands at 0.32 like the nh sweep. Then the diagonal-read account is wrong too,
and the 0.31-0.34 attractor is not about the code's support either.

CAVEAT KEPT IN VIEW. The support numbers above were measured on the LANGUAGE MODEL, which is not
doing 4-pair binding, so ~30 may be a property of enwik8 rather than of the mechanism. That is
exactly why this is run as a trained experiment rather than asserted.

    python evidence/write_sparsity.py
Environment: KS, PAIRS, GAP, STEPS, LR, OPT, H, D, NL, NH, SEEDS, THREADS.
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

KS = [None if x in ('none', 'None', '0') else int(x)
      for x in os.environ.get('KS', 'none,32,16,8').split(',')]
PAIRS = int(os.environ.get('PAIRS', 4))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 6000))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
OPT = os.environ.get('OPT', 'muon')
H = int(os.environ.get('H', 1))
D = int(os.environ.get('D', 128))
NL = int(os.environ.get('NL', 2))
NH = int(os.environ.get('NH', 2))
SEEDS = int(os.environ.get('SEEDS', 4))
SOLVED = float(os.environ.get('SOLVED', 0.80))
NENT = 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 8)))


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


def train_one(k, seed):
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT + 2, D, NL, nh=NH, H=H, write_topk=k)
    p = list(m.parameters())
    opt = (torch.optim.AdamW(p, lr=LR, weight_decay=0.01) if OPT == 'adamw' else Muon(p, lr=LR))
    rng = np.random.default_rng(1000 + seed)
    ev = np.random.default_rng(7)
    dh = D // H
    log = Run(f'ws-p{PAIRS}-k{k if k else "dense"}-dh{dh}-s{seed}',
              config=dict(arm=f'write_sparsity/k{k if k else "dense"}', pairs=PAIRS,
                          write_topk=k, dh=dh, d=D, H=H, nh=NH, nl=NL, lr=LR, opt=OPT,
                          steps=STEPS, seed=seed,
                          capacity_dh_over_k=(dh // k if k else None),
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
    print(f'WRITE SPARSITY  {PAIRS} pairs, d={D} H={H} dh={dh}, nh={NH}, {OPT} lr {LR:g}, '
          f'{STEPS} steps, {SEEDS} seeds')
    print('the read is DIAGONAL -- out(s * silu(og(z))) -- so items must occupy different')
    print(f'coordinates to be separable; predicted capacity is dh/k\n')
    print(f'{"write_topk":>11} {"dh/k":>6} {"solved":>8} {"rate":>6}  seeds')
    print('-' * 56)
    t0 = time.time()
    for k in KS:
        bests = [train_one(k, s) for s in range(SEEDS)]
        ns = sum(b > SOLVED for b in bests)
        print(f'{str(k) if k else "dense":>11} {(dh // k if k else 0):>6} {f"{ns}/{SEEDS}":>8} '
              f'{ns/SEEDS:>6.2f}  ' + ', '.join(f'{b:.3f}' for b in bests), flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ')
    print('  solve rate rises as k falls -> capacity is set by WRITE SPARSITY; the diagonal read')
    print('                                 is the constraint and dh/k is the number of slots.')
    print('  every arm at ~0.32          -> the diagonal-read account is wrong too, and the')
    print('                                 attractor is not about the code support either.')


if __name__ == '__main__':
    main()
