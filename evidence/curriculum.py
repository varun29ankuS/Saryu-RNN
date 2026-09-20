"""Does a curriculum break the wall, with the architecture completely unchanged?

WHY THIS EXISTS, AND WHY IT IS LATE. Fifteen architectural interventions in this project have failed
to move 4-pair recall off ~0.32: n_h, depth, heads, optimiser, gate cap, carving, level-2
signatures, routing, write sparsity, orthogonal axes, beta_init (the delta-rule regime), key-bound
writes with many axes, query key offset, nonlinear state (tanh and top-k), and state sparsity.
Every one of them changed the model. None of them changed the number.

Chen et al., "Anatomy of Associative Recall in Fixed-State Recurrences: A Matched-State
Decomposition, an Interference Wall, and a Curriculum That Breaks It" (arXiv 2609.16183, Sep 2026)
reports the same wall independently -- cells that solve 32-pair recall fall to chance on 4 pairs
against distractors -- and finds the failure is OPTIMISATION, not expressivity. Their fix is a
DISTANCE CURRICULUM on the unchanged architecture: 0.021 -> 1.000, with lock-in improving from
1/10 to 7/10 seeds, and accuracy-gated ramping locking 6/6 at L=512.

Their "lock-in rate" is this project's solve rate, and their "interference wall" is our 0.32
attractor. The one thing we never tried is the one that works.

THE ARMS. Architecture identical in all of them -- stock SaryuV3LM, nh=2, H=8, d=128, 2 layers.
Only the ORDER OF THE TRAINING DATA changes.

    uniform    the standard task at the target difficulty throughout. The control, and what every
               previous experiment here ran.
    gap        distance curriculum: the query starts adjacent to the pairs (gap 0) and the gap
               ramps to the target. This is the arm closest to the paper's.
    pairs      start at 2 pairs -- which this model solves reliably -- and ramp to 4.
    gated      advance the gap only when held-out accuracy clears a threshold, so the schedule is
               driven by the model rather than by the step count. The paper's strongest variant.

REGISTERED PREDICTIONS, before running:
  P1  uniform stays at ~0.32, reproducing every previous run.
  P2  at least one curriculum arm reaches > 0.8 on some seeds -- the wall is crossed WITHOUT any
      architectural change.
  P3  gated has the highest solve rate, per the paper.

FALSIFIER: every arm sits at 0.32. Then the curriculum result does not transfer to this
architecture, and the wall here is not the wall they broke.

Solve rate over seeds is the statistic; escape is bimodal and a mean is meaningless.

    python evidence/curriculum.py
Environment: PAIRS, GAP, STEPS, LR, OPT, D, H, NL, NH, SEEDS, ARMS, THREADS.
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

PAIRS = int(os.environ.get('PAIRS', 4))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 6000))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
OPT = os.environ.get('OPT', 'muon')
D = int(os.environ.get('D', 128))
H = int(os.environ.get('H', 8))
NL = int(os.environ.get('NL', 2))
NH = int(os.environ.get('NH', 2))
SEEDS = int(os.environ.get('SEEDS', 4))
ARMS = os.environ.get('ARMS', 'uniform,gap,pairs,gated').split(',')
SOLVED = float(os.environ.get('SOLVED', 0.80))
NENT = 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


def make(rng, n, gap):
    ent = rng.choice(NENT, size=2 * n, replace=False)
    ks, vs = ent[:n], ent[n:]
    rest = np.setdiff1d(np.arange(NENT), ent)
    seq = []
    for k, v in zip(ks, vs):
        seq += [int(k), int(v)]
    if gap:
        seq += [int(x) for x in rng.choice(rest, size=gap, replace=True)]
    i = int(rng.integers(0, n))
    seq += [SEP, int(ks[i])]
    return seq, int(vs[i])


def batch(rng, bs, n, gap):
    xs, ys = zip(*(make(rng, n, gap) for _ in range(bs)))
    return torch.tensor(xs), torch.tensor(ys)


@torch.no_grad()
def evaluate(m, rng, n, gap, reps=4):
    m.eval()
    acc = 0.
    for _ in range(reps):
        x, y = batch(rng, BS, n, gap)
        acc += float((m(x)[:, -1].argmax(-1) == y).float().mean()) / reps
    m.train()
    return acc


def train_one(arm, seed):
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT + 2, D, NL, nh=NH, H=H)          # stock architecture, every arm
    p = list(m.parameters())
    opt = (torch.optim.AdamW(p, lr=LR, weight_decay=0.01) if OPT == 'adamw' else Muon(p, lr=LR))
    rng = np.random.default_rng(1000 + seed)
    ev = np.random.default_rng(7)
    log = Run(f'cur-{arm}-p{PAIRS}-s{seed}',
              config=dict(arm=f'curriculum/{arm}', pairs=PAIRS, gap=GAP, lr=LR, opt=OPT,
                          d=D, H=H, nh=NH, nl=NL, steps=STEPS, seed=seed,
                          chance_top1=round(1 / PAIRS, 4)))
    cur_gap = 0 if arm in ('gap', 'gated') else GAP
    cur_n = 2 if arm == 'pairs' else PAIRS
    best = 0.0
    for s in range(STEPS):
        if arm == 'gap':                                 # linear ramp over the first half
            cur_gap = min(GAP, int(GAP * (2.0 * s / STEPS)))
        elif arm == 'pairs':
            cur_n = min(PAIRS, 2 + int((PAIRS - 2) * (2.0 * s / STEPS)))
        x, y = batch(rng, BS, cur_n, cur_gap)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(p, 1.0); opt.step()
        if s == 0 or (s + 1) % 250 == 0:
            cur_acc = evaluate(m, ev, cur_n, cur_gap)    # accuracy at the CURRENT difficulty
            tgt_acc = evaluate(m, ev, PAIRS, GAP)        # and at the TARGET, which is what counts
            if arm == 'gated' and cur_acc > 0.85 and cur_gap < GAP:
                cur_gap += 1                             # advance only when the model has locked in
            best = max(best, tgt_acc)
            log.log(s + 1, loss=float(ce), **{'eval/top1': tgt_acc, 'eval/best': best,
                                              'cur/acc': cur_acc, 'cur/gap': cur_gap,
                                              'cur/pairs': cur_n})
    log.done()
    return best


def main():
    print(f'CURRICULUM  {PAIRS} pairs, target gap {GAP}, {OPT} lr {LR:g}, d={D} H={H} nh={NH} '
          f'nl={NL}, {STEPS} steps, {SEEDS} seeds')
    print('THE ARCHITECTURE IS IDENTICAL IN EVERY ARM. Only the order of the data changes.')
    print(f'accuracy is always measured at the TARGET difficulty ({PAIRS} pairs, gap {GAP})\n')
    print(f'{"arm":>9} {"solved":>8} {"rate":>6}  best per seed')
    print('-' * 56)
    t0 = time.time()
    for arm in ARMS:
        bests = [train_one(arm, s) for s in range(SEEDS)]
        ns = sum(b > SOLVED for b in bests)
        print(f'{arm:>9} {f"{ns}/{SEEDS}":>8} {ns/SEEDS:>6.2f}  '
              + ', '.join(f'{b:.3f}' for b in bests), flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ')
    print('  a curriculum arm crosses 0.8 -> the wall is OPTIMISATION and the fix is the data')
    print('                                  order, not the architecture. Fifteen interventions')
    print('                                  were the wrong axis.')
    print('  everything at 0.32           -> the published curriculum result does not transfer')
    print('                                  here, and our wall is not the one they broke.')


if __name__ == '__main__':
    main()
