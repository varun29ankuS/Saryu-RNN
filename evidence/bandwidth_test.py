"""Is the limit the NUMBER OF PAIRS, or the BITS the read must carry?

WHY THIS EXISTS. Every measurement in this project conflates two things, because in standard MQAR
they move together: as pairs go 2 -> 4 -> 8, both the number of stored items AND the number of bits
the readout must deliver go up. So "n=4 fails" has never distinguished

    PAIR-COUNT LIMIT     the state cannot keep four associations apart
    BANDWIDTH LIMIT      the read delivers about one bit, whatever is stored

The second is the sharper hypothesis, and it comes from the architecture rather than from capacity:
a reflection writes along ONE direction u with amplitude -beta*(u.h), set by the existing state and
not by the value, and a query touches the state only through the SCALAR u_q.h. One scalar per
reflection per head at read time. There is no bilinear (key x value) write and no bilinear read
anywhere in the layer. If that is the constraint, then what matters is log2(choices) at the query,
not how many pairs sit in the state.

Note this is NOT a capacity regime at all. An ideal one-shot read of an HRR store at d=128 gets
0.983 at EIGHT pairs (retrieval_phase_transition.txt). Our model fails at FOUR, and fails at
exactly 1/n rather than degrading -- and an SNR-limited read never sits exactly at chance. So the
failure is orders of magnitude below any superposition bound and is not shaped like one.

THE DESIGN THAT SEPARATES THEM. Draw n pairs but let the values come from only m distinct symbols,
each used n/m times. The state must still hold n associations; the query still selects among m.

    (n=2, m=2)   1 bit, 2 pairs    the configuration that already works (1.000)
    (n=4, m=2)   1 bit, 4 pairs    THE TEST
    (n=8, m=2)   1 bit, 8 pairs    the stronger version
    (n=4, m=4)   2 bits, 4 pairs   the configuration that already fails (0.25)
    (n=8, m=8)   3 bits, 8 pairs   fails harder

REGISTERED PREDICTIONS, before running:
  P1  BANDWIDTH: accuracy tracks m and ignores n. (4,2) and (8,2) go near 1.0; (4,4) and (8,8)
      stay at 1/m.
  P2  PAIR-COUNT: accuracy tracks n and ignores m. (4,2) sits at 0.5 = chance despite needing the
      same single bit that (2,2) delivers perfectly.

FALSIFIER for the bandwidth account: (4,2) lands at chance. Then the number of stored pairs is the
limit and "one reflection, one scalar" is wrong.

Chance is 1/m, so compare each arm against its OWN chance line, printed alongside.

    python evidence/bandwidth_test.py
Environment: ARMS, GAP, STEPS, LR, OPT, NH, NL, SEED, SEEDS, THREADS.
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

# arms are "pairs:values"
ARMS = os.environ.get('ARMS', '2:2,4:2,8:2,4:4,8:8')
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 6000))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
OPT = os.environ.get('OPT', 'muon')
NH = int(os.environ.get('NH', 2))
NL = int(os.environ.get('NL', 2))
SEEDS = int(os.environ.get('SEEDS', 2))
D, NENT = 128, 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 8)))


def make(rng, n, m):
    """n pairs whose values are drawn from only m distinct symbols, each used n/m times."""
    ent = rng.choice(NENT, size=n + m, replace=False)
    ks, vpool = ent[:n], ent[n:]
    reps = np.repeat(np.arange(m), n // m)            # which symbol each pair gets
    rng.shuffle(reps)
    vs = vpool[reps]                                  # the value of each pair
    rest = np.setdiff1d(np.arange(NENT), ent)
    seq = []
    for k, v in zip(ks, vs):
        seq += [int(k), int(v)]
    seq += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, n))
    seq += [SEP, int(ks[i])]
    return seq, int(vs[i])


def batch(rng, bs, n, m):
    xs, ys = zip(*(make(rng, n, m) for _ in range(bs)))
    return torch.tensor(xs), torch.tensor(ys)


def train_one(n, m, seed):
    torch.manual_seed(seed)
    mdl = SaryuV3LM(NENT + 2, D, NL, nh=NH)
    p = list(mdl.parameters())
    opt = (torch.optim.AdamW(p, lr=LR, weight_decay=0.01) if OPT == 'adamw' else Muon(p, lr=LR))
    rng = np.random.default_rng(1000 + seed)
    ev = np.random.default_rng(7)
    log = Run(f'bw-p{n}-v{m}-s{seed}',
              config=dict(arm=f'bandwidth/p{n}v{m}', pairs=n, values=m,
                          bits=round(float(np.log2(m)), 2), lr=LR, opt=OPT, nh=NH, nl=NL,
                          steps=STEPS, chance_top1=round(1 / m, 4)))
    every = max(1, STEPS // 30)
    best = 0.0
    for s in range(STEPS):
        x, y = batch(rng, BS, n, m)
        ce = F.cross_entropy(mdl(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(p, 1.0); opt.step()
        if s == 0 or (s + 1) % every == 0:
            mdl.eval()
            with torch.no_grad():
                acc = 0.
                for _ in range(4):
                    x2, y2 = batch(ev, BS, n, m)
                    acc += float((mdl(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            mdl.train()
            best = max(best, acc)
            log.log(s + 1, loss=float(ce), **{'eval/top1': acc})
    log.done()
    return best


def main():
    arms = [tuple(int(v) for v in a.split(':')) for a in ARMS.split(',')]
    print(f'BANDWIDTH vs PAIR COUNT   {OPT} lr {LR:g}, n_h {NH}, {NL} layers, {STEPS} steps, '
          f'{SEEDS} seeds')
    print('n pairs stored, values drawn from m distinct symbols -> the read must deliver '
          'log2(m) bits\n')
    print(f'{"pairs":>6} {"values":>7} {"bits":>6} {"chance":>8} {"best top1":>11}   verdict')
    print('-' * 62)
    t0 = time.time()
    for n, m in arms:
        assert n % m == 0, f'{n}:{m} -- pairs must divide evenly among values'
        accs = [train_one(n, m, s) for s in range(SEEDS)]
        a = float(np.mean(accs))
        ch = 1.0 / m
        verdict = ('SOLVED' if a > 0.5 + ch / 2 else 'at chance' if a < ch * 1.35 else 'partial')
        print(f'{n:>6} {m:>7} {np.log2(m):>6.1f} {ch:>8.3f} {a:>11.3f}   {verdict}  '
              f'(seeds {", ".join(f"{x:.3f}" for x in accs)})', flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ')
    print('  (4,2) and (8,2) solved  -> the limit is BITS AT THE READ, not pairs stored.')
    print('                             "one reflection, one scalar" survives.')
    print('  (4,2) at chance         -> the limit is the NUMBER OF PAIRS; bandwidth account fails.')


if __name__ == '__main__':
    main()
