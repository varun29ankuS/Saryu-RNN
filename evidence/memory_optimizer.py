"""Does the optimizer explain the matrix memory's slow convergence, and its bimodality?

WHY. Two days of matrix-memory experiments all used AdamW. This repository's own evidence says
that is the wrong choice for recall:

    optimizer_escape.txt, 2-pair MQAR, identical lr 3e-4, same seed
        step 1500   MUON 1.000   ADAMW 0.474
        step 2500   MUON 0.995   ADAMW 0.542
    Muon SOLVES it; AdamW stalls near 0.5.

    optimizer_local.txt, enwik8, 3 seeds
        mean bpc is a tie (2.2463 vs 2.2464) but the SEED SPREAD is 0.0202 against 0.0054,
        i.e. Muon is 3.7x tighter.

    optimizer_5m.txt concluded "keep AdamW" -- correctly, for LANGUAGE MODELLING, where loss was
    a tie. Every experiment since has been associative RECALL, where our own data says otherwise.

It also bears on two things that were blamed on the architecture:
  the ~1.8x excess steps-to-converge over a faithful DeltaNet at matched gap (750 transformer,
    1250 DeltaNet, 2250 Saryu+memory), and
  the bimodality that has appeared everywhere today -- 1.000/0.031, 0.398/0.992 -- which was
    called a seed lottery. A 3.7x tighter seed spread is what sitting further from a chaotic
    boundary looks like (Sohl-Dickstein 2024, the fractal hyperparameter boundary).

NOTE ON WHAT IS ALREADY MIXED. gate_retention.py always used Muon, so the gate-bound result
(6/7 against 1/3) is a MUON result. memory_on, gap_breakdown, both_fixes, state_volume,
delta_reference and width_scaling all used AdamW. Our two headline results were measured under
different optimizers and nobody chose that.

REGISTERED PREDICTIONS:
  P1  Muon beats AdamW on final score at gap 4, or reaches it in fewer steps.
  P2  Muon's seed spread is smaller. This is the one that matters most, because if the
      bimodality is an optimizer artefact then several conclusions drawn today were reading
      optimizer behaviour and calling it architecture.
  P3  the gap between Saryu+memory and a faithful DeltaNet narrows under Muon.

FALSIFIER: Muon is no better on any of the three. Then AdamW was a fine choice, the excess over
DeltaNet is architectural, and the bimodality is real rather than an artefact.

    python evidence/memory_optimizer.py
Environment: STEPS, SEEDS, GAP, THREADS.
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                  # noqa: E402
from saryu.metrics import Run                                      # noqa: E402
from binding_long import Muon                                      # noqa: E402
warnings.filterwarnings('ignore', category=UserWarning)

STEPS = int(os.environ.get('STEPS', 3000))
SEEDS = int(os.environ.get('SEEDS', 3))
GAP = int(os.environ.get('GAP', 4))
PAIRS, D, H, NH, NL, BS, NENT = 4, 128, 8, 2, 2, 32, 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 4)))
ARMS = [('adamw', 1e-3), ('muon', 3e-4), ('muon', 1e-3)]


def make(rng):
    e = rng.choice(NENT, size=2*PAIRS, replace=False); ks, vs = e[:PAIRS], e[PAIRS:]
    rest = np.setdiff1d(np.arange(NENT), e); s = []
    for a, b in zip(ks, vs): s += [int(a), int(b)]
    if GAP: s += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, PAIRS)); s += [SEP, int(ks[i])]
    return s, int(vs[i])


def batch(rng, bs):
    o = [make(rng) for _ in range(bs)]
    return torch.tensor([a for a,_ in o]), torch.tensor([b for _,b in o])


def train(opt_name, lr, seed):
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT+2, D, NL, nh=NH, H=H, memory=True)
    ps = list(m.parameters())
    opt = (torch.optim.AdamW(ps, lr=lr, weight_decay=0.01) if opt_name == 'adamw'
           else Muon(ps, lr=lr))
    rng = np.random.default_rng(1000+seed)
    log = Run(f'mopt-{opt_name}{lr:g}-g{GAP}-s{seed}',
              config=dict(arm=f'memory_optimizer/{opt_name}@{lr:g}', pairs=PAIRS, gap=GAP,
                          opt=opt_name, lr=lr, steps=STEPS, seed=seed, memory=True,
                          chance_top1=round(1/PAIRS, 4),
                          params=sum(p.numel() for p in ps)))
    best, first80 = 0., None
    for s in range(STEPS):
        x, y = batch(rng, BS)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(ps, 1.0); opt.step()
        if (s+1) % 250 == 0:
            ev = np.random.default_rng(7); a = 0.
            m.eval()
            with torch.no_grad():
                for _ in range(4):
                    x2, y2 = batch(ev, BS)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean())/4
            m.train(); best = max(best, a)
            if first80 is None and a > 0.8: first80 = s+1
            log.log(s+1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    return best, first80


def main():
    print(f'MEMORY OPTIMIZER  {PAIRS} pairs, gap {GAP}, memory=True, nl={NL}, {STEPS} steps, '
          f'{SEEDS} seeds, chance {1/PAIRS:.3f}')
    print('reference at this config: AdamW 0.930/0.938 (memory_on), 2250-2500 steps to 0.8')
    print('faithful DeltaNet 1250, transformer 750\n')
    print(f'{"arm":>14}  best per seed              spread   steps to 0.8')
    print('-'*68)
    t0 = time.time()
    for name, lr in ARMS:
        r = [train(name, lr, s) for s in range(SEEDS)]
        b = [x for x,_ in r]; f = [x for _,x in r]
        print(f'{f"{name}@{lr:g}":>14}  ' + ', '.join(f'{x:.3f}' for x in b)
              + f'   {max(b)-min(b):>8.3f}   ' + ', '.join(str(x) for x in f), flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ')
    print('  P2 is the one that matters: a smaller seed spread under Muon means the bimodality')
    print('  seen all day was an optimizer artefact, and several conclusions drawn from AdamW')
    print('  runs were reading optimizer behaviour and calling it architecture.')


if __name__ == '__main__':
    main()
