"""Where along the gap axis does the matrix memory stop working?

both_fixes.txt found the memory at 0.930/0.938 on 4 pairs at gap 4 and 0.039/0.062 at gap 16.
Something breaks in between, and composition cannot be tested until we know where -- two fixes
cannot be shown to interact in a regime where neither works alone.

This sweeps the gap at a budget MATCHED to memory_on.txt (3000 steps), so the 1500-step confound in
both_fixes is removed. memory=True throughout; the question is not whether the memory helps but how
far it carries.

    gap    0    4    8   12   16        4 pairs, chance 0.250

REGISTERED PREDICTIONS:
  P1  gap 4 reproduces ~0.93, confirming the budget is the difference and not something else about
      this harness.
  P2  there is a knee, not a slow decline -- the memory works until the gap exceeds what the gate
      retains (measured at 4-8 tokens without the bound) and then fails abruptly.
  P3  the knee sits near gap 8, because that is where the gap first exceeds gate retention.

FALSIFIER for P2/P3: a smooth decline with no knee, or a knee somewhere unrelated to the retention
figure. Then retention is not what limits the memory's range and the capacity/persistence split
does not explain this.

The largest gap where the memory still works is where the composition test belongs.

    python evidence/gap_breakdown.py
Environment: STEPS, SEEDS, GAPS, THREADS.
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                   # noqa: E402
from saryu.metrics import Run                                       # noqa: E402
warnings.filterwarnings('ignore', category=UserWarning)

STEPS = int(os.environ.get('STEPS', 3000))
SEEDS = int(os.environ.get('SEEDS', 2))
GAPS = [int(g) for g in os.environ.get('GAPS', '0,4,8,12,16').split(',')]
PAIRS, D, H, NH, NL, BS, NENT = 4, 128, 8, 2, 2, 32, 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


def make(rng, gap):
    e = rng.choice(NENT, size=2 * PAIRS, replace=False); ks, vs = e[:PAIRS], e[PAIRS:]
    rest = np.setdiff1d(np.arange(NENT), e); s = []
    for a, b in zip(ks, vs):
        s += [int(a), int(b)]
    if gap:
        s += [int(x) for x in rng.choice(rest, size=gap, replace=True)]
    i = int(rng.integers(0, PAIRS)); s += [SEP, int(ks[i])]
    return s, int(vs[i])


def batch(rng, bs, gap):
    o = [make(rng, gap) for _ in range(bs)]
    return torch.tensor([a for a, _ in o]), torch.tensor([b for _, b in o])


def train(gap, seed):
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT + 2, D, NL, nh=NH, H=H, memory=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    rng = np.random.default_rng(1000 + seed)
    log = Run(f'gb-g{gap}-s{seed}', config=dict(arm=f'gap_breakdown/g{gap}', pairs=PAIRS, gap=gap,
              nl=NL, memory=True, steps=STEPS, seed=seed, chance_top1=0.25,
              params=sum(p.numel() for p in m.parameters())))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS, gap)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 250 == 0:
            ev = np.random.default_rng(7); a = 0.
            m.eval()
            with torch.no_grad():
                for _ in range(4):
                    x2, y2 = batch(ev, BS, gap)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train(); best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    return best


def main():
    print(f'GAP BREAKDOWN  memory=True, {PAIRS} pairs, nl={NL}, {STEPS} steps, {SEEDS} seeds, '
          f'chance 0.250')
    print('memory_on.txt: gap 4 -> 0.930, 0.938.   both_fixes.txt: gap 16 -> 0.039, 0.062\n')
    print(f'{"gap":>5} {"seq":>5}  best per seed        solved>0.8')
    print('-' * 48)
    t0 = time.time()
    for g in GAPS:
        r = [train(g, s) for s in range(SEEDS)]
        print(f'{g:>5} {2*PAIRS+g+2:>5}  ' + ', '.join(f'{x:.3f}' for x in r)
              + f'        {sum(x > 0.8 for x in r)}/{SEEDS}', flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ: the largest gap with a solve is where the composition test belongs.')


if __name__ == '__main__':
    main()
