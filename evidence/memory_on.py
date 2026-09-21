"""Does Saryu with memory=True solve what Saryu without it cannot?

This is the acceptance test for the matrix memory just added to saryu/model.py, not an
exploration. The model is the shipped SaryuV3LM; the only thing that varies is the flag.

LOGGED BASELINES on this exact task (4-pair MQAR, chance 0.250):
    saryu, the shipped recurrence      0.289 - 0.336   (cell-saryu-p4-s0/1/2, base-saryu-p4-s0)
    a 290k transformer                 1.000 x3        (base-transformer-p4-s0/1/2)
    a faithful DeltaNet                1.000 x3        (dref-p4-L2-pos-lr0.001-s0/1/2)

REGISTERED PREDICTION: memory=True exceeds 0.8 and memory=False reproduces ~0.32.

FALSIFIER: memory=True stays near 0.32. Then the component that reaches 1.000 standalone does
not work inside this block, the integration is wrong rather than the idea, and nothing about the
matrix should be claimed for Saryu until that is found.

    python evidence/memory_on.py
Environment: STEPS, SEEDS, LR, THREADS.
"""
from __future__ import annotations
import os, sys, time
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                    # noqa: E402
from saryu.metrics import Run                                       # noqa: E402

STEPS = int(os.environ.get('STEPS', 3000))
SEEDS = int(os.environ.get('SEEDS', 2))
LR = float(os.environ.get('LR', 1e-3))
D, H, NH, NL, BS, GAP, NENT = 128, 8, 2, 2, 32, 4, 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


def make(rng, n):
    e = rng.choice(NENT, size=2 * n, replace=False); ks, vs = e[:n], e[n:]
    rest = np.setdiff1d(np.arange(NENT), e); s = []
    for a, b in zip(ks, vs):
        s += [int(a), int(b)]
    s += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, n)); s += [SEP, int(ks[i])]
    return s, int(vs[i])


def batch(rng, bs, n):
    o = [make(rng, n) for _ in range(bs)]
    return torch.tensor([a for a, _ in o]), torch.tensor([b for _, b in o])


def train(memory, seed):
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT + 2, D, NL, nh=NH, H=H, memory=memory)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(1000 + seed)
    log = Run(f'memon-{"on" if memory else "off"}-p4-s{seed}',
              config=dict(arm=f'memory_on/{"on" if memory else "off"}', pairs=4, params=npar,
                          memory=memory, steps=STEPS, seed=seed, lr=LR, chance_top1=0.25))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS, 4)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 250 == 0:
            m.eval()
            with torch.no_grad():
                ev = np.random.default_rng(7); a = 0.
                for _ in range(4):
                    x2, y2 = batch(ev, BS, 4)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train(); best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    return best, npar


def main():
    print(f'MEMORY ON/OFF  shipped SaryuV3LM, 4 pairs, d={D} H={H} nh={NH} L={NL}, '
          f'{STEPS} steps, {SEEDS} seeds, chance 0.250')
    print('logged: saryu 0.289-0.336   transformer 1.000 x3   faithful DeltaNet 1.000 x3\n')
    t0 = time.time()
    for memory in (False, True):
        r = [train(memory, s) for s in range(SEEDS)]
        print(f'  memory={str(memory):<5} {r[0][1]:>9,} params   '
              + ', '.join(f'{x:.3f}' for x, _ in r), flush=True)
    print(f'\n{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
