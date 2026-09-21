"""How WIDE is the learning rate region where each model trains at all?

THE TWO OPEN PUZZLES. Saryu+memory takes 2250 steps to reach 0.8 on 4-pair MQAR where a faithful
DeltaNet takes 1250 and a transformer 750, and every arm in this project is bimodal. The optimizer
was ruled out for the first (2250 steps under every optimizer and rate tested) and is unresolved
for the second.

ONE MECHANISM COULD EXPLAIN BOTH. Sohl-Dickstein (2024) shows the boundary between hyperparameters
where training converges and diverges is FRACTAL, and that "the best hyperparameters for neural
network training are usually very near the edge of stability". This script was launched because
matrix_decision.txt appeared to record exactly that for us: "DeltaNet needs 3e-3; Saryu diverges
there (loss 4.03) and at 1e-2 (4.18). 1e-3 was already its stability limit."

THAT MOTIVATING QUOTE IS WRONG TWICE, and both errors were found only after this ran. First, the
whole file is retracted -- none of its figures has a run behind it (CLAIMS #25). Second, even taken
at face value it misuses the word: a 64-way task has chance loss ln 64 = 4.159, so "4.03" and
"4.18" describe a model sitting AT CHANCE, not one diverging. Nothing diverged. Which means this
script measured solved-versus-unsolved while Sohl-Dickstein's boundary is converged-versus-diverged
-- a different quantity -- and that is why its result file records it as inconclusive by design.
Left in place as the record of a test whose premise dissolved under checking.

If Saryu's convergent region is NARROW and DeltaNet's is WIDE, then Saryu's optimum sits against
its own divergence boundary -- which would make it both slower (it cannot use a larger rate) and
more seed-sensitive (small perturbations cross the boundary). One mechanism, both symptoms.

WHAT IS MEASURED. Not the best score. The WIDTH of the region: across a geometric learning-rate
sweep, at how many rates does the model learn anything at all, and how far is its best rate from
the rate at which it stops learning.

REGISTERED PREDICTIONS:
  P1  Saryu+memory's convergent region is NARROWER than a faithful DeltaNet's, counted in rates
      that clear a low bar (0.40 against chance 0.25).
  P2  Saryu's best rate sits closer to its upper edge. Quantified as the ratio
      (highest working rate) / (best rate) -- larger means more headroom.
  P3  the transformer, which converges fastest, has the widest region of the three.

FALSIFIER: the regions are comparable in width. Then stability margin explains neither the excess
steps nor the bimodality, both stay unexplained, and the fractal reading does not apply to us.

    python evidence/stability_margin.py
Environment: STEPS, SEEDS, ARCH, LRS, THREADS.
"""
from __future__ import annotations
import os, sys, time, math, warnings
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                  # noqa: E402
from saryu.metrics import Run                                      # noqa: E402
import delta_reference as DR                                       # noqa: E402
warnings.filterwarnings('ignore', category=UserWarning)

STEPS = int(os.environ.get('STEPS', 1200))
SEEDS = int(os.environ.get('SEEDS', 1))
LRS = [float(x) for x in os.environ.get(
    'LRS', '3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2').split(',')]
ARCHS = os.environ.get('ARCH', 'saryu,delta').split(',')
PAIRS, GAP, D, H, NH, NL, BS, NENT = 4, 4, 128, 8, 2, 2, 32, 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


def make(rng):
    e = rng.choice(NENT, size=2*PAIRS, replace=False); ks, vs = e[:PAIRS], e[PAIRS:]
    rest = np.setdiff1d(np.arange(NENT), e); s = []
    for a, b in zip(ks, vs): s += [int(a), int(b)]
    s += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, PAIRS)); s += [SEP, int(ks[i])]
    return s, int(vs[i])


def batch(rng, bs):
    o = [make(rng) for _ in range(bs)]
    return torch.tensor([a for a,_ in o]), torch.tensor([b for _,b in o])


def build(arch):
    if arch == 'saryu':
        return SaryuV3LM(NENT+2, D, NL, nh=NH, H=H, memory=True)
    if arch == 'delta':
        return DR.Net(NENT+2, D, DR.HEADS, DR.HEADDIM, nl=NL, neg_eig=False)
    raise ValueError(arch)


def train(arch, lr, seed):
    torch.manual_seed(seed)
    m = build(arch)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(1000+seed)
    log = Run(f'stab-{arch}-lr{lr:g}-s{seed}',
              config=dict(arm=f'stability/{arch}', arch=arch, lr=lr, pairs=PAIRS, gap=GAP,
                          steps=STEPS, seed=seed, chance_top1=0.25,
                          params=sum(p.numel() for p in m.parameters())))
    best, diverged = 0., False
    for s in range(STEPS):
        x, y = batch(rng, BS)
        ce = F.cross_entropy(m(x)[:, -1], y)
        if not torch.isfinite(ce):
            diverged = True; break
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s+1) % 200 == 0:
            ev = np.random.default_rng(7); a = 0.
            m.eval()
            with torch.no_grad():
                for _ in range(4):
                    x2, y2 = batch(ev, BS)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean())/4
            m.train(); best = max(best, a)
            log.log(s+1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    return best, diverged


def main():
    print(f'STABILITY MARGIN  {PAIRS} pairs, gap {GAP}, nl={NL}, {STEPS} steps, {SEEDS} seed(s), '
          f'chance 0.250')
    print('matrix_decision.txt: "DeltaNet needs 3e-3; Saryu diverges there ... 1e-3 was already')
    print('its stability limit." This measures the WIDTH of each convergent region.\n')
    print(f'{"lr":>8}  ' + '  '.join(f'{a:>10}' for a in ARCHS))
    print('-'*(10+12*len(ARCHS)))
    res = {a: {} for a in ARCHS}
    t0 = time.time()
    for lr in LRS:
        row = []
        for a in ARCHS:
            b = max(train(a, lr, s)[0] for s in range(SEEDS))
            res[a][lr] = b
            row.append(f'{b:>10.3f}')
        print(f'{lr:>8.0e}  ' + '  '.join(row), flush=True)
    print(f'\n{time.time()-t0:.0f}s\nREGION WIDTH  (rates clearing 0.40, chance is 0.250)')
    for a in ARCHS:
        ok = [lr for lr, b in res[a].items() if b > 0.40]
        if not ok:
            print(f'  {a:<8} no rate cleared the bar'); continue
        bestlr = max(res[a], key=lambda l: res[a][l])
        print(f'  {a:<8} {len(ok)}/{len(LRS)} rates   span {min(ok):.0e}-{max(ok):.0e}   '
              f'best {bestlr:.0e}   headroom above best {max(ok)/bestlr:.1f}x')
    print('\n  P1/P2 hold if saryu has fewer working rates and less headroom above its best.')
    print('  FALSIFIED if the widths are comparable: stability margin explains neither puzzle.')


if __name__ == '__main__':
    main()
