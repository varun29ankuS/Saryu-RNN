"""Does this architecture get WORSE as it gets wider? A prediction this repo made and never tested.

THE PREDICTION, from results/trained_geometry.txt item 2, written before this experiment existed:

    "DESIGN RULE, and it runs against normal practice: dh grows with model width while nh stays
     pinned at 2, so LARGER MODELS GET MONOTONICALLY WORSE at directed relation. The quantity to
     hold fixed is the RATIO nh/dh. Every nh sweep in this project varied nh with dh held constant
     and so never saw the ratio."

Every result in this repository is at d = 128, two to four layers, on synthetic tasks. If that rule
is right, none of them survives scale -- and the shipped 25M checkpoint already sits at
tr/dh = 0.957, which composition.txt places in the FAILING band. That is a serious claim about our
own model that has never been checked, and it matters more than matching anyone's layer count.

THE QUANTITY. For a product of nh Householder reflections in a dh-dimensional head, the expected
overlap of a vector with its image is the normalised trace

    tr(R)/dh = 1 - 2*nh/dh

composition.txt measured where chaining breaks on exactly this number: at 0.984 chaining is
indistinguishable from commutative binding (0.496 at k=2), at 0.938 it is partial (0.716), and by
0.750 it matches a permutation. So the bands are known in advance.

THE SWEEP. H is held at 8 and the width d varies, which is the faithful version of the claim --
dh = d/H grows with model width, exactly as it does when anyone scales a model.

    d     dh    nh=2 (the shipped default)      nh = dh/8 (ratio held fixed)
    128   16    tr/dh = 0.750  working          nh=2   0.750
    256   32    tr/dh = 0.875  partial          nh=4   0.750
    384   48    tr/dh = 0.917  partial          nh=6   0.750
    512   64    tr/dh = 0.938  FAILING          nh=8   0.750

d=512 is included because 0.938 is precisely where composition.txt measured chaining to be only
partial (0.716) -- without it the sweep never enters the band the prediction is about.

At d = 128 the two arms are the same model, which is the internal control: any difference between
them at that width is seed noise and sets the scale for reading the rest.

THE TASK is the S_3 word problem -- three slots with FIXED contents, six generators shuffling them,
answer what is in slot j. Pure non-commutative state tracking, nothing to store. Saryu solves it at
1.000 at d=128 (conjunctive.txt), and a matched transformer got 0.562/0.859, so it is also the one
task this architecture demonstrably wins. Chance is 1/3.

REGISTERED PREDICTIONS, before running:
  P1  the nh=2 arm DEGRADES monotonically with width: 1.000 at d=128, lower at 256, lower again
      at 384, lowest at 512 where tr/dh = 0.938 enters the band composition.txt measured. This is the repo's own design rule and the reason to run this.
  P2  the ratio-fixed arm holds at ~1.000 across all three widths, because tr/dh is constant at
      0.750 by construction.
  P3  the degradation tracks tr/dh rather than d as such -- i.e. d=512 with nh=8 (0.750) beats
      d=256 with nh=2 (0.875), even though it is twice the width. That is the control that says
      the trace is the operative quantity and not width per se.

FALSIFIER, and it is the outcome that would retract a standing claim: the nh=2 arm holds up at
d=512, where tr/dh = 0.938 is inside the band composition.txt measured as only partially chaining.
Then the trace-law scaling rule is wrong, "larger models get monotonically worse" comes out of
trained_geometry.txt, and the worry about the 25M checkpoint sitting at 0.957 goes with it.

    python evidence/width_scaling.py
Environment: STEPS, SEEDS, LR, WIDTHS, THREADS.
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
from saryu.model import SaryuV3LM                                   # noqa: E402
from saryu.metrics import Run                                       # noqa: E402
from conjunctive import batch, VOCAB, NGEN                          # noqa: E402

STEPS = int(os.environ.get('STEPS', 2000))
SEEDS = int(os.environ.get('SEEDS', 2))
LR = float(os.environ.get('LR', 1e-3))
H, NL, BS = 8, 2, 32
WIDTHS = [int(w) for w in os.environ.get('WIDTHS', '128,256,384,512').split(',')]
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


def trace_ratio(nh, dh):
    return 1.0 - 2.0 * nh / dh


def band(t):
    return 'working' if t <= 0.80 else ('partial' if t < 0.93 else 'FAILING')


def train(d, nh, seed):
    dh = d // H
    torch.manual_seed(seed)
    m = SaryuV3LM(VOCAB, d, NL, nh=nh, H=H)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(1000 + seed)
    log = Run(f'ws-d{d}-nh{nh}-s{seed}',
              config=dict(arm=f'width_scaling/d{d}-nh{nh}', d=d, H=H, dh=dh, nh=nh, nl=NL,
                          trace_ratio=round(trace_ratio(nh, dh), 4), steps=STEPS, seed=seed,
                          params=sum(p.numel() for p in m.parameters()), chance_top1=0.3333))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS, NGEN, True)                # fixed values -> pure S_3 tracking
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 250 == 0:
            ev = np.random.default_rng(7); a = 0.
            m.eval()
            with torch.no_grad():
                for _ in range(4):
                    x2, y2 = batch(ev, BS, NGEN, True)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train(); best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    return best, sum(p.numel() for p in m.parameters())


def main():
    print(f'WIDTH SCALING  S_3 word problem, H={H} nl={NL}, {STEPS} steps, {SEEDS} seeds, '
          f'chance 0.333')
    print('trained_geometry item 2: "larger models get monotonically worse ... hold nh/dh fixed"')
    print('composition.txt bands on tr/dh: <=0.750 working, 0.938 partial, 0.984 gone\n')
    print(f'{"d":>5} {"dh":>4} {"nh":>4} {"tr/dh":>7} {"band":>8} {"params":>10}  best per seed')
    print('-' * 72)
    t0 = time.time()
    rows = []
    for d in WIDTHS:
        dh = d // H
        arms = [2] if dh // 8 == 2 else [2, dh // 8]      # at d=128 both arms are the same model
        for nh in arms:
            out = [train(d, nh, s) for s in range(SEEDS)]
            r = [x[0] for x in out]
            npar = out[0][1]
            tr = trace_ratio(nh, dh)
            rows.append((d, dh, nh, tr, r))
            print(f'{d:>5} {dh:>4} {nh:>4} {tr:>7.3f} {band(tr):>8} {npar:>10}  '
                  + ', '.join(f'{x:.3f}' for x in r), flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ')
    print('  P1 holds if the nh=2 row falls as d rises.')
    print('  P2 holds if the ratio-fixed rows stay near 1.000 at every width.')
    print('  P3 holds if d=512/nh=8 (tr 0.750) beats d=256/nh=2 (tr 0.875) -- the wider model')
    print('     winning is what says the TRACE is operative and not width itself.')
    print('  Step budget is not a confound: d=128 reaches 0.984 by step 250 and 1.000 by 500,')
    print('  so every arm here gets at least 3x what the easy config needed.')
    print('  FALSIFIED if nh=2 holds at d=512: the scaling rule in trained_geometry.txt is wrong')
    print('     and "larger models get monotonically worse" has to be retracted.')


if __name__ == '__main__':
    main()
