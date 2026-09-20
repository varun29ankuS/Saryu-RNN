"""Does carving a memory subspace give the trained model a memory?

THE PLAN THIS TESTS (docs/memory_geometry.md, evidence/results/memory_mechanisms.txt). By
construction, confining the transports to a tracking subspace leaves the memory subspace fixed, so
the unbind commutes with the transport by disjoint support -- and it was the only fix that needed no
oracle. There it was flat in gap: 4 pairs at k=32 scored 0.970 / 0.960 / 0.980 / 0.975 across gaps
0, 4, 16, 64, where the uncarved store fell to chance.

THE ARMS. Each head's d_h = 16 splits 8 tracking / 8 memory.
  base              what ships. Reflections roam the whole head; the write is linear in z.
  carve             reflection 0 is confined to the tracking dims, so it is the IDENTITY on the
                    memory dims. Reflection 1 is confined to the memory dims and made inert as a
                    transport. Nothing else changes.
  carve+bind        as carve, and the write is reflected about the token's memory direction before
                    being added. This is the only multiplicative (key x value) interaction in the
                    layer; without it the write is linear in z and can superpose but not associate.

Parameter count is identical in all three, and the exact kernel is untouched (the write is built
before the scan, and a masked u is still a unit vector).

REGISTERED PREDICTIONS, before running. Baseline numbers from nh_sweep_shortgap.txt and
nh_sweep_1pair.txt, chance = 1/64 = 0.016:

                          base      predicted for carve+bind
    4 pairs, gap 4       0.113      > 0.20   (scrambling was the constraint)
    4 pairs, gap 64      0.012      > 0.10   (the collapse with distance stops)

  P1  carve+bind beats base at gap 4 by at least 0.09.
  P2  carve+bind at gap 64 stays above 0.10, i.e. it does not collapse to chance.
  P3  carve alone helps less than carve+bind, since carving protects the memory but supplies no
      operation that binds anything into it.

FALSIFIER: if carve+bind lands within noise of base at gap 4, then scrambling is not the binding
constraint and the write rule is -- which would send the work to the matrix memory instead.

    python evidence/carve_test.py
Environment: ARMS, PAIRS, GAPS, STEPS, D, NL, CARVE, LR, SEEDS, THREADS.
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
from nh_sweep import batch, NENT                                      # noqa: E402

D = int(os.environ.get('D', 128))
NL = int(os.environ.get('NL', 2))
CARVE = int(os.environ.get('CARVE', 8))
PAIRS = int(os.environ.get('PAIRS', 4))
GAPS = [int(x) for x in os.environ.get('GAPS', '4,16,64').split(',')]
STEPS = int(os.environ.get('STEPS', 1500))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0').split(',')]
torch.set_num_threads(int(os.environ.get('THREADS', 8)))

ARMS = {
    'base': dict(),
    'carve': dict(carve=CARVE),
    'carve+bind': dict(carve=CARVE, bind_write=True),
}
PICK = [a for a in os.environ.get('ARMS', 'base,carve,carve+bind').split(',') if a in ARMS]


def run(arm, gap, seed):
    seqlen = 2 * PAIRS + gap + 2
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT + 2, D, NL, **ARMS[arm])
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
    rng = np.random.default_rng(1000 + seed)
    t0 = time.time()
    for s in range(STEPS):
        x, y = batch(rng, PAIRS, gap, seqlen, BS)
        loss = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if (s + 1) % max(1, STEPS // 3) == 0:
            el = time.time() - t0
            print(f'      {arm} gap{gap} {s+1}/{STEPS} loss {float(loss):.3f} '
                  f'{el:.0f}s, {el/(s+1)*(STEPS-s-1):.0f}s left', flush=True)
    m.eval()
    ev = np.random.default_rng(99)
    hit = n = 0
    with torch.no_grad():
        for _ in range(8):
            x, y = batch(ev, PAIRS, gap, seqlen, BS)
            hit += int((m(x)[:, -1].argmax(-1) == y).sum()); n += len(y)
    return hit / n, time.time() - t0


def main():
    print('DOES CARVING A MEMORY SUBSPACE GIVE THE MODEL A MEMORY?')
    print(f'MQAR {PAIRS} pairs, d={D}, {NL} layers, d_h={D//8}, carve={CARVE} '
          f'({CARVE} track / {D//8-CARVE} memory), lr={LR:g}, {STEPS} steps, seeds {SEEDS}')
    print(f'chance = 1/{NENT} = {1/NENT:.3f}\n')
    hdr = f'{"arm":<12}' + ''.join(f'{"gap " + str(g):>10}' for g in GAPS)
    print(hdr); print('-' * len(hdr))
    res = {}
    for arm in PICK:
        row = []
        for gap in GAPS:
            accs = [run(arm, gap, sd)[0] for sd in SEEDS]
            row.append(sum(accs) / len(accs))
        res[arm] = row
        print(f'{arm:<12}' + ''.join(f'{v:>10.3f}' for v in row), flush=True)

    if 'base' in res and 'carve+bind' in res:
        print('\nREGISTERED PREDICTIONS')
        i4 = GAPS.index(4) if 4 in GAPS else 0
        d4 = res['carve+bind'][i4] - res['base'][i4]
        p1 = d4 >= 0.09
        print(f'  P1 carve+bind beats base at gap {GAPS[i4]} by >= 0.09: '
              f'{res["base"][i4]:.3f} -> {res["carve+bind"][i4]:.3f} ({d4:+.3f}) '
              f'-> {"HOLDS" if p1 else "FAILS"}')
        if 64 in GAPS:
            v = res['carve+bind'][GAPS.index(64)]
            print(f'  P2 carve+bind at gap 64 stays above 0.10: {v:.3f} '
                  f'-> {"HOLDS" if v > 0.10 else "FAILS"}')
        if 'carve' in res:
            print(f'  P3 carve alone helps less than carve+bind at gap {GAPS[i4]}: '
                  f'{res["carve"][i4]:.3f} vs {res["carve+bind"][i4]:.3f} -> '
                  f'{"HOLDS" if res["carve"][i4] < res["carve+bind"][i4] else "FAILS"}')


if __name__ == '__main__':
    main()
