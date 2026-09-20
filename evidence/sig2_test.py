"""Does a level-2 term give the model the memory that every level-1 fix could not?

THE CLAIM BEING TESTED (docs/memory_geometry.md, addendum). A key-value association is an ordered
pair, so binding is a LEVEL-2 object in the sense of Chen's path signature: level 1 is increments,
level 2 is the ordered pairwise interaction. Saryu's state is level 1 -- a gated, transported sum of
writes -- and the tensor order of a state bounds the order of interaction it can hold. If that is
right, then no level-1 fix can bind, which is what we found:

    raising n_h     0.113 -> 0.004 at gap 4, monotonically worse   (nh_sweep_shortgap.txt)
    carving         0.117 at gap 4, ceiling 0.430 by construction  (carve_test.txt)
    carve+bind      0.184 at gap 4, 0.020 at gap 64                (carve_test.txt)

and a level-2 term should lift it, because by construction the level-2 stores are perfect at every
gap and pair count tested (outer-equiv and bivector, both 1.000, memory_mechanisms.txt).

THE ARM. sig2 adds per head A_t = alpha_t A_{t-1} + p_t k_t^T, read by contraction r_t = A_t q_t,
with its own per-head decay (sigmoid(5) = 0.993, ~150 tokens) because sharing the level-1 gate is
what capped carving at gap 64. Chen's identity makes this associative, so the recurrent and
chunk-parallel forms exist; forward() evaluates the mathematically identical quadratic form, checked
equal to the recurrence at 3.7e-09. sig2-anti keeps only the antisymmetric half -- the Levy area,
the bivector.

THE CONTROL THAT MATTERS. sig2 has 518,306 parameters against the base's 387,218, so a gain could
just be capacity. base-wide is the base at d=152 with 540,056 parameters -- 4% MORE than sig2 -- so
if the level-2 arm wins it is not winning on size.

REGISTERED PREDICTIONS, before running. Chance = 1/64 = 0.016.
  P1  sig2 at gap 4  > 0.40   (base 0.113, best level-1 fix 0.184)
  P2  sig2 at gap 64 > 0.30   (base 0.012; persistence, not just capacity)
  P3  base-wide does NOT reach sig2 at gap 4, i.e. the gain is the level-2 structure and not the
      extra parameters.

FALSIFIER: if base-wide matches sig2, the signature reading is not what is doing the work here and
the result is a parameter-count artefact.

    python evidence/sig2_test.py
Environment: ARMS, PAIRS, GAPS, STEPS, LR, SEEDS, THREADS.
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

NL = int(os.environ.get('NL', 2))
PAIRS = int(os.environ.get('PAIRS', 4))
GAPS = [int(x) for x in os.environ.get('GAPS', '4,64').split(',')]
STEPS = int(os.environ.get('STEPS', 1500))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0').split(',')]
torch.set_num_threads(int(os.environ.get('THREADS', 8)))

ARMS = {
    'base':       (128, dict()),
    'base-wide':  (152, dict()),                       # 540,056 params: 4% MORE than sig2
    'sig2':       (128, dict(sig2=True)),
    'sig2-anti':  (128, dict(sig2=True, sig2_anti=True)),
}
PICK = [a for a in os.environ.get('ARMS', ','.join(ARMS)).split(',') if a in ARMS]


def run(arm, gap, seed):
    d, kw = ARMS[arm]
    seqlen = 2 * PAIRS + gap + 2
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT + 2, d, NL, **kw)
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
    return hit / n


def main():
    print('DOES A LEVEL-2 TERM GIVE THE MODEL A MEMORY?')
    print(f'MQAR {PAIRS} pairs, {NL} layers, lr={LR:g}, {STEPS} steps, seeds {SEEDS}, '
          f'chance {1/NENT:.3f}\n')
    for a in PICK:
        d, kw = ARMS[a]
        n = sum(p.numel() for p in SaryuV3LM(NENT + 2, d, NL, **kw).parameters())
        print(f'  {a:<11} d={d:<4} params {n:>9,}')
    print()
    hdr = f'{"arm":<12}' + ''.join(f'{"gap " + str(g):>10}' for g in GAPS)
    print(hdr); print('-' * len(hdr))
    res = {}
    for arm in PICK:
        row = [sum(run(arm, g, sd) for sd in SEEDS) / len(SEEDS) for g in GAPS]
        res[arm] = row
        print(f'{arm:<12}' + ''.join(f'{v:>10.3f}' for v in row), flush=True)

    print('\n  for reference, same protocol: base 0.113 / 0.012, carve+bind 0.184 / 0.020 '
          '(carve_test.txt)')
    if 'sig2' in res:
        print('\nREGISTERED PREDICTIONS')
        if 4 in GAPS:
            v = res['sig2'][GAPS.index(4)]
            print(f'  P1 sig2 at gap 4 > 0.40:  {v:.3f} -> {"HOLDS" if v > 0.40 else "FAILS"}')
        if 64 in GAPS:
            v = res['sig2'][GAPS.index(64)]
            print(f'  P2 sig2 at gap 64 > 0.30: {v:.3f} -> {"HOLDS" if v > 0.30 else "FAILS"}')
        if 'base-wide' in res and 4 in GAPS:
            i = GAPS.index(4)
            bw, s2 = res['base-wide'][i], res['sig2'][i]
            print(f'  P3 base-wide (4% MORE params) does not reach sig2 at gap 4: '
                  f'{bw:.3f} vs {s2:.3f} -> {"HOLDS" if bw < s2 else "FAILS"}')


if __name__ == '__main__':
    main()
