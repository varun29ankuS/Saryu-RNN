"""Do the two fixes compose? Capacity and persistence, together, for the first time.

WHY THIS IS THE NEXT THING. nh_sweep_shortgap.txt separated two failure modes and nothing moved
either for weeks:

    pairs \\ gap        0        4       64
        1          1.000    1.000    0.012      <- PERSISTENCE: one fact dies over distance
        4          0.199    0.113    0.012      <- CAPACITY: four facts die at any distance

Both now have a fix, and each was measured in the regime where the OTHER does not bind:
    the matrix memory took 4 pairs at gap 4 from 0.281 to 0.930/0.938   (memory_on.txt)
    the gate lower bound took 1 pair at gap 64 from 1/3 solved to 6/7   (gate_lower_bound_READ)

Neither has been run where both bind. If they compose, this is a working model. If they interfere,
every conclusion drawn from the two separately needs qualifying.

THE DESIGN. 4 pairs at gap 16 -- four facts, and a distance well past the 4-8 tokens of retention
the gate collapses to without the bound (gate_retention.txt measured 25 -> 8 -> 4). Depth 4,
because the gate bound is a per-LAYER ladder and needs layers to differentiate. A 2x2 factorial so
the interaction is visible rather than inferred:

    memory=F gate_lb=F    the control; expect the old ~0.1-0.2
    memory=T gate_lb=F    capacity fixed only
    memory=F gate_lb=T    persistence fixed only
    memory=T gate_lb=T    both

AND A TRANSFORMER BASELINE, because this project has now produced two VOID experiments by running
a factorial on a task nothing could learn -- state_volume at gap 24, and the conjunctive task where
a matched transformer also sat at chance. Without the baseline, "the fixes do not compose" and "the
task is too hard here" are indistinguishable. That mistake is not being made a third time.

REGISTERED PREDICTIONS, before running:
  P1  the transformer solves it (> 0.8). If not, the task is out of reach at this budget and the
      whole factorial is void -- report that and stop, do not read the Saryu rows.
  P2  each single fix beats the control, but neither reaches the transformer: capacity alone still
      loses the facts over 16 tokens, persistence alone still cannot hold four of them.
  P3  THE ONE THAT MATTERS. both together beats both singles by more than they differ from the
      control -- i.e. a positive interaction, not a sum. That is what "compose" means.

FALSIFIER for P3: both-on is no better than the better single fix. Then the two address the same
underlying limit rather than two separate ones, the capacity/persistence decomposition in
nh_sweep_shortgap is wrong, and one of the fixes is redundant.

    python evidence/both_fixes.py
Environment: STEPS, SEEDS, LR, GAP, PAIRS, NL, THREADS.
"""
from __future__ import annotations

import os
import sys
import time

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                   # noqa: E402
from saryu.metrics import Run                                       # noqa: E402

STEPS = int(os.environ.get('STEPS', 1500))
SEEDS = int(os.environ.get('SEEDS', 2))
LR = float(os.environ.get('LR', 1e-3))
GAP = int(os.environ.get('GAP', 16))
PAIRS = int(os.environ.get('PAIRS', 4))
NL = int(os.environ.get('NL', 4))
D, H, NH, BS, NENT = 128, 8, 2, 32, 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 4)))
# nn.TransformerEncoder prints a nested-tensor warning carrying an absolute path,
# which is how absolute paths have twice reached results files and blocked a push.
warnings.filterwarnings('ignore', category=UserWarning)


def make(rng, n):
    e = rng.choice(NENT, size=2 * n, replace=False)
    ks, vs = e[:n], e[n:]
    rest = np.setdiff1d(np.arange(NENT), e)
    s = []
    for a, b in zip(ks, vs):
        s += [int(a), int(b)]
    s += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, n))
    s += [SEP, int(ks[i])]
    return s, int(vs[i])


def batch(rng, bs):
    o = [make(rng, PAIRS) for _ in range(bs)]
    return torch.tensor([a for a, _ in o]), torch.tensor([b for _, b in o])


class TF(nn.Module):
    """The baseline. Same block budget, so 'the task is learnable here' is an honest statement."""

    def __init__(self, vocab, d, nl, nhead=4, maxlen=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(maxlen, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, 2 * d, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl)
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        L = x.shape[1]
        h = self.emb(x) + self.pos[:L]
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        return self.head(self.lnf(self.enc(h, mask=mask)))


def train(arm, seed):
    torch.manual_seed(seed)
    if arm == 'transformer':
        m = TF(NENT + 2, D, NL)
    else:
        mem, lb = arm
        m = SaryuV3LM(NENT + 2, D, NL, nh=NH, H=H, memory=mem, gate_lb=lb)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(1000 + seed)
    tag = 'tf' if arm == 'transformer' else f'm{int(arm[0])}g{int(arm[1])}'
    log = Run(f'bf-{tag}-p{PAIRS}-g{GAP}-s{seed}',
              config=dict(arm=f'both_fixes/{tag}', pairs=PAIRS, gap=GAP, nl=NL, params=npar,
                          steps=STEPS, seed=seed, lr=LR, chance_top1=round(1 / PAIRS, 4)))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 250 == 0:
            ev = np.random.default_rng(7); a = 0.
            m.eval()
            with torch.no_grad():
                for _ in range(4):
                    x2, y2 = batch(ev, BS)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train(); best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    return best, npar


def main():
    L = 2 * PAIRS + GAP + 2
    print(f'BOTH FIXES  {PAIRS} pairs, gap {GAP} (sequence {L}), d={D} H={H} nh={NH} nl={NL}, '
          f'{STEPS} steps, {SEEDS} seeds, chance {1/PAIRS:.3f}')
    print('memory alone at gap 4: 0.281 -> 0.930.  gate bound alone at 1 pair gap 64: 1/3 -> 6/7.')
    print('neither has been run where BOTH bind.\n')
    arms = [('transformer'), (False, False), (True, False), (False, True), (True, True)]
    names = {'transformer': 'transformer (baseline)', (False, False): 'neither (control)',
             (True, False): 'memory only', (False, True): 'gate bound only',
             (True, True): 'BOTH'}
    print(f'{"arm":>24} {"params":>10}  best per seed')
    print('-' * 60)
    t0 = time.time()
    res = {}
    for arm in arms:
        r = [train(arm, s) for s in range(SEEDS)]
        res[arm] = [x for x, _ in r]
        print(f'{names[arm]:>24} {r[0][1]:>10,}  ' + ', '.join(f'{x:.3f}' for x, _ in r),
              flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    tf = float(np.mean(res['transformer']))
    print('READ')
    if tf < 0.8:
        print(f'  P1 FAILS: the transformer reaches only {tf:.3f}. The task is out of reach at')
        print('  this budget, the factorial is VOID, and the Saryu rows must not be read.')
        return
    c = float(np.mean(res[(False, False)]))
    mo = float(np.mean(res[(True, False)]))
    go = float(np.mean(res[(False, True)]))
    bo = float(np.mean(res[(True, True)]))
    print(f'  transformer {tf:.3f}  control {c:.3f}  memory {mo:.3f}  gate {go:.3f}  both {bo:.3f}')
    print(f'  interaction = both - (best single) = {bo - max(mo, go):+.3f}')
    print('  P3 holds if that is clearly positive -- the fixes address two different limits.')
    print('  P3 FALSIFIED if both-on is no better than the better single: they address the SAME')
    print('     limit, the capacity/persistence split is wrong, and one fix is redundant.')


if __name__ == '__main__':
    main()
