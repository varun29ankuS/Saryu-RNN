"""A task that needs BOTH paths, because every task in this repository needs only one.

THE GAP THIS CLOSES. state_volume_gap4.txt established causally that the fact lives in the matrix
-- ablating the memory read at the query position costs 0.84 while ablating the transport there
costs 0.02. But it also showed that on MQAR the transport simply GOES QUIET: retention saturates
at ~10,000 tokens on a sequence of length 14 and the path contributes almost nothing. That is
correct, because MQAR has no order structure at all. It means the task can demonstrate takeover
and cannot demonstrate complementarity.

Group word problems have the opposite defect: they need order and hold no facts. So this project
has never once run a task where both are required, and "the transport tracks while the memory
stores" has been a design intention with one half verified and the other half assumed.

THE TASK. Three slots are filled with values, then a sequence of S_3 generators physically shuffles
the slots, then the query asks what is in slot j.

    [SLOT0 v_a SLOT1 v_b SLOT2 v_c]  [g g g ...]  [SEP IDX_j]   ->  the value now in slot j

  needs STORAGE   three distinct values must survive to the end; a state that holds one fact
                  cannot answer for an arbitrary j
  needs TRACKING  the generators compose in order and S_3 is non-abelian, so a commutative
                  transport cannot get the final permutation right -- this project's own ceiling
                  result puts a diagonal recurrence at a provable 0.385 on S_3

Chance is 1/3. The two sub-tasks are run as controls, so the conjunction is not confounded with
plain difficulty:

    recall-only   the same thing with ZERO generators -- storage alone suffices
    track-only    three slots with FIXED contents -- nothing to store, the composed
                  permutation alone decides the answer. This is the S_3 word problem.

REGISTERED PREDICTIONS, before running:
  P1  memory=True beats memory=False on the conjunctive task. (Storage is necessary.)
  P2  THE ONE THAT MATTERS. On the conjunctive task, ablating the TRANSPORT at the query position
      costs substantially more than the 0.02 it cost on MQAR. That is what complementarity means
      operationally: both paths carry part of the answer at the moment it is produced.
  P3  On recall-only, the transport ablation stays cheap, reproducing the MQAR result. This is the
      control that says P2's effect comes from the added order structure and not from the task
      being harder.

FALSIFIER for P2: the transport ablation stays near zero on the conjunctive task while accuracy is
high. Then the memory is doing the tracking too, the two paths are not dividing labour, and the
design intention in model.py's comments is wrong -- what we have is one general mechanism plus a
redundant one, which would be worth knowing before anything is built on the split.

    python evidence/conjunctive.py
Environment: STEPS, SEEDS, LR, NGEN, THREADS.
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                    # noqa: E402
from saryu.metrics import Run                                       # noqa: E402

STEPS = int(os.environ.get('STEPS', 4000))
SEEDS = int(os.environ.get('SEEDS', 2))
LR = float(os.environ.get('LR', 1e-3))
NGEN = int(os.environ.get('NGEN', 6))       # generators applied, so the composition is non-trivial
D, H, NH, NL, BS = 128, 8, 2, 2, 32
NVAL = 48                                   # value tokens 0..47
SLOT = [NVAL, NVAL + 1, NVAL + 2]           # slot markers
GENS = [NVAL + 3, NVAL + 4]                 # two generators of S_3: a transposition and a 3-cycle
SEP = NVAL + 5
IDX = [NVAL + 6, NVAL + 7, NVAL + 8]        # which slot is being asked for
VOCAB = NVAL + 9
# the two permutations, as index maps on the three slots
PERM = {GENS[0]: (1, 0, 2), GENS[1]: (1, 2, 0)}     # swap(0,1) and the 3-cycle
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


def make(rng, ngen, fixed_vals=False):
    """Always THREE slots. fixed_vals makes the contents constant across examples, so there is
    nothing to store and only the composed permutation decides the answer -- that is the
    track-only control, and it is the S_3 word problem.

    The first version of this control used ONE slot instead, which made the generators inert and
    turned it into a copy task. Caught by sanity-checking the generator before training on it."""
    if fixed_vals:
        vals = [0, 1, 2]                      # constant, so storage cannot help
    else:
        vals = [int(v) for v in rng.choice(NVAL, size=3, replace=False)]
    s = []
    for i in range(3):
        s += [SLOT[i], vals[i]]
    slots = list(vals)
    for _ in range(ngen):
        g = int(rng.choice(GENS))
        s.append(g)
        p = PERM[g]
        slots = [slots[p[0]], slots[p[1]], slots[p[2]]]
    j = int(rng.integers(0, 3))
    s += [SEP, IDX[j]]
    return s, slots[j]


def batch(rng, bs, ngen, fixed_vals=False):
    o = [make(rng, ngen, fixed_vals) for _ in range(bs)]
    return torch.tensor([a for a, _ in o]), torch.tensor([b for _, b in o])


@torch.no_grad()
def ablate(m, ngen, fixed, memory):
    """Zero one path AT THE QUERY POSITION ONLY and re-score. Same instrument as state_volume."""
    m.eval()
    hooks, mode = [], {'kill': None}

    def mk(which):
        def f(_mod, _in, out):
            if mode['kill'] == which:
                out = out.clone(); out[:, -1] = 0
            return out
        return f

    for blk in m.mix:
        hooks.append(blk.out.register_forward_hook(mk('transport')))
        if blk.memory:
            hooks.append(blk.m_out.register_forward_hook(mk('memory')))
    scores = {}
    for kill in (None, 'memory', 'transport'):
        if kill == 'memory' and not memory:
            continue
        mode['kill'] = kill
        ev = np.random.default_rng(7); a = 0.
        for _ in range(8):
            x2, y2 = batch(ev, BS, ngen, fixed)
            a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 8
        scores['full' if kill is None else kill] = a
    for h in hooks:
        h.remove()
    return scores


def train(task, memory, seed):
    ngen, fixed = task
    torch.manual_seed(seed)
    m = SaryuV3LM(VOCAB, D, NL, nh=NH, H=H, memory=memory)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(1000 + seed)
    name = {(NGEN, False): 'both', (0, False): 'recall', (NGEN, True): 'track'}[task]
    log = Run(f'conj-{name}-{"on" if memory else "off"}-s{seed}',
              config=dict(arm=f'conjunctive/{name}-{"on" if memory else "off"}', ngen=ngen,
                          fixed_vals=fixed, memory=memory, steps=STEPS, seed=seed, lr=LR,
                          params=sum(p.numel() for p in m.parameters()),
                          chance_top1=0.3333))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS, ngen, fixed)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 250 == 0:
            ev = np.random.default_rng(7); a = 0.
            m.eval()
            with torch.no_grad():
                for _ in range(4):
                    x2, y2 = batch(ev, BS, ngen, fixed)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train(); best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    return best, ablate(m, ngen, fixed, memory)


def main():
    tasks = [((NGEN, False), 'both    (store 3 + track S_3)'),
             ((0, False), 'recall  (store 3, no generators)'),
             ((NGEN, True), 'track   (fixed values, S_3 only)')]
    print(f'CONJUNCTIVE  {NGEN} generators, d={D} H={H} L={NL}, {STEPS} steps, {SEEDS} seeds')
    print('MQAR reference (state_volume_gap4): transport ablation cost 0.02 where memory cost 0.84')
    print(f'{"task":<34} {"mem":>4} {"chance":>7}  recall per seed')
    print('-' * 74)
    t0 = time.time()
    res = {}
    for task, label in tasks:
        for memory in (False, True):
            r = [train(task, memory, s) for s in range(SEEDS)]
            res[(label, memory)] = r
            print(f'{label:<34} {str(memory):>4} {0.333:>7.3f}  '
                  + ', '.join(f'{b:.3f}' for b, _ in r), flush=True)
    print(f'\nABLATION AT THE QUERY POSITION  (the complementarity test)')
    print(f'{"task":<34} {"full":>7} {"-memory":>8} {"-transport":>11}   transport cost')
    for task, label in tasks:
        for ab in (a for _, a in res[(label, True)]):
            tc = ab['full'] - ab['transport']
            print(f'{label:<34} {ab["full"]:>7.3f} {ab.get("memory", float("nan")):>8.3f} '
                  f'{ab["transport"]:>11.3f}   {tc:+.3f}')
    print(f'\n{time.time()-t0:.0f}s')
    print('READ')
    print('  P2 holds if the transport ablation costs substantially more on "both" than the 0.02')
    print('     it cost on MQAR, AND stays cheap on "recall" (P3, the control).')
    print('  P2 FALSIFIED if it stays near zero on "both" while accuracy is high -- then the')
    print('     memory is doing the tracking too and the two paths are not dividing labour.')


if __name__ == '__main__':
    main()
