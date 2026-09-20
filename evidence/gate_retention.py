"""Is the gap-64 failure a RETENTION failure, and is that retention a training artefact?

THE OBSERVATION THAT PROMPTED THIS. nh_sweep_shortgap.txt, at n_h = 2, d_h = 16:

    pairs \\ gap        0        4       64
        1          1.000    1.000    0.012
        4          0.199    0.113    0.012

Read the FIRST row. One single pair. Nothing to separate it from, no capacity pressure, no
interference -- and it evaporates by gap 64. That row is not about memory capacity at all. It is
retention, and retention in this architecture is the gate: h_t = (1-g) T h_{t-1} + g c_t, so the
state decays with rho = (1-g) and a head holds roughly 1/g tokens.

WHY THAT MIGHT BE A TRAINING FAILURE RATHER THAN AN ARCHITECTURAL ONE. trained_geometry.txt item 5
measured gate retention on the shipped 25M checkpoint and found 1/g reaching 86 and 132 tokens in
layer 2, having emerged unaided -- while three separate attempts to INSTALL a timescale ladder by
hand collapsed to 2-3 tokens within 1000 steps. So this architecture demonstrably learns retention
far longer than 64 tokens. Every MQAR run that hit the gap-64 wall was a 2-layer model at d_h = 16
trained for 1500 steps on synthetic data, and NOBODY EVER MEASURED 1/g IN ONE.

A PRIOR OF OURS THAT POINTS THE OTHER WAY, stated up front so it is not quietly ignored. The
freeze_gate_bias experiment (gate_timescales.txt, and the comment at model.py:238) found that
holding the gate biases fixed left effective context at ~256 characters, concluding "the write rule
is the constraint, not the gate." That was enwik8 at scale and is a different question from MQAR
retention in a small model, but it is evidence against this hypothesis and it is on the record.

ARMS. n_h = 2, H = 8, d = 128 so d_h = 16, matching nh_sweep_shortgap exactly.

    A  gap  0, L=2, 1500 steps    control: should reproduce ~1.000
    B  gap 64, L=2, 1500 steps    the failure: should reproduce ~0.012
    C  gap 64, L=2, 6000 steps    does longer training grow retention?
    D  gap 64, L=4, 1500 steps    does DEPTH grow it? trained_geometry saw the long heads at depth

REGISTERED PREDICTIONS, before running:
  P1  Arm B has 1/g well below 64 in every head, so the fact is overwritten before the query
      arrives. That explains 0.012 with no capacity argument at all.
  P2  Arm A succeeds with short retention too, because the answer is adjacent. Retention only
      binds once there is a gap.
  P3  If C or D grows max retention past 64 AND recall rises with it, the gap-64 wall is a
      training/scale failure and the retention axis reopens.

FALSIFIER, and it is the outcome that kills the hypothesis: if arm B ALREADY has retention above
64 tokens and still scores ~0.012, then the state is being held long enough and the gap failure is
something else. Retention is then not the bottleneck and this line is closed.

Reported per layer and per head, measured on real task batches rather than at initialisation.

    python evidence/gate_retention.py
Environment: STEPS, SEEDS, THREADS.
"""
from __future__ import annotations

import math
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

D = 128
H = 8
NH = 2
BS = 32
LR = 3e-4
NENT = 64
SEP = NENT
SEEDS = int(os.environ.get('SEEDS', 2))
torch.set_num_threads(int(os.environ.get('THREADS', 4)))

# A  gap 0 control, B  the failure, C  longer, D  deeper
ARMS = [
    ('A gap0  L2 1500', 0, 2, 1500),
    ('B gap64 L2 1500', 64, 2, 1500),
    ('C gap64 L2 6000', 64, 2, 6000),
    ('D gap64 L4 1500', 64, 4, 1500),
]
# ONLY=D re-runs the depth arm alone at more seeds. Its first result was 1.000 and 0.031 on two
# seeds, which is a lottery ticket, not a fix; the statistic is the SOLVE RATE over seeds.
ARMS = [a for a in ARMS if a[0][0] in os.environ.get('ONLY', 'ABCD')]


def make(rng, n, gap):
    ent = rng.choice(NENT, size=2 * n, replace=False)
    ks, vs = ent[:n], ent[n:]
    rest = np.setdiff1d(np.arange(NENT), ent)
    s = []
    for k, v in zip(ks, vs):
        s += [int(k), int(v)]
    if gap:
        s += [int(x) for x in rng.choice(rest, size=gap, replace=True)]
    i = int(rng.integers(0, n))
    s += [SEP, int(ks[i])]
    return s, int(vs[i])


def batch(rng, bs, n, gap):
    o = [make(rng, n, gap) for _ in range(bs)]
    return torch.tensor([a for a, _ in o]), torch.tensor([b for _, b in o])


@torch.no_grad()
def gate_retention(m, x):
    """Mean gate per head per layer on REAL task input, and the retention 1/g it implies.

    The gate is g = ((1 - cos(g_proj(z))) / 2).clamp(max=gate_cap) at model.py:349, so hooking
    g_proj and reapplying that transform measures exactly what the recurrence uses."""
    caught = {}
    hooks = []

    def mk(i, blk):
        def hook(_mod, _inp, out):
            g = ((1.0 - torch.cos(out)) / 2.0).clamp(max=blk.gate_cap)
            if blk.g_max is not None:
                g = torch.minimum(g, blk.g_max)
            caught[i] = g.mean(dim=(0, 1))                  # [H], averaged over batch and time
        return hook

    for i, blk in enumerate(m.mix):
        hooks.append(blk.g_proj.register_forward_hook(mk(i, blk)))
    m.eval()
    m(x)
    m.train()
    for h in hooks:
        h.remove()
    # retention in tokens is 1/g
    return {i: (1.0 / g.clamp(min=1e-6)).tolist() for i, g in sorted(caught.items())}


def run_arm(tag, gap, nl, steps, seed):
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT + 2, D, nl, nh=NH, H=H)
    opt = Muon(list(m.parameters()), lr=LR)
    rng = np.random.default_rng(1000 + seed)
    ev = np.random.default_rng(7)
    log = Run(f'gret-{tag.split()[0]}-g{gap}-L{nl}-s{seed}',
              config=dict(arm=f'gate_retention/{tag}', pairs=1, gap=gap, nl=nl, nh=NH, H=H, d=D,
                          dh=D // H, steps=steps, seed=seed, chance_top1=round(1 / NENT, 4)))
    best = 0.
    for s in range(steps):
        x, y = batch(rng, BS, 1, gap)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s == 0 or (s + 1) % 250 == 0:
            with torch.no_grad():
                m.eval()
                a = 0.
                for _ in range(4):
                    x2, y2 = batch(ev, BS, 1, gap)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
                m.train()
            best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    xr, _ = batch(ev, BS, 1, gap)
    ret = gate_retention(m, xr)
    log.done()
    return best, ret


def main():
    print(f'GATE RETENTION  1 pair, d={D} H={H} dh={D//H} nh={NH}, {SEEDS} seeds, chance '
          f'{1/NENT:.3f}')
    print('question: is the gap-64 wall the gate failing to hold, rather than a capacity limit?')
    print('nh_sweep_shortgap measured 1 pair at gap 0 -> 1.000 and at gap 64 -> 0.012\n')
    t0 = time.time()
    rows = []
    for tag, gap, nl, steps in ARMS:
        accs, rets = [], []
        for sd in range(SEEDS):
            a, r = run_arm(tag, gap, nl, steps, sd)
            accs.append(a); rets.append(r)
        # PER SEED, not averaged. The first version of this script printed the seed-MEAN of the
        # per-layer maximum, and arm D came back as "336 tokens" next to a recall of 0.516 that was
        # actually 1.000 and 0.031 -- one seed solving and one at chance. A mean over a bimodal
        # outcome is not a measurement of anything, and it hid which seed carried the retention.
        per_seed = [max(max(r[i]) for i in r) for r in rets]
        per_layer_seed = [{i: max(r[i]) for i in r} for r in rets]
        rows.append((tag, gap, accs, per_layer_seed, per_seed))
        print(f'{tag:<17} recall {np.mean(accs):.3f}   solved {sum(a > 0.8 for a in accs)}/{SEEDS}',
              flush=True)
        for sd, (a, pls, top) in enumerate(zip(accs, per_layer_seed, per_seed)):
            pl = '  '.join(f'L{i}:{v:.0f}' for i, v in sorted(pls.items()))
            print(f'{"":<17}   seed {sd}: recall {a:.3f}   max 1/g  {pl}   overall {top:.0f} tok',
                  flush=True)
    print(f'\n{time.time()-t0:.0f}s\n')
    print('READ, against the registered predictions')
    b = next(r for r in rows if r[0].startswith('B'))
    bmax = max(b[4])
    print(f'  P1 arm B retention {bmax:.0f} tokens (best seed) against a 64-token gap: '
          f'{"HOLDS -- overwritten before the query" if bmax < 64 else "FAILS"}')
    if bmax >= 64 and np.mean(b[2]) < 0.1:
        print('  FALSIFIER FIRED: the state is held long enough and recall still fails.')
        print('  Retention is NOT the bottleneck; this line is closed.')
    for r in rows:
        if r[0].startswith(('C', 'D')):
            solved = sum(a > 0.8 for a in r[2])
            print(f'  {r[0]}: solved {solved}/{SEEDS}, retention per seed '
                  + ', '.join(f'{x:.0f}' for x in r[4]) + ' tok')
    print('  solve RATE is the statistic here, not the mean -- the outcome is bimodal.')


if __name__ == '__main__':
    main()
