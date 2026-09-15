"""Did the read-loop eps guard cost us accuracy? One variable, isolated.

Between glimpse run 1 and run 2 TWO things changed: the query-binding fix (357b09a) and the
read-loop eps guard (34eec90, F.normalize eps 1e-12 -> 1e-4). Content-only fell 1.000 -> 0.793 on
mqar and 0.277 -> 0.078 on chain, and content-only CANNOT have been touched by the query fix:
with use_state and use_time both False the binding is skipped, and the extra F.normalize is
idempotent on an already-normalised vector. So the eps guard is the only live candidate.

WHY THIS MATTERS BEYOND ONE NUMBER. I committed the guard describing it as "provably inert at
healthy magnitudes", on the evidence that trace_logic 8a still reported 0.00e+00 and every
invariant returned byte-identical values. That evidence was from RANDOM INIT IN float64. It says
nothing about a trained trajectory, where ||S r|| does get small - and there the two settings
genuinely differ:

    F.normalize is x / max(||x||, eps), a CLAMP not an addend.
    ||r|| = 1e-6, eps = 1e-12  ->  output norm 1.0    (a tiny vector is renormalised to full size)
    ||r|| = 1e-6, eps = 1e-4   ->  output norm 0.01   (a tiny vector stays small)

Which is correct is a real question - amplifying a collapsed chase to unit norm is amplifying
noise - but it is a BEHAVIOURAL CHANGE, and I asserted it away instead of measuring it.

This runs content-only only, the arm where nothing else differs, at both eps values and two seeds.

usage: cd experiments && STEPS=600 python eps_ablation.py
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

import saryu_block
from glimpse import Net, lag_cos
from recall_tasks import vocab
from shift_arms import batch

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))


def run(eps, task, steps, d, nl, nent, hd, seed, **cell):
    """Patch the eps the block's read loop uses, then train content-only."""
    orig = F.normalize

    def patched(x, dim=-1, eps_=None, **kw):
        return orig(x, dim=dim, eps=eps, **{k: v for k, v in kw.items() if k != 'eps'})

    saryu_block.F.normalize = patched
    try:
        torch.manual_seed(seed)
        _, _, _, V = vocab(nent)
        net = Net(V, d, nl, head_dim=hd, conv=True, n_read=2, tap_init='uniform',
                  bias='l2', use_state=False, use_time=False)
        opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=0.1)
        rng = np.random.default_rng(seed)
        losses = []
        for _ in range(steps):
            x, y = batch(task, 32, rng, nent, **cell)
            loss = F.cross_entropy(net(x), y)
            if not torch.isfinite(loss):
                return dict(acc=float('nan'), note='NON-FINITE')
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            losses.append(float(loss))
        net.eval()
        rng2 = np.random.default_rng(9999)
        corr = []
        with torch.no_grad():
            for _ in range(4):
                x, y = batch(task, 64, rng2, nent, **cell)
                corr += (net(x).argmax(-1) == y).tolist()
        return dict(acc=float(np.mean(corr)), loss1=losses[-1], lag=lag_cos(net), note='')
    finally:
        saryu_block.F.normalize = orig


if __name__ == '__main__':
    STEPS = int(os.environ.get('STEPS', 600))
    D, NL, NENT, HD = 64, 2, 32, 32
    CELLS = [('mqar npairs=2', 'mqar', dict(npairs=2, seqlen=32)),
             ('chain d2 ratio=1.0', 'chain', dict(depth=2, ratio=1.0, seqlen=48))]

    print('READ-LOOP eps ABLATION  content-only, d=%d nl=%d steps=%d, 2 seeds' % (D, NL, STEPS))
    print('the ONLY arm where the query-binding fix is provably a no-op, so eps is isolated\n')
    for cname, task, cell in CELLS:
        print('=== %s ===' % cname, flush=True)
        print('  %-12s %-8s %-8s %-9s %s' % ('eps', 'seed0', 'seed1', 'mean', 'lag_cos(s0)'))
        for eps in (1e-12, 1e-4):
            accs, lag0 = [], None
            for sd in (0, 1):
                r = run(eps, task, STEPS, D, NL, NENT, HD, sd, **cell)
                accs.append(r['acc'])
                if sd == 0:
                    lag0 = r.get('lag')
            print('  %-12.0e %-8.3f %-8.3f %-9.3f %s'
                  % (eps, accs[0], accs[1], float(np.mean(accs)),
                     '-' if lag0 is None else '%.3f' % lag0), flush=True)
        print(flush=True)
    print('READ: if 1e-12 beats 1e-4 materially, the guard bought fp16 safety at a real cost and')
    print('the right answer is a smaller eps, or clamping only when the norm is truly degenerate.')
    print('If they tie, the glimpse regression is seed noise and content-only never moved.')
