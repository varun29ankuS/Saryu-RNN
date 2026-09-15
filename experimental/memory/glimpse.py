"""A LIGHT CPU GLIMPSE of the full stack, trained. Small, cheap, and honest about what it isn't.

saryu_block asserts 9/9 invariants but every number in it is a HAND-SET tap on RANDOM input. This
trains the thing, on CPU, at a size that finishes while you wait. It is a glimpse, not a result.

WHAT IT COMPARES: the four bindings as an ablation inside ONE block, so the only thing moving is
the address. Params are identical across all four (u_proj is allocated even when the state factor
is off, deliberately, so the comparison is not confounded by capacity).

WHAT IT CANNOT SHOW, stated up front because the temptation will be to read it as more:

  THE TASK HAS NO REVISITS, so the episodic benefit - the single largest effect the numerics
  found, 0.736 -> 0.940 - IS NOT EXERCISED HERE AT ALL. `chain` and `mqar` each ask for one fact
  per query and never ask what a content pointed to EARLIER. So the state and time factors can
  only cost here, never pay. A binding arm that merely holds level with content-only is doing
  well, and that is the honest prior going in.

  THE BUDGET IS FAR BELOW THE CEILING. ceiling_sweep needed 6000 steps at d=64 before a
  transformer solved even mqar-2. At the default budget here the ceiling will NOT clear 0.90, so
  ACCURACY IS NOT RANKABLE. What is readable is the loss trajectory, whether every arm trains at
  all, and lag_cos - whether the learned tap keeps S a transition operator.

  Building the task that DOES exercise episodic recall is the next step, and it needs care: the
  last three task bugs here were a unique-sink shortcut at ratio=0.0, a PAD collision, and a
  perturbation LayerNorm ignored. A new generator gets its own verification, not a quick draft.

RESULTS, and they overturn what run 1 seemed to say.

  run 1 (query-binding BUG present)        run 2 (bug fixed, 357b09a)
    mqar   content only        1.000         0.793   lag_cos 0.516  tap_max 0.510
           content*state       0.207         0.035   lag_cos 0.610  tap_max 0.303
           content*time        0.875         0.547   lag_cos 0.518  tap_max 0.273
           content*state*time  0.160         0.035   lag_cos 0.443  tap_max 0.293
    chain  content only        0.277         0.078   lag_cos 0.696  tap_max 0.289
           content*state       0.027         0.027   lag_cos 0.610  tap_max 0.329
           content*time        0.211         0.129   lag_cos 0.680  tap_max 0.316
           content*state*time  0.051         0.047   lag_cos 0.300  tap_max 0.443
    (transformer 0.336 / 0.152 both runs - identical, so seeding is deterministic)

TWO THINGS CHANGED BETWEEN THE RUNS, SO ATTRIBUTION IS NOT CLEAN. Besides the query fix, the
read-loop eps guard landed (34eec90, 1e-12 -> 1e-4). CONTENT-ONLY CANNOT HAVE BEEN AFFECTED BY
THE QUERY FIX - with use_state and use_time both False, the added binding is skipped and the
extra F.normalize is idempotent on an already-normalised vector - yet it fell 1.000 -> 0.793.
So the eps guard is the only live candidate, and my claim that it was "provably inert" was
verified on RANDOM INIT IN float64, not on a trained trajectory. In training ||S r|| does get
small, and there max(||r||, 1e-4) stops renormalising a tiny vector instead of amplifying it -
a real behavioural change, asserted away rather than measured.

eps_ablation.py SETTLED IT, AND I WAS WRONG TO SUSPECT THE GUARD:

    mqar    eps=1e-12  seed0 0.793  seed1 1.000      lag_cos 0.516
            eps=1e-04  seed0 0.793  seed1 0.992      lag_cos 0.516
    chain   eps=1e-12  seed0 0.074  seed1 0.496      lag_cos 0.696
            eps=1e-04  seed0 0.078  seed1 0.316      lag_cos 0.696

SEED 0 IS UNCHANGED BY eps - 0.793 vs 0.793, 0.074 vs 0.078 - and lag_cos is identical to three
figures at both settings. Both glimpse runs used seed 0, so the guard did not cause the drop.
"Provably inert" survives; my suspicion of it did not.

What did cause it: the query fix is mathematically a no-op for content-only but NOT numerically
identical - F.normalize(F.normalize(x)) differs from F.normalize(x) in the last bits, and over
600 chaotic training steps a 1e-7 perturbation at step 1 lands somewhere else entirely. Neither
fix broke anything; run 1 and run 2 are two draws from a divergent trajectory.

AND THE FINDING THAT MATTERS MORE THAN EITHER. Chain accuracy is 0.074 on seed 0 and 0.496 on
seed 1 AT IDENTICAL SETTINGS - a 6.7x spread. At 600 steps and d=64 a single-seed number here is
close to meaningless, and the spread is not the kind you average away: 0.074 vs 0.496 is "did not
find the solution" vs "partly found it", a lottery rather than wobble about a mean, because the
model sits right at the threshold of learning the task at all. Every single-seed comparison in
this file is therefore softer than it reads. The binding collapse (0.035 against 0.793) is a big
enough gap to survive it; the finer readings are not.

THE BINDING ARMS ARE STILL BROKEN, AND THE QUERY FIX WAS NECESSARY BUT NOT SUFFICIENT.
content*state is 0.035 against content-only's 0.793 - worse than the buggy run's 0.207. The
reason is the flaw named but not fixed when the query bug was found: THE TRACKER REFLECTS ON
EVERY TOKEN. A fact written at position t is addressed with state h_t, while the query at the
readout carries h_L. Every fact therefore gets a unique address that nothing can ever query
again. full_stack's numerics worked precisely because its states were REUSED (sA, sB) and each
query carried the same state its fact was written with - a condition a per-token reflection
tracker never satisfies.

The fix is a slowly-varying state: gate the tracker so h only moves when the model chooses, and
facts within a context share an address. That is a design change to SaryuBlock, it wants its own
invariants, and it does NOT block the depth sweep - gpu_run uses ParityBlock, which has no
tracker at all.

usage: cd experiments && STEPS=1200 python glimpse.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recall_tasks import vocab
from saryu_block import SaryuBlock
from shift_arms import SwiGLU, batch, Net as RefNet

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))


class Net(nn.Module):
    def __init__(self, V, d, nl, **blk):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.mix = nn.ModuleList([SaryuBlock(d, **blk) for _ in range(nl)])
        self.ffn = nn.ModuleList([SwiGLU(d) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, idx):
        x = self.emb(idx)
        for m, f in zip(self.mix, self.ffn):
            x = x + m(x)
            x = x + f(x)
        return self.head(self.lnf(x))[:, -1]


def lag_cos(net):
    """cos(k_t, v_(t-1)) with the LEARNED tap - whether S is still a transition operator.
    conv_vs_tap showed tap_max cannot answer this: the short conv can supply the lag on its own."""
    blk = net.mix[0]
    blk.debug = True
    with torch.no_grad():
        net(torch.randint(0, net.emb.num_embeddings, (4, 16)))
    st = blk.trace['steps']
    blk.debug, blk.trace = False, None
    return float(np.mean([float(F.cosine_similarity(st[i][0], st[i - 1][1], dim=-1).mean())
                          for i in range(1, len(st))]))


def train(arm, task, steps, d, nl, nent, blk, bs=32, lr=3e-3, seed=0, **cell):
    torch.manual_seed(seed)
    _, _, _, V = vocab(nent)
    net = RefNet('transformer', V, d, nl) if arm == 'transformer' else Net(V, d, nl, **blk)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    t0, losses = time.time(), []
    for _ in range(steps):
        x, y = batch(task, bs, rng, nent, **cell)
        loss = F.cross_entropy(net(x), y)
        if not torch.isfinite(loss):
            return dict(acc=float('nan'), params=npar, note='NON-FINITE')
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
    lc = None if arm == 'transformer' else lag_cos(net)
    tm = None if arm == 'transformer' else float(net.mix[0].tap_weights().max())
    return dict(acc=float(np.mean(corr)), params=npar, secs=time.time() - t0,
                loss0=losses[0], loss1=losses[-1], lag=lc, tap=tm, note='')


ARMS = [('transformer', None),
        ('content only', dict(use_state=False, use_time=False)),
        ('content*state', dict(use_state=True, use_time=False)),
        ('content*time', dict(use_state=False, use_time=True)),
        ('content*state*time', dict(use_state=True, use_time=True))]

CELLS = [('mqar npairs=2', 'mqar', dict(npairs=2, seqlen=32)),
         ('chain d2 ratio=1.0', 'chain', dict(depth=2, ratio=1.0, seqlen=48))]

if __name__ == '__main__':
    STEPS = int(os.environ.get('STEPS', 1200))
    D, NL, NENT = int(os.environ.get('D', 64)), 2, 32
    HD = int(os.environ.get('HD', 32))
    base = dict(head_dim=HD, conv=True, n_read=2, tap_init='uniform', bias='l2')

    print('GLIMPSE  d=%d head_dim=%d nl=%d steps=%d  - a CODE-AND-TREND check, not a result'
          % (D, HD, NL, STEPS))
    print('the task has NO revisits, so the episodic benefit (0.736->0.940) is NOT tested here;')
    print('the binding arms can only cost on these cells. Holding level is a good outcome.\n')

    for cname, task, cell in CELLS:
        print('=== %s ===' % cname, flush=True)
        print('  %-20s %-8s %-9s %-9s %-9s %-8s %-8s %s'
              % ('arm', 'acc', 'loss0', 'loss1', 'params', 'lag_cos', 'tap_max', 'secs'))
        ceiling = None
        for name, kw in ARMS:
            r = train(name, task, STEPS, D, NL, NENT, dict(base, **kw) if kw else None, **cell)
            if name == 'transformer':
                ceiling = r['acc']
            print('  %-20s %-8.3f %-9.3f %-9.3f %-9d %-8s %-8s %.0f'
                  % (name, r['acc'], r['loss0'], r['loss1'], r['params'],
                     '-' if r['lag'] is None else '%.3f' % r['lag'],
                     '-' if r['tap'] is None else '%.3f' % r['tap'], r['secs']), flush=True)
        print('  ceiling %.3f -> %s\n'
              % (ceiling, 'cell readable' if ceiling > 0.90 else
                 'BELOW 0.90: accuracy here ranks nothing, read loss and lag_cos only'), flush=True)

    print('WHAT TO LOOK FOR')
    print('  every arm trains (loss1 < loss0, nothing non-finite) - the block is sound under SGD')
    print('  lag_cos stays high - the learned tap kept S a transition operator')
    print('  binding arms hold LEVEL with content-only - they pay no tax on a task that cannot')
    print('  reward them. If they lose badly here, the address is costing more than numerics said.')
