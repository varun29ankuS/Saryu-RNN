"""Is the slow learning the architecture, or the supervision density?

THE SUSPICION. Our MQAR harness asks ONE query per sequence and takes the loss at one position, so
a batch of 32 yields 32 labels per step -- 14x fewer than a token-level LM on the same batch, and
96,000 labels over a whole 3000-step run. Meanwhile the trained curve is flat until about step 1000
and still rising at 3000 (evidence/results/trace_base.txt). MQAR is *Multi-Query* Associative
Recall: the standard task puts several queries in one sequence and supervises at every one. We built
the single-query version and threw away a factor of n_pairs in signal.

THE TWO ARMS, identical in every other respect -- same model, steps, batch, lr, pairs, gap:

  single   k0 v0 .. kn vn | filler | SEP kq              loss at 1 position   (what we have run)
  multi    k0 v0 .. kn vn | filler | SEP kq0 vq0 kq1 ..  loss at n positions  (standard MQAR)

In the multi arm each query key is followed by its value, and the loss is taken AT each key
position, predicting the value that follows. Nothing is leaked that the first half did not already
show, and the queries are in random order, so a positional shortcut does not survive.

REGISTERED PREDICTION, before running. If supervision density is the constraint, the multi arm
reaches the single arm's final top1 in roughly 1/n_pairs of the steps, and ends higher. If the two
curves lie on top of each other per STEP, the bottleneck is the architecture or the optimiser and
not the labels.

    python evidence/supervision.py
Environment: PAIRS, GAP, STEPS, BS, LR, SEED, EVERY, THREADS, ARMS.
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
from saryu.model import SaryuV3LM                                     # noqa: E402
from saryu.metrics import Run                                         # noqa: E402

PAIRS = int(os.environ.get('PAIRS', 4))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 3000))
EVERY = int(os.environ.get('EVERY', 250))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
SEED = int(os.environ.get('SEED', 0))
D, NL, NENT = 128, 2, 64
SEP, PAD = NENT, NENT + 1
torch.set_num_threads(int(os.environ.get('THREADS', 8)))

SEQ_SINGLE = 2 * PAIRS + GAP + 2
SEQ_MULTI = 2 * PAIRS + GAP + 1 + 2 * PAIRS
# In the multi arm the query keys sit at fixed offsets, so targets can be gathered without masks.
QPOS = [2 * PAIRS + GAP + 1 + 2 * i for i in range(PAIRS)]


def make(rng, multi):
    ent = rng.choice(NENT, size=2 * PAIRS, replace=False)
    ks, vs = ent[:PAIRS], ent[PAIRS:]
    rest = np.setdiff1d(np.arange(NENT), ent)
    seq = []
    for k, v in zip(ks, vs):
        seq += [int(k), int(v)]
    seq += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    if not multi:
        i = int(rng.integers(0, PAIRS))
        seq += [SEP, int(ks[i])]
        return seq, [int(vs[i])]
    seq += [SEP]
    order = rng.permutation(PAIRS)
    tgt = []
    for j in order:
        seq += [int(ks[j]), int(vs[j])]     # loss is taken AT the key, predicting the value
        tgt.append(int(vs[j]))
    return seq, tgt


def batch(rng, multi, bs):
    xs, ys = zip(*(make(rng, multi) for _ in range(bs)))
    return torch.tensor(xs), torch.tensor(ys)


def loss_and_acc(m, x, y, multi):
    lg = m(x)
    if not multi:
        out = lg[:, -1]
        return F.cross_entropy(out, y[:, 0]), (out.argmax(-1) == y[:, 0]).float().mean()
    out = lg[:, QPOS]                                    # [B, PAIRS, V]
    return (F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1)),
            (out.argmax(-1) == y).float().mean())


def evaluate(m, multi, n=12):
    ev = np.random.default_rng(99)
    hit = tot = 0
    with torch.no_grad():
        for _ in range(n):
            x, y = batch(ev, multi, BS)
            lg = m(x)
            out = lg[:, -1][:, None] if not multi else lg[:, QPOS]
            hit += int((out.argmax(-1) == y).sum()); tot += y.numel()
    return hit / tot


def run(arm):
    multi = arm == 'multi'
    torch.manual_seed(SEED)
    m = SaryuV3LM(NENT + 2, D, NL)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
    rng = np.random.default_rng(1000 + SEED)
    labels = BS * (PAIRS if multi else 1)
    log = Run(f'sup-{arm}', config=dict(arm=arm, pairs=PAIRS, gap=GAP, lr=LR, bs=BS,
                                        labels_per_step=labels, chance_top1=round(1 / NENT, 4),
                                        chance_loss=round(float(np.log(NENT)), 3)))
    print(f'\n{arm}: seqlen {SEQ_MULTI if multi else SEQ_SINGLE}, '
          f'{labels} labels/step, {labels*STEPS:,} total')
    print(f'{"step":>6} {"loss":>7} {"eval":>7} {"labels":>12}')
    t0 = time.time()
    for s in range(STEPS):
        x, y = batch(rng, multi, BS)
        loss, _ = loss_and_acc(m, x, y, multi)
        opt.zero_grad(); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
        opt.step(); sch.step()
        if s == 0 or (s + 1) % EVERY == 0:
            m.eval(); a = evaluate(m, multi); m.train()
            log.log(s + 1, loss=float(loss), **{'eval/top1': a, 'grad/norm': gn,
                                                'labels/seen': labels * (s + 1)})
            print(f'{s+1:>6} {float(loss):>7.3f} {a:>7.3f} {labels*(s+1):>12,}', flush=True)
    log.done(top1=evaluate(m, multi))
    print(f'  {time.time()-t0:.0f}s')
    return evaluate(m, multi)


def main():
    print(f'SUPERVISION DENSITY  {PAIRS} pairs, gap {GAP}, bs {BS}, lr {LR:g}, {STEPS} steps')
    print(f'chance {1/NENT:.3f}')
    res = {}
    for arm in [a for a in os.environ.get('ARMS', 'single,multi').split(',')]:
        res[arm] = run(arm)
    print('\nFINAL')
    for k, v in res.items():
        print(f'  {k:<8} {v:.3f}')
    if 'single' in res and 'multi' in res:
        print(f'\n  multi uses {PAIRS}x the labels per step for essentially the same compute.')
        print('  If the curves match per step, labels were not the constraint.')


if __name__ == '__main__':
    main()
