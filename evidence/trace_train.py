"""Where exactly does MQAR training go wrong? Print the diagnostics, not just the final number.

WHY. Every arm tried so far lands between 0.113 and 0.184 (base, base-wide with 40% more
parameters, carve, carve+bind, sig2, sig2-anti), and base at 4 pairs with ZERO gap reaches only
0.199. A single end-of-run accuracy cannot say whether that is capacity, distance, optimisation, or
a harness ceiling. These columns can:

  loss        against chance ln(64) = 4.159. Below it means something is being learned.
  top1/top5   if top5 is high while top1 is not, the value is IN the state and the readout cannot
              rank it -- a retrieval problem, not a storage one.
  p0..p3      accuracy split by WHICH pair was queried, p3 being the most recently written.
              A memory of capacity one looks like p3 high and p0..p2 at chance. This is the
              single most diagnostic column here.
  lvl2        ||level-2 readout|| / ||level-1 output||, so a dead or ignored branch is visible
              rather than inferred.
  ret         mean gate retention 1/g in tokens: whether the state can even span the gap.

    python evidence/trace_train.py
Environment: ARM, PAIRS, GAP, STEPS, D, NL, LR, EVERY, SEED, THREADS.
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

ARMS = {
    'base': (128, dict()),
    'carve+bind': (128, dict(carve=8, bind_write=True)),
    'sig2': (128, dict(sig2=True)),
    'sig2-anti': (128, dict(sig2=True, sig2_anti=True)),
}
ARM = os.environ.get('ARM', 'base')
PAIRS = int(os.environ.get('PAIRS', 4))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 3000))
EVERY = int(os.environ.get('EVERY', 250))
NL = int(os.environ.get('NL', 2))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
SEED = int(os.environ.get('SEED', 0))
NENT = 64
torch.set_num_threads(int(os.environ.get('THREADS', 8)))


def make(rng, npairs, gap, seqlen):
    """As nh_sweep.make_recall, but also returns WHICH pair was queried."""
    sep, pad = NENT, NENT + 1
    ent = rng.choice(NENT, size=2 * npairs, replace=False)
    ks, vs = ent[:npairs], ent[npairs:]
    rest = np.setdiff1d(np.arange(NENT), ent)
    seq = []
    for k, v in zip(ks, vs):
        seq += [int(k), int(v)]
    seq += [int(x) for x in rng.choice(rest, size=gap, replace=True)]
    i = int(rng.integers(0, npairs))
    seq += [sep, int(ks[i])]
    return [pad] * (seqlen - len(seq)) + seq, int(vs[i]), i, [int(v) for v in vs]


def batch(rng, npairs, gap, seqlen, bs):
    xs, ys, ii, vv = zip(*(make(rng, npairs, gap, seqlen) for _ in range(bs)))
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(ii), torch.tensor(vv)


def evaluate(m, seqlen, n_batches=12):
    ev = np.random.default_rng(99)
    hit = hit5 = tot = inset = 0
    per = np.zeros(PAIRS); cnt = np.zeros(PAIRS)
    with torch.no_grad():
        for _ in range(n_batches):
            x, y, i, vv = batch(ev, PAIRS, GAP, seqlen, BS)
            lg = m(x)[:, -1]
            pred = lg.argmax(-1)
            ok = (pred == y)
            hit += int(ok.sum()); tot += len(y)
            hit5 += int((lg.topk(5, -1).indices == y[:, None]).any(-1).sum())
            # THE decisive split: is the prediction one of the values present in THIS sequence?
            # in_set ~ 1.0 together with top1 ~ 1/PAIRS means the model knows the SET of values
            # and not the key->value pairing -- set membership without binding, which is exactly
            # what a level-1 (sum-of-writes) state can represent.
            inset += int((pred[:, None] == vv).any(-1).sum())
            for j in range(PAIRS):
                sel = (i == j)
                per[j] += float(ok[sel].sum()); cnt[j] += int(sel.sum())
    return hit / tot, hit5 / tot, per / np.maximum(cnt, 1), inset / tot


def diagnostics(m, x):
    """Level-2 usage and gate retention, measured on a real batch."""
    blk = m.mix[0]
    with torch.no_grad():
        z = blk.ln(m.emb(x))
        z = z + blk.conv(z.transpose(1, 2))[..., :z.shape[1]].transpose(1, 2)
        g = ((1.0 - torch.cos(blk.g_proj(z))) / 2.0).clamp(max=0.90)
        ret = float((1.0 / g.clamp_min(1e-6)).median())   # median: the mean is dominated by near-zero gates
        lvl2 = float('nan')
        if getattr(blk, 'sig2', False):
            r = blk.r_out(blk.level2(z))
            base = blk.out(blk.hnorm(torch.zeros(x.shape[0], x.shape[1], blk.H, blk.dh)
                                     ).reshape(x.shape[0], x.shape[1], blk.d))
            del base
            y = m(x)
            lvl2 = float(r.norm() / (y.norm() + 1e-6))
    return ret, lvl2


def main():
    d, kw = ARMS[ARM]
    seqlen = 2 * PAIRS + GAP + 2
    torch.manual_seed(SEED)
    m = SaryuV3LM(NENT + 2, d, NL, **kw)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
    rng = np.random.default_rng(1000 + SEED)
    n = sum(p.numel() for p in m.parameters())
    print(f'TRACE  arm={ARM}  {PAIRS} pairs, gap {GAP}, seqlen {seqlen}, d={d}, {NL} layers, '
          f'{n:,} params')
    print(f'lr={LR:g}, {STEPS} steps, chance top1 {1/NENT:.3f}, chance top5 {5/NENT:.3f}, '
          f'chance loss {np.log(NENT):.3f}\n')
    run = Run(os.environ.get('RUN', f'{ARM}-p{PAIRS}-g{GAP}'),
              config=dict(arm=ARM, pairs=PAIRS, gap=GAP, d=d, nl=NL, lr=LR, steps=STEPS,
                          params=n, chance_top1=round(1/NENT, 4),
                          chance_loss=round(float(np.log(NENT)), 3)))
    cols = (f'{"step":>6} {"loss":>7} {"top1":>7} {"top5":>7} {"inset":>7} '
            + ''.join(f'{"p"+str(j):>7}' for j in range(PAIRS))
            + f' {"ret":>7} {"lvl2":>7} {"gnorm":>8}')
    print(cols); print('-' * len(cols))
    t0 = time.time()
    for s in range(STEPS):
        x, y, _, _ = batch(rng, PAIRS, GAP, seqlen, BS)
        loss = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
        opt.step(); sch.step()
        if s == 0 or (s + 1) % EVERY == 0:
            m.eval()
            a1, a5, per, ins = evaluate(m, seqlen)
            ret, lvl2 = diagnostics(m, x)
            m.train()
            run.log(s + 1, loss=float(loss), **{'eval/top1': a1, 'eval/top5': a5, 'eval/in_set': ins,
                    'grad/norm': gn, 'diag/ret': ret, 'diag/lvl2': lvl2},
                    **{f'eval/p{j}': float(per[j]) for j in range(PAIRS)})
            print(f'{s+1:>6} {float(loss):>7.3f} {a1:>7.3f} {a5:>7.3f} {ins:>7.3f} '
                  + ''.join(f'{v:>7.3f}' for v in per)
                  + f' {ret:>7.1f} {lvl2:>7.3f} {gn:>8.3f}', flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READING IT: p3 high with p0..p2 at chance = a memory that holds one fact.')
    print('  top5 high with top1 low = the value is there and the readout cannot rank it.')
    print('  ret below the gap = the state cannot span the distance at all.')


if __name__ == '__main__':
    main()
