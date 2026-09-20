"""Does the shipped model fail at OVERWRITING? The benchmark MQAR cannot see.

WHY. write_algebra.txt showed, by construction, that deposit and disposition writes are identical
when every key is written once and CATEGORICALLY different when keys are overwritten: deposit
decays as ~1/R while disposition stays at 1.000. Standard MQAR writes each key exactly once, so
every measurement in this repository ran on a task structurally unable to distinguish the two write
rules. That is why the write never appeared as an axis.

THE CONFOUND THIS CONTROLS FOR. The recurrence has DECAY: a write at t survives as (1-g)^(T-t), so
recent writes are already stronger. If a key's R occurrences arrived in neat rounds, GLOBAL recency
would solve the task and nothing would be learned. So the n*R writes are placed at RANDOM
interleaved positions with each key's own values kept in order, and the target is the queried key's
LAST value -- which needs PER-KEY recency, since another key's later write does not change which of
THIS key's values came last.

TWO MEASUREMENT FIXES, after three training experiments were wasted by the same mistake.

  1. ESCAPE IS BIMODAL. A run lands near 1.0 or sits at chance; it does not land in between.
     Averaging a 1.000 seed and a 0.570 seed into 0.785 describes nothing that happened. The
     statistic is therefore SOLVE RATE -- the fraction of seeds that escape within the step budget
     -- treating each seed as a Bernoulli trial. A prior run had its CONTROL solve 1 of 2 seeds,
     which made the whole comparison unreadable.

  2. ACCURACY DOES NOT SAY WHAT THE MODEL DID. A failure decomposition does, from ONE model and no
     extra training. On the samples where the queried key's last value differs from the globally
     last-written value, the output is scored against:

        correct      the queried key's LAST value              per-key recency: the task
        global_last  the most recent value written ANYWHERE    global recency: decay alone
        key_other    another of the queried key's own values   right key, wrong time
        off_key      a value belonging to some other key       no binding at all

     These are mutually exclusive and sum to 1. The previous run's flat ~0.53 at both R=2 and R=4
     is exactly what global recency predicts at n=2 (correct with probability 1/n, independent of
     R) -- but that was inferred from an accuracy number rather than measured, so it is a
     hypothesis, not a finding, until this decomposition runs.

REGISTERED PREDICTIONS, before running:
  P1  the control R=1 has a HIGH solve rate. If it does not, nothing else here is readable.
  P2  R=2 has a much lower solve rate than R=1 at the same key count.
  P3  on the failures, `global_last` is the dominant bucket and `off_key` is small -- the model
      binds fine and simply cannot tell WHICH write to a key came last.
  P4  accuracy is flat in R rather than falling as 1/R. (Already falsified as 1/R by the previous
      run; stated here so the replacement prediction is on the record before it is tested.)

FALSIFIER: R=2 solves at the same rate as R=1. Then decay plus training already handles
overwriting and the write-rule argument does not leave the construction.

    python evidence/overwrite_task.py
Environment: PAIRS, ROUNDS, GAP, STEPS, LR, OPT, NH, NL, SEEDS, SAVE, THREADS.
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
from saryu.metrics import Run                                        # noqa: E402
from binding_long import Muon                                        # noqa: E402

PAIRS = [int(x) for x in os.environ.get('PAIRS', '2').split(',')]
ROUNDS = [int(x) for x in os.environ.get('ROUNDS', '1,2').split(',')]
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 5000))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
OPT = os.environ.get('OPT', 'muon')
NH = int(os.environ.get('NH', 2))
NL = int(os.environ.get('NL', 2))
SEEDS = int(os.environ.get('SEEDS', 6))
SOLVED = float(os.environ.get('SOLVED', 0.80))       # what counts as an escape
SAVE = os.environ.get('SAVE', os.path.join(ROOT, 'checkpoints', 'overwrite'))
D, NENT = 128, 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 8)))


def make(rng, n, R):
    """n keys, each written R times at RANDOM interleaved positions, per-key order preserved.

    Returns (sequence, target, global_last_value, the queried key's own R values). The extra
    fields are what the failure decomposition needs."""
    ent = rng.choice(NENT, size=n + n * R, replace=False)
    ks, vals = ent[:n], ent[n:].reshape(n, R)       # vals[i, r] is key i's r-th value
    order = rng.permutation(n * R)
    owner = np.repeat(np.arange(n), R)[np.argsort(order)]
    seen = np.zeros(n, dtype=int)
    seq = []
    for o in owner:
        seq += [int(ks[o]), int(vals[o, seen[o]])]
        seen[o] += 1
    rest = np.setdiff1d(np.arange(NENT), ent)
    seq += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, n))
    seq += [SEP, int(ks[i])]
    glast = int(vals[owner[-1], R - 1])             # last value written ANYWHERE
    return seq, int(vals[i, R - 1]), glast, [int(v) for v in vals[i]]


def batch(rng, bs, n, R):
    out = [make(rng, n, R) for _ in range(bs)]
    xs = torch.tensor([o[0] for o in out])
    ys = torch.tensor([o[1] for o in out])
    gl = torch.tensor([o[2] for o in out])
    kv = torch.tensor([o[3] for o in out])
    return xs, ys, gl, kv


@torch.no_grad()
def decompose(mdl, n, R, n_batch=12):
    """What did the model output, on the samples where per-key and global recency DISAGREE?"""
    rng = np.random.default_rng(4242)
    buckets = dict(correct=0, global_last=0, key_other=0, off_key=0)
    tot = 0
    for _ in range(n_batch):
        x, y, gl, kv = batch(rng, BS, n, R)
        sel = y != gl                                    # only where the two accounts differ
        if not bool(sel.any()):
            continue
        p = mdl(x[sel])[:, -1].argmax(-1)
        yy, gg, kk = y[sel], gl[sel], kv[sel]
        for a, t, g, ks_ in zip(p.tolist(), yy.tolist(), gg.tolist(), kk.tolist()):
            tot += 1
            if a == t:
                buckets['correct'] += 1
            elif a == g:
                buckets['global_last'] += 1
            elif a in ks_:
                buckets['key_other'] += 1
            else:
                buckets['off_key'] += 1
    return {k: v / max(tot, 1) for k, v in buckets.items()}, tot


def train_one(n, R, seed):
    torch.manual_seed(seed)
    mdl = SaryuV3LM(NENT + 2, D, NL, nh=NH)
    p = list(mdl.parameters())
    opt = (torch.optim.AdamW(p, lr=LR, weight_decay=0.01) if OPT == 'adamw' else Muon(p, lr=LR))
    rng = np.random.default_rng(1000 + seed)
    ev = np.random.default_rng(7)
    log = Run(f'ovw2-p{n}-R{R}-s{seed}',
              config=dict(arm=f'overwrite/p{n}R{R}', pairs=n, rounds=R, lr=LR, opt=OPT, nh=NH,
                          nl=NL, steps=STEPS, seed=seed,
                          chance_top1=round(1 / (n * R), 4),
                          chance_global_recency=round(1 / n, 4)))
    every = 250
    best = 0.0
    for s in range(STEPS):
        x, y, _, _ = batch(rng, BS, n, R)
        ce = F.cross_entropy(mdl(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(p, 1.0); opt.step()
        if s == 0 or (s + 1) % every == 0:
            mdl.eval()
            with torch.no_grad():
                acc = 0.
                for _ in range(4):
                    x2, y2, _, _ = batch(ev, BS, n, R)
                    acc += float((mdl(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            mdl.train()
            best = max(best, acc)
            log.log(s + 1, loss=float(ce), **{'eval/top1': acc, 'eval/best': best})
    mdl.eval()
    dec, ndec = decompose(mdl, n, R)
    log.log(STEPS, **{f'decomp/{k}': v for k, v in dec.items()})
    log.done()
    os.makedirs(SAVE, exist_ok=True)
    torch.save({'state': mdl.state_dict(), 'd': D, 'nl': NL, 'nh': NH, 'vocab': NENT + 2,
                'pairs': n, 'rounds': R, 'seed': seed, 'best': best, 'decomp': dec},
               os.path.join(SAVE, f'ovw2-p{n}-R{R}-s{seed}.pt'))
    return best, dec, ndec


def main():
    print(f'OVERWRITE TASK  {OPT} lr {LR:g}, n_h {NH}, {NL} layers, {STEPS} steps, {SEEDS} seeds')
    print('each key written R times at random interleaved positions; target is its LAST value')
    print(f'solve rate = fraction of seeds reaching top1 > {SOLVED:.2f} (escape is BIMODAL, so the')
    print('mean over seeds is not a meaningful statistic)\n')
    hdr = (f'{"keys":>5} {"R":>3} {"solved":>8} {"rate":>6} | on disagreeing samples: '
           f'{"correct":>8} {"glob_last":>10} {"key_other":>10} {"off_key":>8}')
    print(hdr); print('-' * len(hdr))
    t0 = time.time()
    for n in PAIRS:
        for R in ROUNDS:
            res = [train_one(n, R, s) for s in range(SEEDS)]
            bests = [r[0] for r in res]
            nsolve = sum(b > SOLVED for b in bests)
            agg = {k: float(np.mean([r[1][k] for r in res])) for k in res[0][1]}
            print(f'{n:>5} {R:>3} {f"{nsolve}/{SEEDS}":>8} {nsolve/SEEDS:>6.2f} | '
                  f'{"":>23} {agg["correct"]:>8.3f} {agg["global_last"]:>10.3f} '
                  f'{agg["key_other"]:>10.3f} {agg["off_key"]:>8.3f}', flush=True)
            print(f'{"":>5} {"":>3} seeds {", ".join(f"{b:.3f}" for b in bests)}', flush=True)
    print(f'\n{time.time()-t0:.0f}s   checkpoints in {SAVE}')
    print('READ')
    print('  control R=1 solve rate low        -> harness unusable, fix before reading anything')
    print('  R=2 solve rate << R=1             -> overwriting is a real trained failure')
    print('  global_last dominant on failures  -> the model has GLOBAL recency (decay) and lacks')
    print('                                       PER-KEY recency, which is the missing erase')
    print('  off_key dominant                  -> binding itself failed; a different problem')


if __name__ == '__main__':
    main()
