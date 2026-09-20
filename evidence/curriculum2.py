"""Replicate the one thing that moved -- and watch the manifold radius while it does.

WHAT HAPPENED. Across roughly twenty architectural interventions and sixty seeds, 4-pair recall in
this project has never exceeded 0.39. A pair-count curriculum (2 -> 4 on an unchanged model)
produced 0.773 on one seed of four. That is the only number that has ever moved, and one seed of
four is exactly the sample size that has produced false findings here all day, so the first job is
to find out whether it is real.

THE SECOND THING TO CHECK. manifold_pulse.txt found the first geometric quantity that separates the
regimes: within-class manifold radius at the query position is 0.2-0.3 where the model works
(2 pairs) and 1.3-1.8 where it does not (4 pairs). Manifold capacity theory says separability
collapses as radius grows, so if the curriculum works BECAUSE it keeps the manifolds tight, R_M
should contract on the seeds that lock in and stay fat on the seeds that do not. That turns a
correlation into a mechanism -- or kills it.

ARMS. Architecture identical throughout: stock SaryuV3LM, nh=2, H=8, d=128, 2 layers. Only the
order of the data changes.

    uniform     4 pairs throughout. The control, and what every previous experiment ran.
    slow        equal thirds at 2, 3 and 4 pairs. The first run spent only a quarter of training at
                each of 2 and 3, which may be why it half-worked.
    gated       advance only when held-out accuracy at the CURRENT pair count clears a threshold,
                with a minimum dwell so it cannot race to the target. The first gated arm scored
                BELOW the control (0.156-0.250), which is consistent with advancing too early and
                spending almost no time at 4 pairs -- this version fixes that.

REGISTERED PREDICTIONS, before running:
  P1  slow has a strictly higher solve rate than uniform over 8 seeds.
  P2  on seeds that lock in, R_M at the query position falls toward the 2-pair value (~0.3); on
      seeds that fail it stays near the 4-pair value (~1.5).
  P3  R_M separates locked-in from failed seeds BETTER than loss does at the same step.

FALSIFIER: slow is no better than uniform across 8 seeds. Then 0.773 was a fluke, the curriculum
does not transfer to this architecture, and the only thing that ever moved was noise.

Solve rate over seeds is the statistic; escape is bimodal and a mean over seeds is meaningless.

    python evidence/curriculum2.py
Environment: NPAIRS, GAP, STEPS, LR, OPT, D, H, NL, NH, SEEDS, ARMS, THREADS.
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

TARGET = int(os.environ.get('NPAIRS', 4))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 6000))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
OPT = os.environ.get('OPT', 'muon')
D = int(os.environ.get('D', 128))
H = int(os.environ.get('H', 8))
NL = int(os.environ.get('NL', 2))
NH = int(os.environ.get('NH', 2))
SEEDS = int(os.environ.get('SEEDS', 8))
ARMS = os.environ.get('ARMS', 'uniform,slow,gated').split(',')
SOLVED = float(os.environ.get('SOLVED', 0.80))
NENT = 64
SEP = NENT
NCLS = 8                                     # classes for the R_M probe
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


def make(rng, n, vfix=None):
    pool = np.arange(NENT) if vfix is None else np.setdiff1d(np.arange(NENT), np.arange(NCLS))
    ent = rng.choice(pool, size=2 * n if vfix is None else 2 * n - 1, replace=False)
    if vfix is None:
        ks, vs = [int(z) for z in ent[:n]], [int(z) for z in ent[n:]]
    else:
        ks = [int(ent[0])] + [int(z) for z in ent[1:n]]
        vs = [int(vfix)] + [int(z) for z in ent[n:2 * n - 1]]
    rest = np.setdiff1d(pool, ent)
    s = []
    for k, v in zip(ks, vs):
        s += [k, v]
    s += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, n))
    s += [SEP, ks[i]]
    return s, vs[i]


def batch(rng, bs, n):
    o = [make(rng, n) for _ in range(bs)]
    return torch.tensor([a for a, _ in o]), torch.tensor([b for _, b in o])


@torch.no_grad()
def evaluate(m, rng, n, reps=4):
    m.eval()
    a = 0.
    for _ in range(reps):
        x, y = batch(rng, BS, n)
        a += float((m(x)[:, -1].argmax(-1) == y).float().mean()) / reps
    m.train()
    return a


@torch.no_grad()
def radius(m, n, n_seq=320):
    """Within-class manifold radius at the QUERY position -- the quantity that separated the
    regimes in manifold_pulse.txt (0.2-0.3 where the model works, 1.3-1.8 where it does not)."""
    m.eval()
    rng = np.random.default_rng(31337)
    xs, ys = [], []
    for _ in range(n_seq):
        c = int(rng.integers(0, NCLS))
        s, _ = make(rng, n, vfix=c)
        xs.append(s); ys.append(c)
    x = torch.tensor(xs); y = torch.tensor(ys)
    h = m.emb(x)
    ve = m.vemb(x) if m.vemb is not None else None
    for mix, ffn in zip(m.mix, m.ffn):
        h = h + mix(h, ve)
        h = h + ffn(h)
    Z = m.lnf(h)[:, -1]
    rads = []
    for c in range(NCLS):
        A = Z[y == c]
        if len(A) < 8:
            continue
        mu = A.mean(0)
        rads.append(float((A - mu).norm(dim=-1).mean() / (mu.norm() + 1e-9)))
    m.train()
    return float(np.mean(rads))


def train_one(arm, seed):
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT + 2, D, NL, nh=NH, H=H)
    p = list(m.parameters())
    opt = (torch.optim.AdamW(p, lr=LR, weight_decay=0.01) if OPT == 'adamw' else Muon(p, lr=LR))
    rng = np.random.default_rng(1000 + seed)
    ev = np.random.default_rng(7)
    log = Run(f'c2-{arm}-p{TARGET}-s{seed}',
              config=dict(arm=f'curriculum2/{arm}', pairs=TARGET, gap=GAP, lr=LR, opt=OPT,
                          d=D, H=H, nh=NH, nl=NL, steps=STEPS, seed=seed,
                          chance_top1=round(1 / TARGET, 4)))
    n_cur = 2 if arm in ('slow', 'gated') else TARGET
    dwell = 0
    best, best_R = 0.0, None
    for s in range(STEPS):
        if arm == 'slow':                                   # equal thirds at 2, 3, 4
            n_cur = min(TARGET, 2 + int(3.0 * s / STEPS))
        x, y = batch(rng, BS, n_cur)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(p, 1.0); opt.step()
        dwell += 1
        if s == 0 or (s + 1) % 250 == 0:
            cur = evaluate(m, ev, n_cur)
            tgt = evaluate(m, ev, TARGET)
            R = radius(m, TARGET)
            if arm == 'gated' and n_cur < TARGET and cur > 0.9 and dwell > STEPS // 6:
                n_cur += 1; dwell = 0                       # minimum dwell, so it cannot race
            best = max(best, tgt)
            if best_R is None or tgt >= best:
                best_R = R
            log.log(s + 1, loss=float(ce), **{'eval/top1': tgt, 'eval/best': best,
                                              'cur/acc': cur, 'cur/pairs': n_cur,
                                              'geom/R_M': R})
    log.done()
    return best, radius(m, TARGET)


def main():
    print(f'CURRICULUM 2  target {TARGET} pairs, gap {GAP}, {OPT} lr {LR:g}, {STEPS} steps, '
          f'{SEEDS} seeds')
    print('architecture identical in every arm; only the order of the data changes')
    print(f'R_M is the within-class manifold radius at the query position: ~0.3 where the model')
    print(f'works (2 pairs), ~1.5 where it does not (4 pairs)\n')
    print(f'{"arm":>8} {"solved":>8} {"rate":>6}  best per seed (R_M at end)')
    print('-' * 70)
    t0 = time.time()
    for arm in ARMS:
        res = [train_one(arm, s) for s in range(SEEDS)]
        ns = sum(b > SOLVED for b, _ in res)
        print(f'{arm:>8} {f"{ns}/{SEEDS}":>8} {ns/SEEDS:>6.2f}  '
              + ', '.join(f'{b:.3f}({r:.2f})' for b, r in res), flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ')
    print('  slow > uniform            -> the curriculum is real and the fix is the data order')
    print('  R_M low on locked seeds   -> tight manifolds are the MECHANISM, not a correlate')
    print('  slow == uniform           -> 0.773 was a fluke; nothing has ever moved')


if __name__ == '__main__':
    main()
