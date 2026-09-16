"""The same-order control for the lattice law: S_4 against SL(2,3).

Background
----------
The lattice law says a trained reflection transport that fails a group word problem does not
fail anyhow: it learns a quotient G/N for a normal subgroup N, gets the coset right and guesses
inside it, so its accuracy is exactly 1/|N|. The cross-group evidence (Q_8 against S_4) was
confounded by group order. S_4 and SL(2,3) both have order 24 and disjoint permitted rungs
apart from the endpoints:

    S_4      normal subgroups 1, V_4, A_4, S_4    ->  1.000, 0.250, 0.083, 0.042
    SL(2,3)  normal subgroups 1, Z_2, Q_8, SL(2,3) ->  1.000, 0.500, 0.125, 0.042

Same |G|, same chance level, same vocabulary, architecture, objective and budget.

History (earlier scripts, not in this repository): at n_h = 3 SL(2,3) was not expressible (every
run at chance). At n_h = 8, d = 6, 1200 steps every flat failure landed on its own group's
exclusive rungs (S_4 1 run at 0.250; SL(2,3) 4 runs at 0.500), but only 16 seeds and no
coset check. This run doubles the steps (the budget of the other lattice runs), doubles the
seeds, adds L = 384 to the flatness test and identifies the learned kernel from coset
consistency instead of from the nearest rung.

PREDICTION (committed before running)
-------------------------------------
1. Every flat failure (|acc_12 - acc_96| < 0.05, |acc_96 - acc_384| < 0.05, acc_96 < 0.9)
   has accuracy within 0.03 of 1/|N|, where N is its learned kernel: the smallest normal
   subgroup with coset consistency >= 0.95 at L = 96.
2. No flat failure of either group lands on the other group's exclusive rungs
   (S_4: 0.083, 0.250; SL(2,3): 0.125, 0.500).
FALSIFIED if any flat failure violates 1 or 2.
Non-flat runs are not homomorphisms, so the law says nothing about them; they are reported
and counted, not scored.

Usage
-----
    SHARD=0 NSHARD=3 python evidence/same_order.py    # one shard of the 2 x SEEDS runs
    python evidence/same_order.py --merge             # table + verdict from the shard files
Environment: STEPS (2400), SEEDS (16), D (6), NH (8), SHARD, NSHARD, OUT (evidence/results),
SO_GROUPS (';'-separated, default both; a subset writes shard files tagged with its groups, so
a second subset can share OUT without overwriting the first).

Runs, each committed before it started, all scored by the prediction above:
  results/same_order.txt              2400 steps, 16 seeds per group
  results/same_order_1200/            1200 steps, 16 seeds per group (the earlier runs' budget)
  results/same_order_s4_1800/         1800 steps, S_4 only, 32 seeds. Purpose: 2400 steps gave no
                                      S_4 failures and 1200 steps gave one steady one, so the S_4
                                      side of prediction 2 rested on a single run. Result: 0 steady
                                      failures in 32, nothing tested.
  results/same_order_d3/              D=3, both groups, 2400 steps, 16 seeds per group. Purpose: at
                                      d = 6 S_4 fails by drifting, not by settling. d = 3 is the
                                      smallest width with a faithful S_4 representation and none for
                                      SL(2,3) (its 3-dim irrep has kernel Z_2), so SL(2,3) is expected
                                      to cap at 0.500 and S_4 can fail steadily (an earlier
                                      dimension control: S_4 at 0.258 and 0.235).
                                      Bug: the group list defaulted to 'S_4,SL(2,3)' split on ',',
                                      which cuts 'SL(2,3)' apart, and the GROUPS variable used to
                                      override it is reserved by bash and never reached Python. So
                                      this run (and the S_4-only run above, correct by accident)
                                      trained S_4 only. S_4 result kept as run; SL(2,3) run
                                      separately afterwards with the same settings (*_SL23.txt).
                                      Result: 20 steady failures, 13 on an exclusive rung, none on
                                      the other group's rungs; 1 violation of prediction 1 (SL(2,3)
                                      seed 9, 0.469 vs 0.500 with tolerance 0.03), so FALSIFIED as
                                      registered. Not registered: in binomial standard errors (1024
                                      sequences) that miss is z = -2.00, the only |z| > 1.96 of 28
                                      steady failures across all runs; the tolerance was too tight.
The largest normal subgroup, G itself, always has coset consistency 1, so a kernel always
exists; a run that is merely bad shows up as kernel |G| with accuracy off 1/|G|.

REFERENCES
----------
Grazzi et al. 2024. Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.
    arXiv 2411.12537.
Siems et al. 2025. DeltaProduct: Improving State-Tracking in Linear RNNs via Householder
    Products. arXiv 2502.10297.
"""
import glob
import itertools
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'transport'))
from diagnose import coset_consistency  # noqa: E402

torch.set_num_threads(4)
BS, TRAIN_L = 256, 12
STEPS = int(os.environ.get('STEPS', 2400))
SEEDS = int(os.environ.get('SEEDS', 16))
D, NH = int(os.environ.get('D', 6)), int(os.environ.get('NH', 8))
EVAL_L = (12, 96, 384)
OUT = os.environ.get('OUT', os.path.join(HERE, 'results'))


def s4_table():
    els = list(itertools.permutations(range(4)))
    ix = {p: i for i, p in enumerate(els)}
    return 'S_4', torch.tensor([[ix[tuple(b[a[i]] for i in range(4))] for b in els] for a in els])


def sl23_table():
    """2x2 matrices over F_3 with det 1."""
    els = [m for m in itertools.product(range(3), repeat=4) if (m[0] * m[3] - m[1] * m[2]) % 3 == 1]
    ix = {m: i for i, m in enumerate(els)}

    def mul(x, y):
        a, b, c, d = x
        e, f, g, h = y
        return ((a * e + b * g) % 3, (a * f + b * h) % 3, (c * e + d * g) % 3, (c * f + d * h) % 3)
    return 'SL(2,3)', torch.tensor([[ix[mul(x, y)] for y in els] for x in els])


def normal_subgroups(tab):
    """All normal subgroups, as unions of conjugacy classes closed under the product."""
    n = tab.shape[0]
    e = next(i for i in range(n) if all(tab[i, j].item() == j for j in range(n)))
    inv = {a: next(b for b in range(n) if tab[a, b].item() == e) for a in range(n)}
    classes, seen = [], set()
    for a in range(n):
        if a in seen:
            continue
        c = {tab[tab[g, a], inv[g]].item() for g in range(n)}
        classes.append(c)
        seen |= c
    out = []
    for mask in range(1 << len(classes)):
        s = set().union(*[c for i, c in enumerate(classes) if mask >> i & 1])
        if e in s and all(tab[a, b].item() in s for a in s for b in s):
            out.append(sorted(s))
    return sorted(out, key=len)


class Model(nn.Module):
    """Token-indexed product of NH reflections with free beta (init 2), as in sameorder3."""

    def __init__(self, n, d=D, nh=NH):
        super().__init__()
        self.v = nn.Parameter(torch.randn(n, nh, d) * 0.5)
        self.b = nn.Parameter(torch.full((n, nh), 2.0))
        self.h0 = nn.Parameter(torch.randn(d) * 0.3)
        self.out = nn.Linear(d, n)
        self.nh, self.d = nh, d

    def T(self, h, tok):
        V, B = self.v[tok], self.b[tok]
        for i in range(self.nh):
            u = F.normalize(V[:, i], dim=-1)
            h = h - B[:, i:i + 1] * (h * u).sum(-1, keepdim=True) * u
        return h

    def forward(self, x):
        h = self.h0[None].expand(x.shape[0], -1)
        o = []
        for t in range(x.shape[1]):
            h = self.T(h, x[:, t])
            o.append(h)
        return self.out(torch.stack(o, 1))


def batch(tab, bs, L, g):
    n = tab.shape[0]
    x = torch.randint(0, n, (bs, L), generator=g)
    acc = x[:, 0].clone()
    ys = [acc.clone()]
    for t in range(1, L):
        acc = tab[acc, x[:, t]]
        ys.append(acc.clone())
    return x, torch.stack(ys, 1)


def hom_loss(m, tab, n=64):
    h = torch.randn(n, m.d)
    a = torch.randint(0, tab.shape[0], (n,))
    b = torch.randint(0, tab.shape[0], (n,))
    return (m.T(m.T(h, a), b) - m.T(h, tab[a, b])).norm(dim=1).mean() / m.d ** .5


def run(tab, seed):
    n = tab.shape[0]
    torch.manual_seed(seed)
    m = Model(n)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=.01)
    g = torch.Generator().manual_seed(500 + seed)
    for _ in range(STEPS):
        x, y = batch(tab, BS, TRAIN_L, g)
        loss = F.cross_entropy(m(x).reshape(-1, n), y.reshape(-1)) + hom_loss(m, tab)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
        opt.step()
    ge = torch.Generator().manual_seed(99)
    acc, pred96, true96 = {}, [], []
    with torch.no_grad():
        for L in EVAL_L:
            a = 0.
            for _ in range(4):
                x, y = batch(tab, 256, L, ge)
                p = m(x)[:, -1].argmax(-1)
                a += (p == y[:, -1]).float().mean().item()
                if L == 96:
                    pred96.append(p)
                    true96.append(y[:, -1])
            acc[L] = a / 4
        gl = float(hom_loss(m, tab, 300))
    return acc, torch.cat(pred96), torch.cat(true96), gl


def rungs(tab):
    return sorted({1.0 / len(s) for s in normal_subgroups(tab)})


def classify(row, own, other):
    """Score one run against the prediction. Returns (flat, failure, verdict string)."""
    a12, a96, a384 = row['acc']['12'], row['acc']['96'], row['acc']['384']
    flat = abs(a12 - a96) < 0.05 and abs(a96 - a384) < 0.05
    if a96 >= 0.9:
        return flat, False, 'solved' if flat else 'solved at 96, drifts'
    if not flat:
        return False, True, 'not flat (not scored)'
    k = row['kernel']
    ceiling = 1.0 / k
    if abs(a96 - ceiling) > 0.03:
        return True, True, f'VIOLATES 1: acc {a96:.3f} off 1/|N| = {ceiling:.3f}'
    if any(abs(ceiling - r) < 1e-9 for r in set(other) - set(own)):
        return True, True, f'VIOLATES 2: lands on the other group\'s rung {ceiling:.3f}'
    return True, True, f'on own rung 1/{k} = {ceiling:.3f}'


def merge():
    tabs = dict([s4_table(), sl23_table()])
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, 'same_order_shard*.txt'))):
        with open(path) as f:
            rows += [json.loads(line[5:]) for line in f if line.startswith('#ROW ')]
    rows.sort(key=lambda r: (r['group'] != 'S_4', r['seed']))
    R = {name: rungs(t) for name, t in tabs.items()}
    lines = [f'SAME ORDER, DIFFERENT LATTICE  [n_h={NH}, d={D}, {STEPS} steps, train L={TRAIN_L}]', '']
    for name in tabs:
        other = [o for o in tabs if o != name][0]
        lines.append(f'  {name:>8}  rungs ' + ', '.join(f'{r:.3f}' for r in R[name])
                     + '   exclusive ' + ', '.join(f'{r:.3f}' for r in sorted(set(R[name]) - set(R[other]))))
    lines.append('')
    hdr = f'{"group":>8} {"sd":>3} {"L=12":>6} {"L=96":>6} {"L=384":>6} {"grp-law":>8} {"kernel":>6}  verdict'
    lines += [hdr, '-' * 78]
    scored = violations = informative = 0
    for r in rows:
        other = [o for o in tabs if o != r['group']][0]
        flat, fail, verdict = classify(r, R[r['group']], R[other])
        scored += flat and fail
        violations += verdict.startswith('VIOLATES')
        # only a failure on a rung the other group lacks can tell the two lattices apart
        excl = set(R[r['group']]) - set(R[other])
        informative += verdict.startswith('on own') and any(abs(1.0 / r['kernel'] - x) < 1e-9 for x in excl)
        k = str(r['kernel'])
        lines.append(f'{r["group"]:>8} {r["seed"]:>3} {r["acc"]["12"]:>6.3f} {r["acc"]["96"]:>6.3f} '
                     f'{r["acc"]["384"]:>6.3f} {r["grp_law"]:>8.4f} {k:>6}  {verdict}')
    lines += ['', f'  runs {len(rows)}   flat failures scored {scored}   on an exclusive rung {informative}'
              f'   violations {violations}',
              '  VERDICT: ' + ('FALSIFIED' if violations else
                               'prediction holds' if scored else 'no flat failures, nothing tested')]
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(OUT, 'same_order.txt'), 'w') as f:
        f.write(text + '\n')


def main():
    shard, nshard = int(os.environ.get('SHARD', 0)), int(os.environ.get('NSHARD', 1))
    # ';' separates groups: 'SL(2,3)' itself contains a comma. Not named GROUPS: bash reserves that
    # name, so `GROUPS=... python` never reaches the environment.
    env = os.environ.get('SO_GROUPS')
    groups = env.split(';') if env else ['S_4', 'SL(2,3)']
    jobs = [(t, s) for t in (s4_table(), sl23_table()) if t[0] in groups for s in range(SEEDS)]
    assert jobs, f'SO_GROUPS={env!r} matches no group'
    tag = '' if not env else '_' + '_'.join(''.join(c for c in g if c.isalnum()) for g in groups)
    path = os.path.join(OUT, f'same_order_shard{shard}{tag}.txt')
    with open(path, 'w') as f:
        for i, ((name, tab), seed) in enumerate(jobs):
            if i % nshard != shard:
                continue
            t0 = time.time()
            acc, pred, true, gl = run(tab, seed)
            subs = {len(s): s for s in normal_subgroups(tab)}
            cc = coset_consistency(pred, true, tab, subs)
            kernel = min(k for k, v in cc.items() if v['score'] >= 0.95)
            row = {'group': name, 'seed': seed, 'acc': {str(L): acc[L] for L in EVAL_L},
                   'grp_law': gl, 'kernel': kernel,
                   'coset': {str(k): round(v['score'], 4) for k, v in cc.items()},
                   'steps': STEPS, 'secs': round(time.time() - t0)}
            msg = (f'{name:>8} {seed:>3} ' + ' '.join(f'{acc[L]:.3f}' for L in EVAL_L)
                   + f'  kernel {kernel}  ({row["secs"]}s)')
            print(msg, flush=True)
            f.write(msg + '\n#ROW ' + json.dumps(row) + '\n')
            f.flush()


if __name__ == '__main__':
    merge() if '--merge' in sys.argv else main()
