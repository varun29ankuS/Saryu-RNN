"""The lattice law names a SUBGROUP, not just an order. Q_8 is where that can be tested.

THE GAP THIS CLOSES. Paper B shows accuracies landing on 1/|N| and calls that the lattice law. But
1/|N| depends only on the ORDER of N, so every result so far is consistent with a weaker claim: that
the model collapses by some factor, and the factor happens to divide |G|. To separate the two you
need a group with several distinct normal subgroups of the SAME order, so that the accuracy cannot
tell them apart and only the errors can.

Q_8 is exactly that group. It is Hamiltonian -- every subgroup is normal -- and it has THREE distinct
subgroups of order 4:

    <i> = {1, -1, i, -i}      <j> = {1, -1, j, -j}      <k> = {1, -1, k, -k}

All three give the same rung, 1/4 = 0.250. A model that lands there has learned Q_8/<x> for exactly
one x, and coset consistency should say which.

THE SHARP PART: the wrong subgroups have a PREDICTED score, and it is neither 1 nor chance. Suppose
the model learned Q_8/<i>: it knows which coset of <i> the answer lies in and guesses uniformly among
those 4 elements. Score that against <j>. Whichever 4-element <i>-coset the truth lies in, it splits
exactly half-and-half across the two cosets of <j>, so the model is right about <j> half the time:

    correct subgroup   1.000        the other two order-4 subgroups   0.500
    Z_2 = {1,-1}       0.500        Q_8 itself                        1.000
    trivial {1}        0.250        (= the raw accuracy)

REGISTERED PREDICTIONS, before running. For every run landing on the 0.250 rung:
  P1  exactly ONE order-4 subgroup scores above 0.95.
  P2  the other two score within 0.10 of 0.500 -- not chance (0.25), not perfect.
  P3  across seeds, the identified subgroup is not always the same one, i.e. the model is not
      picking a fixed subgroup because of how the group was indexed.

FALSIFIER: a 0.250-rung run where two or more order-4 subgroups score above 0.95, or where none
does. Either would mean the accuracy is a collapse factor rather than a quotient by a named subgroup.

    python evidence/which_subgroup.py
Environment: SEEDS, STEPS, NH, D, LAM, THREADS.
"""
from __future__ import annotations

import itertools
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

NH = int(os.environ.get('NH', 4))
D = int(os.environ.get('D', 4))
STEPS = int(os.environ.get('STEPS', 2000))
BS, TRAIN_L = 256, 12
LAM = float(os.environ.get('LAM', 1.0))
SEEDS = [int(s) for s in os.environ.get('SEEDS', ','.join(str(i) for i in range(10))).split(',')]
torch.set_num_threads(int(os.environ.get('THREADS', 8)))

# Q_8 as index = base + 4*sign, base 0,1,2,3 = 1,i,j,k and sign 0,1 = +,-.
NAMES = ['1', 'i', 'j', 'k', '-1', '-i', '-j', '-k']
_BM = [[(0, 0), (1, 0), (2, 0), (3, 0)], [(1, 0), (0, 1), (3, 0), (2, 1)],
       [(2, 0), (3, 1), (0, 1), (1, 0)], [(3, 0), (2, 0), (1, 1), (0, 1)]]
TABLE = [[0] * 8 for _ in range(8)]
for _x in range(8):
    for _y in range(8):
        _rb, _f = _BM[_x % 4][_y % 4]
        TABLE[_x][_y] = _rb + 4 * ((_x // 4 + _y // 4 + _f) % 2)
TAB = torch.tensor(TABLE)
N_G = 8
IDENT = next(e for e in range(N_G) if all(TABLE[e][x] == x for x in range(N_G)))
INV = [next(y for y in range(N_G) if TABLE[x][y] == IDENT) for x in range(N_G)]


def subgroups():
    """Every subgroup, found by closure, with normality checked rather than assumed."""
    out = []
    for r in range(1, N_G + 1):
        for sub in itertools.combinations(range(N_G), r):
            S = set(sub)
            if IDENT not in S:
                continue
            if all(TABLE[a][b] in S for a in S for b in S) and all(INV[a] in S for a in S):
                normal = all(TABLE[TABLE[g][a]][INV[g]] in S for g in S for a in S
                             for g in range(N_G))
                if normal:
                    out.append(frozenset(S))
    return sorted(set(out), key=lambda s: (len(s), sorted(s)))


def label(S):
    if len(S) == 1:
        return '{1}'
    if len(S) == N_G:
        return 'Q_8'
    gens = [NAMES[g] for g in sorted(S) if NAMES[g] not in ('1', '-1')]
    return '<' + (gens[0] if gens else '-1') + '>'


class Reflect(nn.Module):
    """Paper B Section 2: T_g = prod_i (I - beta_{g,i} u u^T), linear readout."""

    def __init__(self, n, d):
        super().__init__()
        self.v = nn.Parameter(torch.randn(n, NH, d) * 0.5)
        self.b = nn.Parameter(torch.full((n, NH), 2.0))
        self.h0 = nn.Parameter(torch.randn(d) * 0.3)
        self.out = nn.Linear(d, n)
        self.d = d

    def T(self, h, tok):
        V, B = self.v[tok], self.b[tok]
        for i in range(NH):
            u = F.normalize(V[:, i], dim=-1)
            h = h - B[:, i:i + 1] * (h * u).sum(-1, keepdim=True) * u
        return h

    def forward(self, x):
        h = self.h0[None].expand(x.shape[0], -1)
        o = []
        for t in range(x.shape[1]):
            h = self.T(h, x[:, t]); o.append(h)
        return self.out(torch.stack(o, 1))


def batch(bs, L, gen):
    x = torch.randint(0, N_G, (bs, L), generator=gen)
    acc = x[:, 0].clone()
    ys = [acc.clone()]
    for t in range(1, L):
        acc = TAB[acc, x[:, t]]
        ys.append(acc.clone())
    return x, torch.stack(ys, 1)


def run(seed, subs):
    torch.manual_seed(seed)
    m = Reflect(N_G, D)
    op = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(500 + seed)
    for _ in range(STEPS):
        x, y = batch(BS, TRAIN_L, g)
        hh = torch.randn(64, D)
        a = torch.randint(0, N_G, (64,)); b = torch.randint(0, N_G, (64,))
        law = (m.T(m.T(hh, a), b) - m.T(hh, TAB[a, b])).norm(dim=1).mean() / D ** 0.5
        loss = F.cross_entropy(m(x).reshape(-1, N_G), y.reshape(-1)) + LAM * law
        op.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); op.step()

    m.eval()
    with torch.no_grad():
        accs = {}
        for L in (12, 96):
            ge = torch.Generator().manual_seed(7 + L)
            x, y = batch(512, L, ge)
            accs[L] = float((m(x)[:, -1].argmax(-1) == y[:, -1]).float().mean())
        ge = torch.Generator().manual_seed(99)
        x, y = batch(512, 96, ge)
        pred, true = m(x)[:, -1].argmax(-1), y[:, -1]
        hh = torch.randn(512, D)
        a = torch.randint(0, N_G, (512,)); b = torch.randint(0, N_G, (512,))
        viol = float((m.T(m.T(hh, a), b) - m.T(hh, TAB[a, b])).norm(dim=1).mean() / D ** 0.5)
    cos = {}
    for S in subs:
        # p and t lie in the same left coset of S iff t^{-1} p is in S
        cos[label(S)] = float(sum(1 for p, t in zip(pred, true)
                                  if TABLE[INV[int(t)]][int(p)] in S) / len(true))
    return accs, viol, cos


def main():
    subs = subgroups()
    order4 = [S for S in subs if len(S) == 4]
    print('WHICH SUBGROUP, NOT JUST WHICH ORDER:  Q_8')
    print(f'n_h={NH} d={D} {STEPS} steps, {len(SEEDS)} seeds\n')
    print(f'Normal subgroups found by closure ({len(subs)}), all normal since Q_8 is Hamiltonian:')
    for S in subs:
        print(f'   {label(S):>5}  order {len(S)}  rung {1/len(S):.3f}   '
              f'{{{", ".join(NAMES[g] for g in sorted(S))}}}')
    print(f'\nThe three order-4 subgroups share the rung 0.250, so accuracy cannot separate them.')
    print('Predicted for a 0.250 run: correct 1.000, other two order-4 0.500, <-1> 0.500.\n')
    cols = [label(S) for S in subs]
    hdr = f'{"sd":>3} {"L=12":>6} {"L=96":>6} {"g-law":>7} ' + ''.join(f'{c:>8}' for c in cols)
    print(hdr); print('-' * len(hdr))
    rows = []
    for sd in SEEDS:
        t0 = time.time()
        accs, viol, cos = run(sd, subs)
        rows.append((sd, accs, viol, cos))
        print(f'{sd:>3} {accs[12]:>6.3f} {accs[96]:>6.3f} {viol:>7.4f} '
              + ''.join(f'{cos[c]:>8.3f}' for c in cols) + f'   ({time.time()-t0:.0f}s)', flush=True)

    o4 = [label(S) for S in order4]
    onrung = [r for r in rows if abs(r[1][96] - 0.25) < 0.04 and r[2] < 0.02]
    print(f'\nRuns on the 0.250 rung that are also homomorphisms (g-law < 0.02): {len(onrung)}')
    ok1 = ok2 = 0
    picked = []
    for sd, accs, viol, cos in onrung:
        hi = [c for c in o4 if cos[c] > 0.95]
        rest = [cos[c] for c in o4 if c not in hi]
        good1 = len(hi) == 1
        good2 = all(abs(v - 0.5) < 0.10 for v in rest)
        ok1 += good1; ok2 += good2
        picked += hi
        print(f'  sd{sd}: identified {hi if hi else "NONE"}, others '
              f'{[round(v,3) for v in rest]}  '
              f'-> P1 {"ok" if good1 else "FAIL"}, P2 {"ok" if good2 else "FAIL"}')
    if onrung:
        print(f'\n  P1 exactly one order-4 subgroup above 0.95: {ok1}/{len(onrung)}')
        print(f'  P2 the other two within 0.10 of 0.500:      {ok2}/{len(onrung)}')
        print(f'  P3 subgroups actually chosen across seeds: {sorted(set(picked))} '
              f'({len(set(picked))} distinct)')
        print(f'\n  VERDICT: {"NOT falsified" if ok1 == len(onrung) else "FALSIFIED"}')
    else:
        print('  No run landed on the 0.250 rung as a homomorphism; nothing to score.')


if __name__ == '__main__':
    main()
