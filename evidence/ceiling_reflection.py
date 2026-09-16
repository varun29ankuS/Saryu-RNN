"""Does the REFLECTION transport clear the abelian ceiling? Measured, not inherited.

WHY THIS EXISTS. An earlier solvability ladder in this project (log
evidence/results/abelian_ladder.txt) reports a column labelled "Saryu" that clears the ceiling on
S_3 at 0.778. That column is NOT a reflection transport: the script behind it builds the transport
from alternating Givens pairings (a butterfly) and reads out through a 3-layer MLP, and its own
banner says so -- "Saryu (butterfly + curvature)". Quoting it as evidence about Householder
reflections would be wrong, and it was quoted that way in a draft of paper B before this existed.

So we measure the thing the paper is about: the exact transport of paper B Section 2 -- one learned
reflection table per group element, n_h reflections per element, beta free, LINEAR readout, the
group-law penalty -- at the length where the ceiling is proven.

THE COMPARISON. The abelian ceiling is a theorem, not a baseline: a transport whose elements commute
cannot exceed it at any width, depth or training budget. At L = 8 it is exact by enumeration
(recomputed in evidence/verify_paperB.py):

    Z_6   1.0000   abelian, so the multiset already determines the product
    S_3   0.3850   matched to Z_6 in order, so only NON-COMMUTATIVITY can separate them
    Q_8   0.5323   a second group, because clearing the bound on one group is a thin result

Q_8 matters for a second reason. Its normal subgroups have orders 1, 2, 4, 4, 4, 8, so its rungs are
1.000, 0.500, 0.250, 0.125 -- and 0.500 sits just BELOW the 0.5323 ceiling. A run that learns the
Z_2 quotient therefore lands a hair under a bound it did not have to obey, which is a place where
the rung structure and the ceiling can be told apart.

    python evidence/ceiling_reflection.py
Environment: NH, D, STEPS, SEEDS, LAM, L.
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
L = int(os.environ.get('L', 8))
STEPS = int(os.environ.get('STEPS', 2000))
BS = 256
LAM = float(os.environ.get('LAM', 1.0))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
torch.set_num_threads(int(os.environ.get('THREADS', 8)))

# All three are EXACT, by enumeration over every word of length 8 (verify_paperB.py part 1):
# 1287 multisets for the order-6 groups, 6435 for Q_8. The larger groups on the old ladder are
# Monte-Carlo estimates and are deliberately not compared against here.
CEILING = {'S_3': 0.3850, 'Z_6': 1.0000, 'Q_8': 0.5323}
CHANCE = {'S_3': 1 / 6, 'Z_6': 1 / 6, 'Q_8': 1 / 8}


def groups():
    p3 = list(itertools.permutations(range(3)))
    ix = {p: i for i, p in enumerate(p3)}
    s3 = [[ix[tuple(b[a[i]] for i in range(3))] for b in p3] for a in p3]
    z6 = [[(a + b) % 6 for b in range(6)] for a in range(6)]
    # Q_8 by its multiplication table: {+-1, +-i, +-j, +-k} as 0..7, the same construction
    # verify_paperB.py uses to enumerate the normal subgroups and the ceiling.
    bm = [[(0, 0), (1, 0), (2, 0), (3, 0)], [(1, 0), (0, 1), (3, 0), (2, 1)],
          [(2, 0), (3, 1), (0, 1), (1, 0)], [(3, 0), (2, 0), (1, 1), (0, 1)]]
    q8 = [[0] * 8 for _ in range(8)]
    for x in range(8):
        for y in range(8):
            rb, f = bm[x % 4][y % 4]
            q8[x][y] = rb + 4 * ((x // 4 + y // 4 + f) % 2)
    return {'S_3': torch.tensor(s3), 'Z_6': torch.tensor(z6), 'Q_8': torch.tensor(q8)}


class Reflect(nn.Module):
    """Paper B Section 2 exactly: T_g = prod_i (I - beta_{g,i} u u^T), linear readout."""

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
            h = self.T(h, x[:, t])
            o.append(h)
        return self.out(torch.stack(o, 1))


def run(name, tab, seed):
    n = tab.shape[0]
    torch.manual_seed(seed)
    m = Reflect(n, D)
    op = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(500 + seed)

    def batch(bs, gen):
        x = torch.randint(0, n, (bs, L), generator=gen)
        acc = x[:, 0].clone()
        ys = [acc.clone()]
        for t in range(1, L):
            acc = tab[acc, x[:, t]]
            ys.append(acc.clone())
        return x, torch.stack(ys, 1)

    for _ in range(STEPS):
        x, y = batch(BS, g)
        hh = torch.randn(64, D)
        a = torch.randint(0, n, (64,)); b = torch.randint(0, n, (64,))
        law = (m.T(m.T(hh, a), b) - m.T(hh, tab[a, b])).norm(dim=1).mean() / D ** 0.5
        loss = F.cross_entropy(m(x).reshape(-1, n), y.reshape(-1)) + LAM * law
        op.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); op.step()

    ge = torch.Generator().manual_seed(99)
    with torch.no_grad():
        hit = tot = 0
        for _ in range(4):
            x, y = batch(256, ge)
            hit += (m(x)[:, -1].argmax(-1) == y[:, -1]).sum().item(); tot += len(y)
        hh = torch.randn(256, D)
        a = torch.randint(0, n, (256,)); b = torch.randint(0, n, (256,))
        law = float((m.T(m.T(hh, a), b) - m.T(hh, tab[a, b])).norm(dim=1).mean() / D ** 0.5)
    return hit / tot, law


def main():
    print(f'REFLECTION TRANSPORT vs THE ABELIAN CEILING | n_h={NH} d={D} L={L} '
          f'{STEPS} steps | seeds {SEEDS}')
    print('The ceiling is proven: no commuting transport can exceed it at any size.\n')
    print(f'{"group":>6} {"ceiling":>9} {"seed":>5} {"accuracy":>9} {"group-law":>10}  verdict')
    print('-' * 62)
    G = groups()
    for name, tab in G.items():
        accs = []
        for sd in SEEDS:
            t0 = time.time()
            acc, law = run(name, tab, sd)
            accs.append(acc)
            v = ('ABOVE ceiling' if acc > CEILING[name] + 0.03 else
                 'at ceiling' if acc > CEILING[name] - 0.03 else 'below')
            v += f'  (chance {CHANCE[name]:.3f})' if acc < CHANCE[name] + 0.03 else ''
            print(f'{name:>6} {CEILING[name]:>9.4f} {sd:>5} {acc:>9.3f} {law:>10.4f}  {v}'
                  f'   ({time.time()-t0:.0f}s)', flush=True)
        mu = sum(accs) / len(accs)
        sd_ = (sum((a - mu) ** 2 for a in accs) / max(1, len(accs) - 1)) ** 0.5
        print(f'{name:>6} {"mean":>9} {"":>5} {mu:>9.3f} +- {sd_:.3f}\n')
    print('A reflection transport is non-commuting, so the ceiling does not bind it. This')
    print('measures whether training actually reaches above the bound, which is a separate')
    print('question from whether the architecture permits it.')


if __name__ == '__main__':
    main()
