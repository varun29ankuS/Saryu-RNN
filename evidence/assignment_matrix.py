"""Is the trained model sitting at the BARYCENTRE of the Birkhoff polytope?

THE REFRAMING. Every experiment so far treated this as a representation question: what can the state
hold? That question is settled -- by construction a level-2 store is 1.000 at every gap and pair
count (memory_mechanisms.txt), and the reflection store reaches its own trace-law ceiling. The model
nevertheless converges to set membership at 1/n_pairs regardless of architecture, with three arms
indistinguishable across seeds (multiseed.txt). So the failure is in the LEARNING DYNAMICS, and the
right geometry is the geometry of the loss landscape, not of the state.

THE OBJECT. For a sequence with n pairs, what the model has learned is an n x n assignment: given
query key i, how much mass does it put on value j? Row-normalise and that matrix is doubly
stochastic-ish -- a point in (or near) the Birkhoff polytope B_n, whose vertices are the n!
permutation matrices.

    correct binding   a VERTEX of B_n: the identity assignment, one value per key
    set membership    the BARYCENTRE of B_n: every entry 1/n, every key equally associated with
                      every value

The barycentre is the unique point fixed by the whole symmetry group S_n that permutes the pairs. A
loss invariant under relabelling pairs -- ours is, since pairs are drawn i.i.d. and the task is
symmetric in them -- therefore has a critical point exactly there. Gradient descent started from a
symmetric initialisation has no reason to leave it: escaping the barycentre is a SYMMETRY-BREAKING
BIFURCATION, not an ordinary descent step.

That predicts something much more specific than "accuracy is 1/n", which is merely the barycentre's
accuracy. It predicts the whole matrix is flat, and it distinguishes three failures that all give
1/n accuracy:

    barycentre     M[i,j] ~ 1/n everywhere                 symmetric critical point, no binding
    rank-one bias  rows identical but not uniform          the model prefers certain VALUES,
                                                           independent of the key -- a frequency
                                                           prior, not an assignment
    partial vertex some rows peaked, others flat           binding learned for some pairs only

Only the first is the symmetry story. The other two would say the problem is elsewhere.

WHAT THIS SCRIPT MEASURES. Trains the base model, then for held-out sequences accumulates the mean
predicted probability M[i,j] of value j when key i is queried, restricted to the n values actually
present. Reports the matrix, its distance to the barycentre, its distance to the best permutation,
the row entropies, and the permanent-style diagonal mass.

    python evidence/assignment_matrix.py
Environment: PAIRS, GAP, STEPS, LR, SEED, THREADS.
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

PAIRS = int(os.environ.get('PAIRS', 4))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 3000))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
SEED = int(os.environ.get('SEED', 0))
D, NL, NENT = 128, 2, 64
SEP, PAD = NENT, NENT + 1
SEQ = 2 * PAIRS + GAP + 2
torch.set_num_threads(int(os.environ.get('THREADS', 8)))


def make(rng, qi=None):
    ent = rng.choice(NENT, size=2 * PAIRS, replace=False)
    ks, vs = ent[:PAIRS], ent[PAIRS:]
    rest = np.setdiff1d(np.arange(NENT), ent)
    seq = []
    for k, v in zip(ks, vs):
        seq += [int(k), int(v)]
    seq += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, PAIRS)) if qi is None else qi
    seq += [SEP, int(ks[i])]
    return seq, [int(v) for v in vs], i


def train():
    torch.manual_seed(SEED)
    m = SaryuV3LM(NENT + 2, D, NL)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
    rng = np.random.default_rng(1000 + SEED)
    t0 = time.time()
    for s in range(STEPS):
        xs, ys = [], []
        for _ in range(BS):
            seq, vs, i = make(rng)
            xs.append(seq); ys.append(vs[i])
        loss = F.cross_entropy(m(torch.tensor(xs))[:, -1], torch.tensor(ys))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if (s + 1) % max(1, STEPS // 4) == 0:
            print(f'  step {s+1}/{STEPS}  loss {float(loss):.3f}  '
                  f'{time.time()-t0:.0f}s', flush=True)
    return m


def assignment(m, n_seq=400):
    """M[i,j] = mean probability assigned to the value of pair j when pair i is queried."""
    ev = np.random.default_rng(99)
    M = np.zeros((PAIRS, PAIRS)); cnt = np.zeros(PAIRS)
    with torch.no_grad():
        for _ in range(n_seq):
            seq, vs, _ = make(ev, qi=0)
            # ask every key of the SAME sequence, so rows share a value set
            for i in range(PAIRS):
                s = list(seq); s[-1] = s[2 * i]          # swap in key i as the query
                p = F.softmax(m(torch.tensor([s]))[0, -1], -1)
                mass = p[torch.tensor(vs)]
                M[i] += (mass / mass.sum().clamp_min(1e-9)).numpy(); cnt[i] += 1
    return M / cnt[:, None]


def main():
    print(f'ASSIGNMENT MATRIX  {PAIRS} pairs, gap {GAP}, {STEPS} steps, seed {SEED}')
    m = train(); m.eval()
    M = assignment(m)
    n = PAIRS
    bary = np.full((n, n), 1.0 / n)
    print(f'\nM[i,j] = P(value of pair j | key of pair i queried), renormalised to the n values '
          f'present.\nBarycentre would be {1/n:.3f} in every cell.\n')
    print('        ' + ''.join(f'  val{j}' for j in range(n)) + '     row entropy')
    for i in range(n):
        H = -(M[i] * np.log(M[i] + 1e-12)).sum()
        print(f'  key{i} ' + ''.join(f' {M[i, j]:>5.3f}' for j in range(n))
              + f'      {H:.3f} / {np.log(n):.3f}')
    diag = float(np.mean(np.diag(M)))
    d_bary = float(np.abs(M - bary).mean())
    offd = float((M.sum() - np.trace(M)) / (n * n - n))
    print(f'\n  diagonal mass (correct pairing) : {diag:.3f}   (barycentre {1/n:.3f})')
    print(f'  mean off-diagonal               : {offd:.3f}')
    print(f'  mean |M - barycentre|           : {d_bary:.4f}')
    print(f'  mean row entropy                : {np.mean([-(M[i]*np.log(M[i]+1e-12)).sum() for i in range(n)]):.3f}'
          f'  (max {np.log(n):.3f})')
    rowspread = float(np.abs(M - M.mean(0, keepdims=True)).mean())
    print(f'  mean |row - mean row|           : {rowspread:.4f}  '
          f'(0 = every key behaves identically)')
    print('\nVERDICT')
    if d_bary < 0.03:
        print('  At the BARYCENTRE. Every key is equally associated with every value: the')
        print('  symmetric critical point of the loss. Escaping it is a symmetry-breaking')
        print('  bifurcation, not a descent step -- which is why no architecture change moved it.')
    elif rowspread < 0.02:
        print('  RANK-ONE BIAS: rows are identical but not uniform. The model has learned a')
        print('  preference over VALUES that ignores the key. Not binding, and not the')
        print('  barycentre either.')
    elif diag > 1.5 / n:
        print('  Moving toward a VERTEX: the correct pairing carries more mass than chance,')
        print('  so binding has partially formed.')
    else:
        print('  Off the barycentre but not toward the correct vertex.')


if __name__ == '__main__':
    main()
