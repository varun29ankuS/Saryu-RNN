"""Matrix-with-contraction vs wide-vector-with-unbinding, at EQUAL state budget, trained.

    THE CONCLUSIONS DRAWN FROM THIS SCRIPT ARE RETRACTED. See the banner on
    results/matrix_decision.txt. Its nine logged runs scored delta 0.133/0.148/0.164 and
    hrr 0.039/0.039/0.031; the figures of 0.984 and 0.836 that were reported have no run
    behind them. Both cells here were also missing components their own references require --
    see ShortConv and unitary() below, and evidence/delta_reference.py for a matrix cell that
    actually reaches published behaviour (1.000, logged in runs/dref-*).


WHAT THE BASELINE ESTABLISHED. A 290k-parameter transformer solves 4-pair MQAR at 1.000 where our
387k-parameter recurrence has never exceeded 0.39 in 170 runs. So the task is solvable at this
budget and the wall is architectural. The derivation says why: our query's transport is a COMMON
FACTOR multiplying every stored item identically, so it cannot select, leaving only a diagonal
output gate to do a job that needs an item axis.

DeltaNet does not have that problem. Its read S q expands to sum_i v_i (k_i . q): the query's dot
product with each stored key weights that item, which is selection. And its transition
(I - beta k k^T) is RANK ONE -- it erases one direction and leaves every item whose key is
orthogonal to k untouched -- where ours rotates the entire state on every token.

THE CLAIM THIS TESTS. Our own construction said a wide vector with proper binding beats a matrix at
equal stored floats (0.993 vs 0.527 at 128 pairs). If that survives training, a vector state with an
unbinding read is cheaper than DeltaProduct at equal capacity, and it is the one claim in this
project we would still own. Constructions here have failed three times, so this is the trained test.

THREE CELLS, ALL WITH THE SAME NUMBER OF STATE FLOATS.
    delta    matrix S (dk x dv), rank-1 erase, outer-product write, read by contraction S q
    hrr      wide vector, unitary-key convolution binding, read by correlation conj(q) * h
    saryu    the shipped recurrence, as the control

REGISTERED PREDICTIONS, before running:
  P1  delta solves 4 pairs (> 0.8). It has the selecting read the baseline says is required, and
      the published results say it handles MQAR.
  P2  saryu does not, reproducing ~0.32.
  P3  hrr is the open one. If it matches delta, a vector state with an unbinding read is enough and
      the matrix is unnecessary. If it fails while delta succeeds, the MATRIX is doing the work,
      not the binding, and the vector route is dead.

FALSIFIER FOR THE VECTOR ROUTE: hrr lands at ~0.32 while delta solves. Then every wide-vector
result in this project was a construction artefact and the answer is the matrix state.

    python evidence/cell_shootout.py
Environment: NPAIRS, GAP, STEPS, LR, D, DSTATE, SEEDS, THREADS.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                     # noqa: E402
from saryu.metrics import Run                                        # noqa: E402
from binding_long import Muon                                        # noqa: E402

TARGET = int(os.environ.get('NPAIRS', 4))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 5000))
BS = 32
LR = float(os.environ.get('LR', 1e-3))
D = int(os.environ.get('D', 128))
DSTATE = int(os.environ.get('DSTATE', 1024))     # state floats, identical for delta and hrr
SEEDS = int(os.environ.get('SEEDS', 3))
NENT = 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


class ShortConv(nn.Module):
    """Depthwise causal convolution, kernel 4, over the sequence.

    THE COMPONENT THIS FILE WAS MISSING, and it is the one its own result file calls decisive.
    matrix_decision.txt: "The first DeltaNet here scored 0.13 because it had no short causal
    convolution -- the component arXiv 2609.16183 names as dominant... With it: 0.094 -> 0.961 at
    the same learning rate." That fix was never committed here, so the published script scored 0.07
    where the published result says 0.984.

    Why it is load-bearing for MQAR specifically: the cells derive k, v and q from the SAME token,
    but the task binds key token i to value token i+1. Without a conv mixing adjacent positions
    there is no path by which a key can ever meet its value, which is why the vector cell sat at
    exactly uniform loss, ln(64) = 4.159, at every learning rate."""

    def __init__(self, d, k=4):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, groups=d, padding=k - 1)
        self.k = k

    def forward(self, x):                                   # [B, L, D]
        return self.conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)


class DeltaCell(nn.Module):
    """S <- S(I - b k k^T) + b v k^T ;  o = S q.   State is dk*dv floats.

    The transition is RANK ONE: it removes only the k direction, so every stored item whose key is
    orthogonal to k survives untouched. That is the difference from a global rotation."""

    def __init__(self, d, dk, dv):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.sc = ShortConv(d)                                   # see ShortConv: was missing
        self.k = nn.Linear(d, dk)
        self.v = nn.Linear(d, dv)
        self.q = nn.Linear(d, dk)
        self.b = nn.Linear(d, 1)
        self.out = nn.Linear(dv, d)
        self.dk, self.dv = dk, dv

    def forward(self, x):
        z = self.sc(self.ln(x))
        B, L, _ = z.shape
        S = torch.zeros(B, self.dv, self.dk, device=z.device, dtype=z.dtype)
        outs = []
        for t in range(L):
            zt = z[:, t]
            k = F.normalize(self.k(zt), dim=-1)
            q = F.normalize(self.q(zt), dim=-1)
            v = self.v(zt)
            beta = torch.sigmoid(self.b(zt))                            # [B,1]
            Sk = torch.einsum('bvk,bk->bv', S, k)                       # what k currently holds
            S = (S - beta[..., None] * torch.einsum('bv,bk->bvk', Sk, k)
                   + beta[..., None] * torch.einsum('bv,bk->bvk', v, k))
            outs.append(torch.einsum('bvk,bk->bv', S, q))               # SELECTION by contraction
        return self.out(torch.stack(outs, 1))


def unitary(x):
    """Unit magnitude at every frequency, so the convolution inverse is exact.

    STRAIGHT-THROUGH, and this is load-bearing. Plain X/|X| annihilates the gradient with respect
    to the magnitude: the forward value is scale-invariant, so nothing upstream ever learns how big
    to make the key. The first version of this cell sat at chance loss for 5000 steps for exactly
    that reason. The straight-through estimator keeps the unit-magnitude forward value and passes
    the gradient through as if this were the identity, which is what took the cell to 0.836.

    This fix was applied when the result in matrix_decision.txt was measured, and was NOT committed
    into this file -- so the published script scored 0.023 where the published result says 0.836.
    Found on 2026-09-20 by rerunning it as the baseline of evidence/hrr_cleanup.py."""
    X = torch.fft.rfft(x, dim=-1)
    Xn = X / (X.abs() + 1e-6)
    return torch.fft.irfft(X + (Xn - X).detach(), n=x.shape[-1], dim=-1)


class HRRCell(nn.Module):
    """h <- a*h + g*(k conv v) ;  o = correlate(q, h).   State is dm floats."""

    def __init__(self, d, dm):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.sc = ShortConv(d)                                   # see ShortConv: was missing
        self.k = nn.Linear(d, dm)
        self.v = nn.Linear(d, dm)
        self.q = nn.Linear(d, dm)
        self.g = nn.Linear(d, 1)
        self.a = nn.Linear(d, 1)
        self.out = nn.Linear(dm, d)
        self.dm = dm

    def forward(self, x):
        z = self.sc(self.ln(x))
        B, L, _ = z.shape
        k = unitary(self.k(z))
        q = unitary(self.q(z))
        v = self.v(z)
        g = torch.sigmoid(self.g(z))
        a = torch.sigmoid(self.a(z))
        K = torch.fft.rfft(k, dim=-1)
        Q = torch.fft.rfft(q, dim=-1)
        V = torch.fft.rfft(v, dim=-1)
        h = torch.zeros(B, K.shape[-1], device=z.device, dtype=K.dtype)
        outs = []
        for t in range(L):
            h = a[:, t] * h + g[:, t] * (K[:, t] * V[:, t])             # bind, then accumulate
            outs.append(torch.fft.irfft(torch.conj(Q[:, t]) * h, n=self.dm, dim=-1))   # unbind
        return self.out(torch.stack(outs, 1))


class Net(nn.Module):
    def __init__(self, kind, vocab, d, dstate):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        dk = dv = int(math.isqrt(dstate))
        self.cell = DeltaCell(d, dk, dv) if kind == 'delta' else HRRCell(d, dstate)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, d))
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.emb(x)
        h = h + self.cell(h)
        h = h + self.ffn(h)
        return self.head(self.lnf(h))


def make(rng, n):
    e = rng.choice(NENT, size=2 * n, replace=False)
    ks, vs = e[:n], e[n:]
    rest = np.setdiff1d(np.arange(NENT), e)
    s = []
    for k, v in zip(ks, vs):
        s += [int(k), int(v)]
    s += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, n))
    s += [SEP, int(ks[i])]
    return s, int(vs[i])


def batch(rng, bs, n):
    o = [make(rng, n) for _ in range(bs)]
    return torch.tensor([a for a, _ in o]), torch.tensor([b for _, b in o])


def train(kind, seed):
    torch.manual_seed(seed)
    if kind == 'saryu':
        m = SaryuV3LM(NENT + 2, D, 2, nh=2, H=8)
        opt = Muon(list(m.parameters()), lr=3e-4)
    else:
        m = Net(kind, NENT + 2, D, DSTATE)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    npar = sum(p.numel() for p in m.parameters())
    rng = np.random.default_rng(1000 + seed)
    ev = np.random.default_rng(7)
    log = Run(f'cell-{kind}-p{TARGET}-s{seed}',
              config=dict(arm=f'cell/{kind}', pairs=TARGET, params=npar, state_floats=DSTATE,
                          steps=STEPS, seed=seed, chance_top1=round(1 / TARGET, 4)))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS, TARGET)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s == 0 or (s + 1) % 250 == 0:
            m.eval()
            with torch.no_grad():
                a = 0.
                for _ in range(4):
                    x2, y2 = batch(ev, BS, TARGET)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train()
            best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    return best, npar


def main():
    print(f'CELL SHOOTOUT  {TARGET} pairs, d={D}, {DSTATE} state floats for delta and hrr, '
          f'{STEPS} steps, {SEEDS} seeds, chance {1/TARGET:.3f}')
    print('transformer reference on this exact task: 1.000\n')
    print(f'{"cell":>8} {"params":>9} {"state":>7}  best per seed')
    print('-' * 52)
    t0 = time.time()
    for kind in ('delta', 'hrr', 'saryu'):
        r = [train(kind, s) for s in range(SEEDS)]
        print(f'{kind:>8} {r[0][1]:>9,} {DSTATE:>7}  '
              + ', '.join(f'{x:.3f}' for x, _ in r), flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ')
    print('  delta solves, hrr solves -> a vector state with an unbinding read is enough')
    print('  delta solves, hrr fails  -> the MATRIX does the work; the vector route is dead')
    print('  neither solves           -> a selecting read is not sufficient either')


if __name__ == '__main__':
    main()
