"""Does assembling the measured choices actually beat the default parameterisation?

Every ingredient was measured in isolation this session. This asks the only question that
matters for calling Saryu an architecture: do they compose?

    baseline   fla-style defaults -- beta = 2*sigmoid(0) = 1.0 (a PROJECTION), independent
               householder init, MLP readout. This is not a strawman: it is how
               GatedDeltaProduct initialises, with b_proj at xavier gain 2**-2.5 so its
               output starts near zero.
    saryu      beta init 2.0 (reflection), correlated householder init, LINEAR readout
    saryu+aux  the above plus the table-free kernel loss

Task: the S_4 word problem, trained at L=12, evaluated out to L=96 (8x). Chance 0.042.
Rungs 1.000 / 0.250 / 0.083 / 0.042 -- failures land ON them, so the nearest rung says
which quotient a run reached rather than just how badly it did.

REFERENCES
  [Siems et al. 2025] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products, arXiv 2502.10297
"""
from __future__ import annotations

import itertools
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import SaryuLayer
from diagnose import character_norm, group_law_error

torch.set_num_threads(4)

K = 4
ELS = list(itertools.permutations(range(K)))
IX = {p: i for i, p in enumerate(ELS)}
TAB = torch.tensor([[IX[tuple(b[a[i]] for i in range(K))] for b in ELS] for a in ELS])
N = len(ELS)
RUNGS = [1.0, 0.25, 1 / 12, 1 / 24]


class Baseline(nn.Module):
    """fla-style: beta starts at 1.0 (projection), independent init, MLP readout."""

    def __init__(self, dim=4, n_h=3):
        super().__init__()
        self.v = nn.Parameter(torch.randn(N, n_h, dim) * 0.5)
        self.raw = nn.Parameter(torch.zeros(N, n_h))        # 2*sigmoid(0) = 1.0
        self.h0 = nn.Parameter(torch.randn(dim) * 0.3)
        self.readout = nn.Sequential(nn.Linear(dim, 128), nn.GELU(),
                                     nn.Linear(128, 128), nn.GELU(),
                                     nn.Linear(128, N))
        self.dim, self.n_h = dim, n_h
        self.transport = self
        self.input_dependent = False

    def step(self, state, tok):
        v, raw = self.v[tok], self.raw[tok]
        beta = 2.0 * torch.sigmoid(raw)
        for i in range(self.n_h):
            u = F.normalize(v[..., i, :], dim=-1)
            state = state - beta[..., i:i + 1] * (state * u).sum(-1, keepdim=True) * u
        return state

    def states(self, tokens):
        s = self.h0.expand(tokens.shape[0], self.dim)
        out = []
        for t in range(tokens.shape[1]):
            s = self.step(s, tokens[:, t])
            out.append(s)
        return torch.stack(out, 1)

    def forward(self, tokens):
        return self.readout(self.states(tokens))


def batch(bs, L, g):
    x = torch.randint(0, N, (bs, L), generator=g)
    acc = x[:, 0].clone()
    ys = [acc.clone()]
    for t in range(1, L):
        acc = TAB[acc, x[:, t]]
        ys.append(acc.clone())
    return x, torch.stack(ys, 1)


def hom_loss(model, n=64):
    h = torch.randn(n, model.dim)
    a = torch.randint(0, N, (n,))
    b = torch.randint(0, N, (n,))
    lhs = model.transport.step(model.transport.step(h, a), b)
    rhs = model.transport.step(h, TAB[a, b])
    return (lhs - rhs).norm(dim=1).mean() / model.dim ** 0.5


def run(arm, seed, steps=2500):
    torch.manual_seed(seed)
    if arm == 'baseline':
        m = Baseline()
    else:
        m = SaryuLayer(dim=4, n_classes=N, n_h=3, vocab=N,
                       correlation=0.5, beta_init=2.0)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(500 + seed)
    for _ in range(steps):
        x, y = batch(256, 12, g)
        loss = F.cross_entropy(m(x).reshape(-1, N), y.reshape(-1)) + hom_loss(m)
        if arm == 'saryu+aux':
            loss = loss + m.kernel_loss()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    ge = torch.Generator().manual_seed(99)
    accs = []
    with torch.no_grad():
        for L in (12, 96):
            a = 0.0
            for _ in range(4):
                x, y = batch(256, L, ge)
                a += (m(x)[:, -1].argmax(-1) == y[:, -1]).float().mean().item()
            accs.append(a / 4)
        cn = character_norm(m, N)
        raw = m.transport.raw
        beta = float((2.0 * torch.sigmoid(raw)).mean()
                     if getattr(m.transport, 'beta_param', 'sigmoid') == 'sigmoid'
                     else raw.mean())
    return accs, group_law_error(m, TAB), cn, beta


if __name__ == '__main__':
    print('SARYU vs THE DEFAULT PARAMETERISATION', flush=True)
    print(flush=True)
    print(f'  S_{K} word problem, train L=12, eval L=96 (8x). chance {1/N:.3f}', flush=True)
    print('  rungs 1.000 / 0.250 / 0.083 / 0.042; <chi,chi> = 2.00 iff faithful', flush=True)
    print(flush=True)
    print(f'{"arm":>10} {"sd":>3} {"L=12":>8} {"L=96":>8} {"rung":>7} '
          f'{"grp-law":>8} {"<chi,chi>":>10} {"beta":>6}', flush=True)
    print('-' * 68, flush=True)
    summary = {}
    for arm in ['baseline', 'saryu', 'saryu+aux']:
        solved = 0
        for seed in range(6):
            t0 = time.time()
            accs, gl, cn, beta = run(arm, seed)
            solved += accs[-1] > 0.9
            near = min(RUNGS, key=lambda v: abs(v - accs[-1]))
            print(f'{arm:>10} {seed:>3} {accs[0]:>8.3f} {accs[1]:>8.3f} {near:>7.3f} '
                  f'{gl:>8.4f} {cn:>10.3f} {beta:>6.3f}   ({time.time()-t0:.0f}s)',
                  flush=True)
        summary[arm] = solved
        print(f'{arm:>10}  == {solved}/6 solved at 8x length', flush=True)
        print(flush=True)
    print('  ' + '  |  '.join(f'{k}: {v}/6' for k, v in summary.items()), flush=True)
