"""An activation for beta where 2.0 is EXACTLY attainable and STICKY.

The geometric parameterisation (the one that should have been obvious):
    I - beta u u^T acts on span(u) by multiplication by (1 - beta), so
        beta = 0 -> +1 identity,  beta = 1 -> 0 projection,  beta = 2 -> -1 REFLECTION.
    A reflection is a rotation by PI in that eigendirection. Setting 1 - beta = cos(theta):

        beta = 1 - cos(theta)

    * beta = 2 EXACTLY at theta = pi (sigmoid never gets there),
    * d beta / d theta = sin(theta) = 0 at theta = pi, so the reflection is a STATIONARY
      point -- sticky, like the clamp but smooth rather than a hand-imposed box,
    * range [0, 2] by construction, so beta cannot run away.
    Under this view today's drift is not "beta decayed" but theta falling short of pi:
    the transport rotates by less than a half turn and fails to close.

Today's measurements say beta = 2 is never a stable place to be:

    sigmoid   beta = 2*sigmoid(x) reaches 2 only ASYMPTOTICALLY -- a true reflection is
              unreachable. beta_act.py measured that as 0/4 vs 2/4.
    free      beta = 2 is attainable but has no reason to STAY. Every learned run today
              drifted: 1.294 (n_h=3), 1.619 (n_h=2), 1.932 (n_h=4).
    frozen    beta = 2 exactly, and the results are the best in the sweep where the parity
              selection rule is satisfiable (S_4 n_h=2: group-law 0.0045, det +1.000,
              3/4 solved) and catastrophic where it is not (n_h=3: group-law 0.978, 0/4).

So the regime that wins depends on whether det = (-1)^n_h is legal, and we would rather
not have to know in advance. A hard sigmoid gives that:

    beta = 2 * clamp(x, 0, 1)

  * beta = 2 is reached EXACTLY at x >= 1 (unlike sigmoid), and
  * the gradient there is ZERO, so once orthogonal the model STAYS orthogonal
    (unlike free), while
  * below saturation beta slides freely, so it can escape the obstruction when the
    selection rule forbids a reflection (which is what A_4 at odd n_h needs).

PREDICTION, before running:
    even n_h -> clamp saturates at 2 and matches FROZEN (low group-law, det +1)
    odd  n_h -> clamp slides OFF 2 and matches LEARNED, avoiding frozen's 0/4 collapse
    i.e. clamp >= max(frozen, learned) on solve rate, without being told which regime.
"""
from __future__ import annotations

import itertools
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transport'))
from diagnose import character_norm, group_law_error  # noqa: E402

torch.set_num_threads(4)


def make_group(name):
    if name == 'S_4':
        els = sorted(itertools.permutations(range(4)))
    elif name == 'A_4':
        def par(p):
            return sum(p[i] > p[j] for i in range(4) for j in range(i + 1, 4)) % 2
        els = [p for p in sorted(itertools.permutations(range(4))) if par(p) == 0]
    else:
        raise ValueError(name)
    ix = {p: i for i, p in enumerate(els)}
    k = len(els[0])
    T = torch.tensor([[ix[tuple(b[a[i]] for i in range(k))] for b in els] for a in els])
    return T, len(els)


def hh_init(n_tok, n_h, dim, corr=0.5):
    v = F.normalize(torch.randn(n_tok, n_h, dim), dim=-1)
    if n_h > 1 and corr and dim > 1:
        a, rest = v[:, :1], v[:, 1:]
        perp = F.normalize(rest - (rest * a).sum(-1, keepdim=True) * a, dim=-1)
        v = torch.cat([a, corr * a + math.sqrt(max(0.0, 1 - corr ** 2)) * perp], 1)
    return v


class Saryu(nn.Module):
    """mode: free | sigmoid | clamp | frozen"""

    def __init__(self, N, dim, n_h, mode):
        super().__init__()
        self.N, self.dim, self.n_h, self.mode = N, dim, n_h, mode
        self.v = nn.Parameter(hh_init(N, n_h, dim, 0.5))
        if mode == 'free':
            raw0 = torch.full((N, n_h), 2.0)
        elif mode == 'sigmoid':
            raw0 = torch.full((N, n_h), 0.0)          # 2*sigmoid(0) = 1.0
        elif mode == 'clamp':
            raw0 = torch.full((N, n_h), 1.0)          # saturated at beta = 2 exactly
        elif mode == 'cos':
            raw0 = torch.full((N, n_h), math.pi)      # theta = pi  ->  beta = 2 exactly
        elif mode == 'frozen':
            raw0 = torch.full((N, n_h), 2.0)
        else:
            raise ValueError(mode)
        self.raw = nn.Parameter(raw0, requires_grad=(mode != 'frozen'))
        self.h0 = nn.Parameter(torch.randn(dim) * 0.3)
        self.readout = nn.Linear(dim, N)
        self.beta_param = 'free'
        self.transport = self
        self.input_dependent = False

    def beta(self, tok):
        r = self.raw[tok]
        if self.mode == 'sigmoid':
            return 2.0 * torch.sigmoid(r)
        if self.mode == 'clamp':
            return 2.0 * r.clamp(0.0, 1.0)
        if self.mode == 'cos':
            return 1.0 - torch.cos(r)
        return r

    def step(self, s, tok):
        v, beta = self.v[tok], self.beta(tok)
        for i in range(self.n_h):
            u = F.normalize(v[..., i, :], dim=-1)
            s = s - beta[..., i:i + 1] * (s * u).sum(-1, keepdim=True) * u
        return s

    def forward(self, tokens):
        s = self.h0.expand(tokens.shape[0], self.dim)
        out = []
        for t in range(tokens.shape[1]):
            s = self.step(s, tokens[:, t])
            out.append(s)
        return self.readout(torch.stack(out, 1))

    def beta_now(self):
        with torch.no_grad():
            return float(self.beta(torch.arange(self.N)).mean())

    def frac_at_two(self, tol=1e-4):
        with torch.no_grad():
            b = self.beta(torch.arange(self.N))
            return float((b > 2.0 - tol).float().mean())


def batch(TAB, N, bs, L, g):
    x = torch.randint(0, N, (bs, L), generator=g)
    acc = x[:, 0].clone()
    ys = [acc.clone()]
    for t in range(1, L):
        acc = TAB[acc, x[:, t]]
        ys.append(acc.clone())
    return x, torch.stack(ys, 1)


def hom_loss(m, TAB, n=64):
    h = torch.randn(n, m.dim)
    a = torch.randint(0, m.N, (n,)); b = torch.randint(0, m.N, (n,))
    return (m.step(m.step(h, a), b) - m.step(h, TAB[a, b])).norm(dim=1).mean() / m.dim ** 0.5


def det_mean(m):
    with torch.no_grad():
        eye = torch.eye(m.dim)
        cols = [m.step(eye[j].unsqueeze(0).expand(m.N, m.dim), torch.arange(m.N))
                for j in range(m.dim)]
        return float(torch.det(torch.stack(cols, -1)).mean())


def run(gname, dim, n_h, mode, seed, steps=2000):
    TAB, N = make_group(gname)
    torch.manual_seed(seed)
    m = Saryu(N, dim, n_h, mode)
    ps = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=2e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(500 + seed)
    for _ in range(steps):
        x, y = batch(TAB, N, 256, 12, g)
        loss = F.cross_entropy(m(x).reshape(-1, N), y.reshape(-1)) + hom_loss(m, TAB)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
    ge = torch.Generator().manual_seed(99)
    with torch.no_grad():
        a = 0.0
        for _ in range(4):
            x, y = batch(TAB, N, 256, 96, ge)
            a += (m(x)[:, -1].argmax(-1) == y[:, -1]).float().mean().item()
    return (a / 4, m.beta_now(), m.frac_at_two(), det_mean(m),
            group_law_error(m, TAB), character_norm(m, N))


if __name__ == '__main__':
    print('BETA ACTIVATION: make 2.0 exactly attainable AND sticky')
    print('  clamp: beta = 2*clamp(x,0,1) -- exact at x>=1, zero gradient there')
    print('  cos:   beta = 1-cos(theta)   -- exact at theta=pi, d beta/d theta = 0 there')
    print('  prediction: both saturate at 2 when parity allows, slide off when it does not')
    print()
    for gname in ('S_4', 'A_4'):
        print(f'### {gname}  d=3')
        print(f'{"mode":>9} {"n_h":>4} {"beta":>7} {"%at2":>6} {"det":>7} {"solved":>7} '
              f'{"grp-law":>8} {"<chi,chi>":>10}   seeds')
        print('-' * 88)
        for n_h in (2, 3):
            for mode in ('sigmoid', 'free', 'clamp', 'cos', 'frozen'):
                rows = [run(gname, 3, n_h, mode, s) for s in range(4)]
                accs = [r[0] for r in rows]
                solved = sum(a > 0.9 for a in accs)
                print(f'{mode:>9} {n_h:>4} {np.mean([r[1] for r in rows]):>7.3f} '
                      f'{np.mean([r[2] for r in rows]):>6.2f} '
                      f'{np.mean([r[3] for r in rows]):>7.3f} {solved:>5}/4 '
                      f'{np.mean([r[4] for r in rows]):>8.4f} '
                      f'{np.mean([r[5] for r in rows]):>10.3f}   '
                      f'{" ".join(f"{a:.3f}" for a in accs)}', flush=True)
            print()
