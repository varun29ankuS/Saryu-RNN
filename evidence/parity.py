"""Does the PARITY OBSTRUCTION force beta off 2, and does even n_h remove it?

The argument
-----------
A product of n_h Householder reflections at beta = 2 has det = (-1)^n_h for EVERY token,
including whichever token carries the identity. A homomorphism needs det(T_e) = +1 and det
multiplicative, so at ODD n_h and exact beta = 2 no faithful homomorphism exists at all.
The model's only escape is to leave the orthogonal group -- and that is exactly what the
benchmark measures: beta drifting from 2.0 down to ~1.46-1.68.

But beta < 2 is a CONTRACTION. It dissipates. And losslessness is the whole reason the
transport extrapolates to 42x length. So odd n_h may be silently trading away the
architecture's central property to escape an obstruction that even n_h does not have.

At EVEN n_h the image lies in SO(d), and the relevant groups sit there exactly:
    S_4 is the rotation group of the CUBE          -> S_4 < SO(3)
    A_4 is the rotation group of the TETRAHEDRON   -> A_4 < SO(3)

Predictions, stated before running
----------------------------------
  1. n_h even  -> beta STAYS near 2 (no obstruction to escape).
  2. n_h odd   -> beta DRIFTS well below 2 (it must, to be a homomorphism at all).
  3. n_h even  -> equal or better solve rate at d=3, with lower group-law error.
  4. FROZEN beta = 2 should work at even n_h and fail at odd n_h. This is the sharpest
     test: it removes the escape route entirely, so the obstruction becomes decisive.
"""
from __future__ import annotations

import itertools
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import os  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transport'))
from diagnose import character_norm, group_law_error  # noqa: E402

torch.set_num_threads(4)

ELS = sorted(itertools.permutations(range(4)))
IX = {p: i for i, p in enumerate(ELS)}
TAB = torch.tensor([[IX[tuple(b[a[i]] for i in range(4))] for b in ELS] for a in ELS])
N = len(ELS)
RUNGS = [1.0, 0.25, 1 / 12, 1 / 24]


def hh_init(n_tokens, n_h, dim, corr=0.5):
    v = F.normalize(torch.randn(n_tokens, n_h, dim), dim=-1)
    if n_h > 1 and corr and dim > 1:
        a, rest = v[:, :1], v[:, 1:]
        perp = F.normalize(rest - (rest * a).sum(-1, keepdim=True) * a, dim=-1)
        v = torch.cat([a, corr * a + math.sqrt(max(0.0, 1 - corr ** 2)) * perp], 1)
    return v


class Saryu(nn.Module):
    def __init__(self, dim, n_h, freeze_beta=False, beta_init=2.0):
        super().__init__()
        self.v = nn.Parameter(hh_init(N, n_h, dim, 0.5))
        b = torch.full((N, n_h), beta_init)
        self.raw = nn.Parameter(b, requires_grad=not freeze_beta)
        self.h0 = nn.Parameter(torch.randn(dim) * 0.3)
        self.readout = nn.Linear(dim, N)
        self.dim, self.n_h = dim, n_h
        self.beta_param = 'free'
        self.transport = self
        self.input_dependent = False

    def step(self, s, tok):
        v, beta = self.v[tok], self.raw[tok]
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


def batch(bs, L, g):
    x = torch.randint(0, N, (bs, L), generator=g)
    acc = x[:, 0].clone()
    ys = [acc.clone()]
    for t in range(1, L):
        acc = TAB[acc, x[:, t]]
        ys.append(acc.clone())
    return x, torch.stack(ys, 1)


def hom_loss(m, n=64):
    h = torch.randn(n, m.dim)
    a = torch.randint(0, N, (n,)); b = torch.randint(0, N, (n,))
    return (m.step(m.step(h, a), b) - m.step(h, TAB[a, b])).norm(dim=1).mean() / m.dim ** 0.5


def det_of(m):
    """mean |det(T_g)| over tokens -- 1.0 exactly iff the transports are orthogonal."""
    with torch.no_grad():
        eye = torch.eye(m.dim)
        cols = [m.step(eye[j].unsqueeze(0).expand(N, m.dim), torch.arange(N))
                for j in range(m.dim)]
        M = torch.stack(cols, -1)
        return float(torch.det(M).abs().mean()), float(torch.det(M).mean())


def run(dim, n_h, freeze, seed, steps=2000):
    torch.manual_seed(seed)
    m = Saryu(dim, n_h, freeze_beta=freeze)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=2e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(500 + seed)
    for _ in range(steps):
        x, y = batch(256, 12, g)
        loss = F.cross_entropy(m(x).reshape(-1, N), y.reshape(-1)) + hom_loss(m)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    ge = torch.Generator().manual_seed(99)
    with torch.no_grad():
        a = 0.0
        for _ in range(4):
            x, y = batch(256, 96, ge)
            a += (m(x)[:, -1].argmax(-1) == y[:, -1]).float().mean().item()
        a /= 4
        adet, sdet = det_of(m)
        return a, float(m.raw.mean()), group_law_error(m, TAB), character_norm(m, N), sdet


if __name__ == '__main__':
    print('PARITY OBSTRUCTION: does odd n_h force beta off 2?')
    print(f'  S_4, d=3, train L=12 -> eval L=96.  S_4 IS the rotation group of the cube,')
    print('  so at EVEN n_h (det=+1, image in SO(3)) it should fit with beta at 2.')
    print('  At ODD n_h every T_g has det=-1 at beta=2, which no homomorphism allows.')
    print()
    print(f'{"n_h":>4} {"beta":>7} {"acc":>7} {"rung":>7} {"det":>8} {"grp-law":>8} '
          f'{"<chi,chi>":>10}   seeds')
    print('-' * 76)
    for freeze in (False, True):
        print(f'--- beta {"FROZEN at 2.0" if freeze else "learnable (init 2.0)"} ---')
        for n_h in (2, 3, 4):
            accs, betas, dets, gls, cns = [], [], [], [], []
            for seed in range(4):
                a, b, gl, cn, sdet = run(3, n_h, freeze, seed)
                accs.append(a); betas.append(b); dets.append(sdet)
                gls.append(gl); cns.append(cn)
            import numpy as np
            best = max(accs)
            near = min(RUNGS, key=lambda v: abs(v - best))
            print(f'{n_h:>4} {np.mean(betas):>7.3f} {best:>7.3f} {near:>7.3f} '
                  f'{np.mean(dets):>8.3f} {np.mean(gls):>8.4f} {np.mean(cns):>10.3f}   '
                  f'{" ".join(f"{x:.3f}" for x in accs)}', flush=True)
        print()
