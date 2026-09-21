"""A FAITHFUL DeltaNet cell, matched component-by-component against the reference implementation.

WHY THIS FILE EXISTS. evidence/cell_shootout.py contains a DeltaCell that scored 0.133-0.164 on
4-pair MQAR across three logged seeds, while published DeltaNet handles MQAR comfortably. The
results file built on it claimed 0.984, which no run log supports and which is retracted. Before
any matrix state is built into Saryu, there has to be a matrix reference that actually reproduces
published behaviour -- otherwise we are again building on a number with nothing behind it.

COMPARED AGAINST fla-org/flash-linear-attention, fla/layers/delta_net.py. What the old cell was
missing, in the order it matters:

  1. OUTPUT NORMALISATION. The reference applies a per-head RMSNorm to the read BEFORE the output
     projection. Ours went straight from S q into a Linear. The contraction read has no natural
     scale -- it grows with how much is stored -- so without a norm the downstream layer sees a
     signal whose magnitude drifts with sequence position.
  2. BETA RANGE. Reference: beta = sigmoid(b_proj(x)), and with allow_neg_eigval it is DOUBLED to
     (0, 2). Ours capped at (0, 1). The transition (I - beta k k^T) has eigenvalue 1 - beta along
     k, so beta > 1 makes it NEGATIVE. That is exactly Grazzi et al. 2411.12537, "Unlocking
     State-Tracking in Linear RNNs Through Negative Eigenvalues" -- a paper already in this
     project's refs.bib and cited in its own README. We had it on the shelf and capped it away.
  3. ACTIVATIONS. Reference applies silu to q, k and v. Ours applied none.
  4. SHORT CONVOLUTION, one per projection (q, k, v separately), kernel 4. Ours had none at all
     until today, and then a single conv on the shared input.
  5. HEAD STRUCTURE. Reference splits key/value dims across num_heads. Ours was one head.

REGISTERED PREDICTION, before running: with all five present this reaches > 0.8 on 4-pair MQAR,
which is what the published results imply and what the old cell never did.

FALSIFIER: if it still lands near 0.15, the gap is NOT these components, the failure is in the
harness or the task construction, and no conclusion about matrix states can be drawn from this
codebase until that is found.

    python evidence/delta_reference.py
Environment: NPAIRS2, STEPS, SEEDS, LR, D, HEADS, HEADDIM, NEGEIG, THREADS.
"""
from __future__ import annotations

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
from saryu.metrics import Run                                        # noqa: E402

STEPS = int(os.environ.get('STEPS', 3000))
SEEDS = int(os.environ.get('SEEDS', 2))
LR = float(os.environ.get('LR', 3e-3))
D = int(os.environ.get('D', 128))
HEADS = int(os.environ.get('HEADS', 4))
HEADDIM = int(os.environ.get('HEADDIM', 16))          # 4 x 16 x 16 = 1024 state floats
NEGEIG = os.environ.get('NEGEIG', '1') == '1'
BS = 32
GAP = 4
NENT = 64
SEP = NENT
PAIRS = [int(p) for p in os.environ.get('NPAIRS2', '4').split(',')]
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


class ShortConvolution(nn.Module):
    """Depthwise causal conv, kernel 4 -- one per projection, as in the reference."""

    def __init__(self, d, k=4):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, groups=d, padding=k - 1)

    def forward(self, x):
        return self.conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)


class DeltaNetRef(nn.Module):
    """S_t = S_{t-1}(I - b k k^T) + b v k^T ;  o_t = S_t q_t, per head.

    beta in (0,2) makes the transition's eigenvalue along k negative, which is what lets a single
    step do more than contract -- the negative-eigenvalue result."""

    def __init__(self, d, heads, head_dim, neg_eig=True):
        super().__init__()
        self.h, self.dk = heads, head_dim
        inner = heads * head_dim
        self.ln = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, inner, bias=False)
        self.k_proj = nn.Linear(d, inner, bias=False)
        self.v_proj = nn.Linear(d, inner, bias=False)
        self.q_conv, self.k_conv, self.v_conv = (ShortConvolution(inner) for _ in range(3))
        self.b_proj = nn.Linear(d, heads)
        self.o_norm = nn.RMSNorm(head_dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(head_dim)
        self.o_proj = nn.Linear(inner, d, bias=False)
        self.neg_eig = neg_eig

    def forward(self, x):
        z = self.ln(x)
        B, L, _ = z.shape
        q = F.silu(self.q_conv(self.q_proj(z)))
        k = F.silu(self.k_conv(self.k_proj(z)))
        v = F.silu(self.v_conv(self.v_proj(z)))
        q, k, v = (t.view(B, L, self.h, self.dk) for t in (q, k, v))
        q = F.normalize(q, dim=-1)                                   # qk_norm = 'l2'
        k = F.normalize(k, dim=-1)
        beta = torch.sigmoid(self.b_proj(z))                         # [B,L,H]
        if self.neg_eig:
            beta = beta * 2.0                                        # allow_neg_eigval
        S = torch.zeros(B, self.h, self.dk, self.dk, device=z.device, dtype=z.dtype)
        outs = []
        for t in range(L):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]                   # [B,H,dk]
            bt = beta[:, t][..., None]                               # [B,H,1]
            Sk = torch.einsum('bhvk,bhk->bhv', S, kt)                # what k currently holds
            S = S + torch.einsum('bhv,bhk->bhvk', bt * (vt - Sk), kt)
            outs.append(torch.einsum('bhvk,bhk->bhv', S, qt))
        o = self.o_norm(torch.stack(outs, 1))                        # per-head RMSNorm, then project
        return self.o_proj(o.reshape(B, L, self.h * self.dk))


class Net(nn.Module):
    def __init__(self, vocab, d, heads, head_dim, nl=2, neg_eig=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.cells = nn.ModuleList([DeltaNetRef(d, heads, head_dim, neg_eig) for _ in range(nl)])
        self.ffns = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                                                 nn.Linear(2 * d, d)) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.emb(x)
        for c, f in zip(self.cells, self.ffns):
            h = h + c(h)
            h = h + f(h)
        return self.head(self.lnf(h))


def make(rng, n):
    ent = rng.choice(NENT, size=2 * n, replace=False)
    ks, vs = ent[:n], ent[n:]
    rest = np.setdiff1d(np.arange(NENT), ent)
    s = []
    for a, b in zip(ks, vs):
        s += [int(a), int(b)]
    s += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, n))
    s += [SEP, int(ks[i])]
    return s, int(vs[i])


def batch(rng, bs, n):
    o = [make(rng, n) for _ in range(bs)]
    return torch.tensor([a for a, _ in o]), torch.tensor([b for _, b in o])


def train(npairs, nl, seed, neg_eig=NEGEIG):
    torch.manual_seed(seed)
    m = Net(NENT + 2, D, HEADS, HEADDIM, nl=nl, neg_eig=neg_eig)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(1000 + seed)
    ev = np.random.default_rng(7)
    # LR IS IN THE TAG. Without it a learning-rate sweep writes every config to the same filename
    # and each one silently overwrites the last, which is how the first parity sweep left only its
    # final (worst) learning rate in runs/ while the winning number survived in stdout alone.
    log = Run(f'dref-p{npairs}-L{nl}-{"neg" if neg_eig else "pos"}-lr{LR:g}-s{seed}',
              config=dict(arm=f'delta_reference/L{nl}-{"neg" if neg_eig else "pos"}', pairs=npairs,
                          params=npar, heads=HEADS, head_dim=HEADDIM,
                          state_floats=HEADS * HEADDIM * HEADDIM, steps=STEPS, seed=seed,
                          lr=LR, chance_top1=round(1 / npairs, 4)))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS, npairs)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 250 == 0:
            m.eval()
            with torch.no_grad():
                ev2 = np.random.default_rng(7)
                a = 0.
                for _ in range(4):
                    x2, y2 = batch(ev2, BS, npairs)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train()
            best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    return best, npar


def main():
    print(f'FAITHFUL DeltaNet  H={HEADS} dk={HEADDIM} -> {HEADS*HEADDIM*HEADDIM} state floats, '
          f'lr={LR}, {STEPS} steps, {SEEDS} seeds')
    print('the old cell_shootout DeltaCell, logged: 0.133, 0.148, 0.164 at 4 pairs')
    print('transformer on the same task, logged: 1.000 x3\n')
    t0 = time.time()
    print(f'{"config":>22} {"params":>9}  best per seed')
    print('-' * 56)
    for npairs in PAIRS:
        for nl in (1, 2):
            for neg in ([True, False] if nl == 2 else [True]):
                r = [train(npairs, nl, s, neg) for s in range(SEEDS)]
                tag = f'p={npairs} L={nl} beta<{2 if neg else 1}'
                print(f'{tag:>22} {r[0][1]:>9,}  '
                      + ', '.join(f'{x:.3f}' for x, _ in r), flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ: > 0.8 means the architecture was never on trial, our implementation was.')
    print('      still ~0.15 means the gap is the harness or the task, not these components.')


if __name__ == '__main__':
    main()
