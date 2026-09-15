"""The three arms for SPEC_shift_vs_gdn.md, pure PyTorch so they run and are VERIFIED on CPU first.

fla is Triton/GPU-only, so its GatedDeltaNet cannot run locally at all. GDNRef below is a
sequential pure-PyTorch Gated DeltaNet following Yang et al. 2412.06464 / 2406.06484 Sec.3.3.
That serves two purposes: it is the baseline for local verification, and on GPU it becomes the
REFERENCE the fla kernel is checked against before any result is believed.

THE DESIGN, and why the two pieces belong together:

  TIED MAP        k_t = blend of p_{t-j} for j>=1,  v_t = p_t,  ONE shared percept map phi.
                  S then maps p_{t-lag} -> p_t: a TRANSITION OPERATOR, so S^d walks d hops.
                  Untied (every fla layer) makes S a lookup table where S.S is meaningless -
                  measured at 0.120 on two hops, chance (compose_probe.py).
  ITERATED READ   q <- normalize(S q), n_read times. This is ONLY meaningful when the map is
                  tied: chasing a pointer requires the retrieved value to be a legal key. With
                  untied k/v it is nonsense, which is why no DeltaNet-family model does it.

Together they give multi-hop from ONE layer. Separately, neither does: a tied map with one read
gives one hop; an iterated read on an untied map gives noise. The 2x2 is the experiment.

SHARPNESS CONSTRAINT, measured in sharpness_sweep.py: composition survives while the dominant
tap weight is >= ~0.90 and collapses by 0.54. So the tap is a softmax at tau <= 0.3, NOT a free
conv, and the dominant weight is logged every eval as a failure mode to detect.

usage: python experiments/shift_arms.py          # CPU smoke: verifies all arms train

REFERENCES
  [Yang et al. 2024a] Parallelizing Linear Transformers with the Delta Rule over Sequence Length, arXiv 2406.06484
  [Yang et al. 2024b] Gated Delta Networks: Improving Mamba2 with Delta Rule, arXiv 2412.06464
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recall_tasks import make_mqar, make_chain, vocab

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class TiedBlock(nn.Module):
    """OURS. Matrix state, delta rule, TIED percept map with a sharpness-constrained lag tap,
    and an optional ITERATED read. `tied=False` gives the untied control (what fla layers do)."""

    def __init__(self, d, H=4, n_read=1, tau=0.25, ntap=4, tied=True, tap_init='shift'):
        super().__init__()
        assert d % H == 0
        self.d, self.H, self.dh = d, H, d // H
        self.n_read, self.tau, self.ntap, self.tied = n_read, tau, ntap, tied
        self.ln = nn.LayerNorm(d)
        self.phi = nn.Linear(d, d, bias=False)            # THE shared percept map
        if not tied:                                       # control: independent k and v maps
            self.k_alt = nn.Linear(d, d, bias=False)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.beta = nn.Linear(d, H, bias=True)
        self.dec = nn.Linear(d, H, bias=True)
        nn.init.constant_(self.dec.bias, 3.0)              # decay ~0.95 at init, not 0.5
        #   TAP INIT IS A MEASURABLE QUESTION, NOT A GIVEN. Initialising at the hard shift means
        #   the tap never has to LEARN the lag - the smoke showed it sitting at [1,0,0,0] simply
        #   because it started there. tap_init='uniform' tests whether the right lag is
        #   discoverable; 'shift' is the prior that encodes it.
        self.tap = nn.Parameter(torch.zeros(ntap))
        if tap_init == 'shift':
            with torch.no_grad():
                self.tap[0] = 4.0
        self.og = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.hnorm = RMSNorm(self.dh)

    def tap_weights(self):
        return F.softmax(self.tap / self.tau, dim=0)

    def forward(self, x):
        B, L, _ = x.shape
        H, dh = self.H, self.dh
        z = self.ln(x)
        p = F.normalize(self.phi(z).view(B, L, H, dh), dim=-1)       # percepts, one map
        a = self.tap_weights()
        #   key lags the value: tap j selects p_{t-(j+1)}, so v_t = p_t is always AHEAD of k_t.
        kacc = torch.zeros_like(p)
        for j in range(self.ntap):
            lag = j + 1
            kacc = kacc + a[j] * F.pad(p, (0, 0, 0, 0, lag, 0))[:, :L]
        k = F.normalize(kacc, dim=-1)
        if not self.tied:
            k = F.normalize(self.k_alt(z).view(B, L, H, dh), dim=-1)  # independent map: control
        v = p
        q = F.normalize(self.q_proj(z).view(B, L, H, dh), dim=-1)
        beta = 2.0 * torch.sigmoid(self.beta(z))
        dec = torch.sigmoid(self.dec(z))
        S = torch.zeros(B, H, dh, dh, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]
            at = dec[:, t].unsqueeze(-1).unsqueeze(-1)
            bt = beta[:, t].unsqueeze(-1).unsqueeze(-1)
            S = at * S
            Sk = torch.einsum('bhvk,bhk->bhv', S, kt)
            S = S - bt * Sk.unsqueeze(-1) * kt.unsqueeze(-2)
            S = S + bt * vt.unsqueeze(-1) * kt.unsqueeze(-2)
            r = qt
            for _ in range(self.n_read):                   # POINTER CHASE: only sane when tied
                r = torch.einsum('bhvk,bhk->bhv', S, r)
                if self.n_read > 1:
                    r = F.normalize(r, dim=-1)
            outs.append(r)
        o = self.hnorm(torch.stack(outs, 1)).reshape(B, L, self.d)
        return self.out(o * F.silu(self.og(z)))


class GDNRef(nn.Module):
    """Gated DeltaNet, sequential pure-PyTorch reference. Separate k/v projections, SiLU+L2
    feature map, Mamba-style decay - i.e. the incumbent, as implemented in fla."""

    def __init__(self, d, H=4):
        super().__init__()
        assert d % H == 0
        self.d, self.H, self.dh = d, H, d // H
        self.ln = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.beta = nn.Linear(d, H, bias=True)
        self.a_proj = nn.Linear(d, H, bias=True)
        A = torch.empty(H).uniform_(0.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A.clamp_min(1e-4)))
        dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.og = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.hnorm = RMSNorm(self.dh)

    def forward(self, x):
        B, L, _ = x.shape
        H, dh = self.H, self.dh
        z = self.ln(x)
        q = F.normalize(F.silu(self.q_proj(z)).view(B, L, H, dh), dim=-1)
        k = F.normalize(F.silu(self.k_proj(z)).view(B, L, H, dh), dim=-1)
        v = F.silu(self.v_proj(z)).view(B, L, H, dh)
        beta = torch.sigmoid(self.beta(z))
        g = -torch.exp(self.A_log) * F.softplus(self.a_proj(z) + self.dt_bias)
        dec = torch.exp(g)
        S = torch.zeros(B, H, dh, dh, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]
            at = dec[:, t].unsqueeze(-1).unsqueeze(-1)
            bt = beta[:, t].unsqueeze(-1).unsqueeze(-1)
            S = at * S
            Sk = torch.einsum('bhvk,bhk->bhv', S, kt)
            S = S - bt * Sk.unsqueeze(-1) * kt.unsqueeze(-2)
            S = S + bt * vt.unsqueeze(-1) * kt.unsqueeze(-2)
            outs.append(torch.einsum('bhvk,bhk->bhv', S, qt))
        o = self.hnorm(torch.stack(outs, 1)).reshape(B, L, self.d)
        return self.out(o * F.silu(self.og(z)))


class Attn(nn.Module):
    """THE CEILING. Causal attention with RoPE. Without it a low score is unreadable."""

    def __init__(self, d, H=4):
        super().__init__()
        self.H, self.dh = H, d // H
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.o.weight)

    def rope(self, t, pos):
        B, H, L, dh = t.shape
        half = dh // 2
        inv = 1.0 / (10000 ** (torch.arange(half, device=t.device).float() / half))
        ang = pos[:, None].float() * inv[None]
        c, s = ang.cos()[None, None], ang.sin()[None, None]
        a, b = t[..., :half], t[..., half:]
        return torch.cat([a * c - b * s, a * s + b * c], dim=-1)

    def forward(self, x):
        B, L, C = x.shape
        z = self.ln(x)
        q, k, v = (u.view(B, L, self.H, self.dh).transpose(1, 2) for u in self.qkv(z).chunk(3, -1))
        pos = torch.arange(L, device=x.device)
        q, k = self.rope(q, pos), self.rope(k, pos)
        m = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=~m)
        return x + self.o(a.transpose(1, 2).reshape(B, L, C))


class SwiGLU(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__()
        h = mult * d
        self.ln = nn.LayerNorm(d)
        self.w1, self.w3 = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        z = self.ln(x)
        return self.w2(F.silu(self.w1(z)) * self.w3(z))


class Net(nn.Module):
    def __init__(self, arm, V, d, nl, H=4, n_read=1):
        super().__init__()
        self.arm = arm
        self.emb = nn.Embedding(V, d)
        nn.init.normal_(self.emb.weight, std=0.02)

        def mk():
            if arm == 'transformer':
                return Attn(d, H)
            if arm == 'gdn':
                return GDNRef(d, H)
            return TiedBlock(d, H, n_read=n_read, tied=(arm != 'untied'))
        self.mix = nn.ModuleList([mk() for _ in range(nl)])
        self.ffn = nn.ModuleList([SwiGLU(d) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, idx):
        x = self.emb(idx)
        for m, f in zip(self.mix, self.ffn):
            x = x + m(x) if self.arm != 'transformer' else m(x)
            x = x + f(x)
        return self.head(self.lnf(x))[:, -1]            # readout at the QUERY position

    def tap_report(self):
        for m in self.mix:
            #   the untied control overwrites k with k_alt, so its tap is UNUSED. Reporting it
            #   implied it carried meaning; it does not.
            if isinstance(m, TiedBlock) and m.tied:
                w = m.tap_weights().detach()
                return float(w.max()), [round(float(t), 3) for t in w]
        return None, None


def batch(task, bs, rng, nent, **kw):
    xs, ys = [], []
    misses = 0
    while len(xs) < bs:
        r = make_mqar(rng, nent=nent, **kw) if task == 'mqar' else make_chain(rng, nent=nent, **kw)
        if r is None:
            #   the generators return None when a cell CANNOT be built (too few entities, or a
            #   sequence longer than seqlen) - deterministically, every time. Retrying forever
            #   hangs a GPU session silently until the 9-hour cap.
            misses += 1
            if misses > 1000 and not xs:
                raise ValueError('%s cell cannot be built: nent=%d %s' % (task, nent, kw))
            continue
        xs.append(r[0]); ys.append(r[1])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def run(arm, task, steps, d, nl, nent, bs=32, lr=3e-3, n_read=1, seed=0, log=None, **kw):
    torch.manual_seed(seed)
    _, _, _, V = vocab(nent)
    m = Net(arm, V, d, nl, n_read=n_read)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    t0, losses = time.time(), []
    for i in range(steps):
        x, y = batch(task, bs, rng, nent, **kw)
        loss = F.cross_entropy(m(x), y)
        if not torch.isfinite(loss):
            return dict(arm=arm, acc=float('nan'), params=npar, note='NON-FINITE LOSS')
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        losses.append(float(loss))
    m.eval()
    rng2 = np.random.default_rng(9999)
    corr = []
    with torch.no_grad():
        for _ in range(6):
            x, y = batch(task, 64, rng2, nent, **kw)
            corr += (m(x).argmax(-1) == y).tolist()
    mx, w = m.tap_report()
    return dict(arm=arm, acc=float(np.mean(corr)), params=npar, secs=time.time() - t0,
                loss0=losses[0], loss1=losses[-1], tap_max=mx, tap=w)


if __name__ == '__main__':
    STEPS = int(os.environ.get('STEPS', 400))
    D, NL, NENT = int(os.environ.get('D', 64)), 2, 32
    print('CPU SMOKE - does every arm TRAIN, and can the ceiling SOLVE an easy cell?')
    print('  d=%d layers=%d steps=%d  (NOT the real experiment - this only verifies the code)' % (D, NL, STEPS))
    print('  fla is Triton/GPU-only, so `gdn` here is a sequential pure-PyTorch reference.')
    print()
    print('MQAR, 4 pairs - the EASIEST cell. The ceiling must clear 0.90 or nothing below is readable.')
    print('  %-12s %8s %9s %9s %9s %7s  %s' % ('arm', 'acc', 'loss0', 'loss1', 'params', 'secs', 'tap'))
    for arm in ('transformer', 'gdn', 'tied', 'untied'):
        r = run(arm, 'mqar', STEPS, D, NL, NENT, npairs=4, seqlen=32,
                n_read=(2 if arm in ('tied', 'untied') else 1))
        print('  %-12s %8.3f %9.3f %9.3f %9d %7.0f  %s'
              % (r['arm'], r['acc'], r['loss0'], r['loss1'], r['params'], r.get('secs', 0),
                 ('max %.3f %s' % (r['tap_max'], r['tap'])) if r.get('tap_max') else '-'), flush=True)
    print()
    print('CHAIN depth 2, no distractors - needs COMPOSITION, which is the whole claim.')
    print('  %-12s %8s %9s %9s %9s %7s  %s' % ('arm', 'acc', 'loss0', 'loss1', 'params', 'secs', 'tap'))
    for arm in ('transformer', 'gdn', 'tied', 'untied'):
        r = run(arm, 'chain', STEPS, D, NL, NENT, depth=2, ratio=0.0, seqlen=32,
                n_read=(2 if arm in ('tied', 'untied') else 1))
        print('  %-12s %8.3f %9.3f %9.3f %9d %7.0f  %s'
              % (r['arm'], r['acc'], r['loss0'], r['loss1'], r['params'], r.get('secs', 0),
                 ('max %.3f %s' % (r['tap_max'], r['tap'])) if r.get('tap_max') else '-'), flush=True)
    print()
    print('  READ: this is a CODE CHECK at a tiny scale, not a result. What must hold:')
    print('   1. every arm trains - loss1 < loss0, no NaN')
    print('   2. the transformer clears 0.90 on MQAR-4, else the task or harness is broken')
    print('   3. the tap stays sharp (max >= 0.90); if it decays, composition is being lost')
    print('   4. params are comparable across arms before any real comparison is run')
