"""Saryu v3: a recurrent language model whose state is carried by Householder reflections.

Each layer keeps one small VECTOR state per head. Every token, per head:

    T_t = H_2 H_1,   H_i = I - beta_i u_i u_i^T      u_i unit, beta_i = 1 - cos(theta_i) in [0, 2]
    h_t = a_t * T_t h_{t-1} + b_t                     a_t = 1 - g_t,  b_t = g_t * c_t,
                                                      g_t = (1 - cos(phi_t)) / 2, per head

beta = 2 is a pure reflection (norm-preserving, eigenvalue -1); reflections do not commute, so the
state depends on the ORDER of the tokens. The gate g decides how much of the state is replaced by
new content c. The recurrence is trained with an exact chunk-parallel kernel (`chunkwise`), which
matches the step-by-step recurrence (`sequential`) to float precision.

Block anatomy: LayerNorm -> depthwise causal conv(4) -> transport -> per-head RMSNorm ->
SiLU output gate, interleaved with a SwiGLU feed-forward.

This module is the model code of the trained checkpoints, extracted verbatim from the training
scripts (5M v4b and 25M runs, enwik8):

    checkpoints/saryu_v4b_last.pt   5,016,097 params  d=320  use_vemb=True
    checkpoints/saryu_25m.pt       25,183,861 params  d=744  use_vemb=False

Loading either into this module reproduces the original code's logits exactly.

REFERENCES
  [Grazzi et al. 2024] Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues, arXiv 2411.12537
  [Siems et al. 2025] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products, arXiv 2502.10297
  [Yang et al. 2024a] Parallelizing Linear Transformers with the Delta Rule over Sequence Length, arXiv 2406.06484
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NH, NHH = 2, 8      # n_h reflections per head, number of heads
CHUNK = 8           # chunk length of the parallel kernel


# ------------------------------------------------------------ chunkwise kernel (proven)
def chunkwise(h0, u, beta, a, b, C=CHUNK):
    """h_t = a_t T_t h_{t-1} + b_t, chunk-parallel. Inverse-free influence form; exact for
    all beta (chunkwise.py: max err ~4e-7 over 15 configs)."""
    B, L, n_h, d = u.shape
    outs = []
    h = h0
    eye = torch.eye(C * n_h, device=u.device)
    rr = torch.arange(C * n_h, device=u.device)[:, None]
    jpos = torch.cat([torch.tensor([-1], device=u.device),
                      (torch.arange(C, device=u.device) + 1) * n_h - 1])
    amask = (jpos[None, :] < rr).to(u.dtype)
    rmask = (rr < (torch.arange(C, device=u.device)[None, :] + 1) * n_h).to(u.dtype)
    for c0 in range(0, L, C):
        uc = u[:, c0:c0 + C].reshape(B, C * n_h, d)
        bc_ = beta[:, c0:c0 + C].reshape(B, C * n_h)
        ac = a[:, c0:c0 + C]
        bb = b[:, c0:c0 + C]
        R = C * n_h
        # v3.1 NUMERICAL FIX: the naive alpha = cumprod(a) then b/alpha overflows when a
        # chunk stacks strong writes (a small -> alpha ~ 1e-50). All PHYSICAL quantities are
        # ratios alpha_t/alpha_j <= 1, so work in log space centered on the chunk mean: every
        # exponential is then bounded by exp(range/2), which C=8 and a >= 0.1 keep < 1e5.
        la = torch.cumsum(torch.log(ac.clamp_min(1e-6)), dim=1)          # [B,C]
        ref = la.mean(dim=1, keepdim=True)
        sc_in = torch.exp(-(la - ref))                                   # 1/alpha_j, centered
        sc_out = torch.exp(la - ref)                                     # alpha_t, centered
        sc0 = torch.exp(ref)                                             # e_0 carries alpha_0=1
        E = torch.cat([h[:, None, :] * sc0[:, :, None],
                       bb * sc_in[:, :, None]], dim=1)
        Ue = torch.einsum('brd,bjd->brj', uc, E)
        rhs = (Ue * amask).sum(-1)
        G = torch.einsum('brd,bsd->brs', uc, uc)
        Lmat = eye.expand(B, R, R) + torch.tril(G * bc_[:, None, :], diagonal=-1)
        y = torch.linalg.solve_triangular(Lmat, rhs[:, :, None], upper=False)[:, :, 0]
        Ecum = torch.cumsum(E, dim=1)[:, 1:]
        corr = torch.einsum('brd,brj->bjd', uc, (bc_ * y)[:, :, None] * rmask)
        hc = (Ecum - corr) * sc_out[:, :, None]
        outs.append(hc)
        h = hc[:, -1]
    return torch.cat(outs, 1)


def sequential(h0, u, beta, a, b):
    B, L, n_h, d = u.shape
    s = h0
    outs = []
    for t in range(L):
        for i in range(n_h):
            ui = u[:, t, i, :]
            s = s - beta[:, t, i:i + 1] * (s * ui).sum(-1, keepdim=True) * ui
        s = a[:, t:t + 1] * s + b[:, t]
        outs.append(s)
    return torch.stack(outs, 1)


# ------------------------------------------------------------ v3 block
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class SaryuV3Block(nn.Module):
    USE_KERNEL = True    # NaN-guard can flip this to the sequential path

    def __init__(self, d, nh=NH, H=NHH, timescales=False, write_scale=False, gate_w_scale=0.01,
                 freeze_gate_bias=False, gate_ceiling=False, chunk=CHUNK):
        """timescales / write_scale are OFF by default: the defaults are the trained configuration,
        so the released checkpoints load and reproduce exactly. They address the measured limit that
        the trained models stop using context at about 512 characters
        (evidence/results/effective_context.txt):

        timescales   every head starts at the same gate (g ~ 0.04, retention ~25 tokens). With this
                     on, the per-head gate biases are log-spaced so retention 1/g spans ~2 to
                     ~10,000 tokens -- heads with different clocks.
        write_scale  the write is g*c, so a head that wants to remember (g -> 0) also stops writing.
                     A learned per-head scale w makes the write (g*w)*c, so a long-memory head can
                     still write when it does write. The state stays bounded: the update is a convex
                     mix of h and w*c, so ||h_t|| <= max(||h_0||, max_s w||c_s||)."""
        super().__init__()
        assert d % H == 0
        self.d, self.nh, self.H, self.dh = d, nh, H, d // H
        # The kernel is exact for ANY chunk size, so this is purely a performance knob. The default
        # 8 is what the released checkpoints were trained with, which keeps their logits identical.
        # On a T4, fwd+bwd of one 25M layer at context 512 is 131 ms at 8 and 21 ms at 64
        # (evidence/results/kernel_profile.txt), so training should pass 32 or 64.
        self.chunk = chunk
        self.ln = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, 4, padding=3, groups=d)      # depthwise causal
        self.v_proj = nn.Linear(d, nh * d, bias=False)           # reflections, per head
        self.b_proj = nn.Linear(d, nh * H, bias=True)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.zeros_(self.b_proj.weight)
        nn.init.constant_(self.b_proj.bias, math.pi)
        self.g_proj = nn.Linear(d, H)                            # PER-HEAD scalar cos gate
        nn.init.constant_(self.g_proj.bias, 0.4)
        gh = torch.logspace(math.log10(0.5), math.log10(1e-4), H)   # the per-head timescale ladder
        if timescales:                       # g = (1-cos phi)/2, so phi = arccos(1-2g)
            with torch.no_grad():
                self.g_proj.bias.copy_(torch.arccos(1.0 - 2.0 * gh))
                # The ladder lives in the bias, and the default input term erases it: at width 144
                # the rows of g_proj.weight have norm ~0.8, swinging the gate by ~+-0.6 radians
                # against a 0.03-radian bias for the long-memory head, so every head ended up
                # retaining 2-4 tokens (evidence/results/gate_timescales.txt). Shrinking the input
                # term lets each head start at its own timescale and learn to modulate around it.
                self.g_proj.weight.mul_(gate_w_scale)
        # A HARD per-head ceiling on the gate. Initialising the ladder (attempt 2) and freezing it
        # (attempt 3) both failed: training moved the bias, then grew the input weights instead, so
        # every head ended at 2-3 tokens of retention either way. With this on, head h can never
        # exceed g_h, so its retention can only be LONGER than the ladder -- the model cannot route
        # around it. If the effective context still stops at ~256, long memory is useless to the
        # model on this task and the gate is not the thing to fix.
        self.register_buffer('g_max', gh.clone() if gate_ceiling else None)
        if freeze_gate_bias:
            # attempt 2 gave heads retentions of 2 to 11,011 tokens and training collapsed every one
            # of them to 2-3 within 1000 steps (evidence/results/gate_timescales.txt). Freezing the
            # bias leaves only the input term learnable, so the ladder cannot be destroyed: if the
            # effective context STILL stops at ~256, the write rule is the constraint, not the gate.
            self.g_proj.bias.requires_grad_(False)
        # softplus(0.5413) = 1.0, so the write starts exactly as it does without this option
        self.w_raw = nn.Parameter(torch.full((H,), 0.5413)) if write_scale else None
        self.c_proj = nn.Linear(d, d)
        self.hnorm = RMSNorm(self.dh)
        self.og = nn.Linear(d, d)                                # silu output gate
        self.out = nn.Linear(d, d)
        self.h0 = nn.Parameter(torch.randn(H, self.dh) * 0.3)

    def mix(self, z, ve=None):
        """returns u [B,L,H,nh,dh], beta [B,L,H,nh], a [B,L,H], b [B,L,H,dh]"""
        B, L, _ = z.shape
        u = F.normalize(self.v_proj(z).view(B, L, self.H, self.nh, self.dh), dim=-1)
        beta = 1.0 - torch.cos(self.b_proj(z).view(B, L, self.H, self.nh))
        g = ((1.0 - torch.cos(self.g_proj(z))) / 2.0).clamp(max=0.90)   # alpha-underflow guard
        if self.g_max is not None:
            g = torch.minimum(g, self.g_max)          # per-head ceiling: retention can only grow
        cz = self.c_proj(z)
        if ve is not None:                      # value embeds into the write
            cz = cz + ve
        c = cz.view(B, L, self.H, self.dh)
        gw = g if self.w_raw is None else g * F.softplus(self.w_raw).clamp(max=4.0)
        return u, beta, 1.0 - g, gw[..., None] * c

    def forward(self, x, ve=None):
        B, L, _ = x.shape
        z = self.ln(x)
        z = self.conv(z.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u, beta, a, b = self.mix(z, ve)
        # fold heads into batch for the kernel
        uf = u.permute(0, 2, 1, 3, 4).reshape(B * self.H, L, self.nh, self.dh)
        bf_ = beta.permute(0, 2, 1, 3).reshape(B * self.H, L, self.nh)
        af = a.permute(0, 2, 1).reshape(B * self.H, L)
        bbf = b.permute(0, 2, 1, 3).reshape(B * self.H, L, self.dh)
        h0 = self.h0[None].expand(B, self.H, self.dh).reshape(B * self.H, self.dh)
        s = (chunkwise(h0, uf, bf_, af, bbf, self.chunk) if SaryuV3Block.USE_KERNEL
             else sequential(h0, uf, bf_, af, bbf))
        s = s.view(B, self.H, L, self.dh).permute(0, 2, 1, 3)
        s = self.hnorm(s).reshape(B, L, self.d)
        return self.out(s * F.silu(self.og(z)))


class SwiGLU(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__()
        h = mult * d
        self.ln = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        z = self.ln(x)
        return self.w2(F.silu(self.w1(z)) * self.w3(z))


class SaryuV3LM(nn.Module):
    kind = 'saryu-v3'

    def __init__(self, vocab, d, nl, use_vemb=False, timescales=False, write_scale=False,
                 gate_w_scale=0.01, freeze_gate_bias=False, gate_ceiling=False, chunk=CHUNK):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.vemb = nn.Embedding(vocab, d) if use_vemb else None
        if self.vemb is not None:
            nn.init.normal_(self.vemb.weight, std=0.02)
        self.mix = nn.ModuleList([SaryuV3Block(d, timescales=timescales, write_scale=write_scale,
                                               gate_w_scale=gate_w_scale, chunk=chunk,
                                               freeze_gate_bias=freeze_gate_bias,
                                               gate_ceiling=gate_ceiling)
                                  for _ in range(nl)])
        self.ffn = nn.ModuleList([SwiGLU(d) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.d, self.nl = d, nl

    def forward(self, idx):
        x = self.emb(idx)
        ve = self.vemb(idx) if self.vemb is not None else None
        for m, f in zip(self.mix, self.ffn):
            x = x + m(x, ve)
            x = x + f(x)
        return self.head(self.lnf(x))

    def state_floats(self, L=None):
        return self.d * self.nl


def load_checkpoint(path, map_location='cpu'):
    """Build a SaryuV3LM from a training checkpoint ({'state', 'd', 'arm', ...}).
    Vocabulary size and the value embedding are read off the saved weights."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    st = ck['state']
    model = SaryuV3LM(st['emb.weight'].shape[0], ck['d'], len({k.split('.')[1] for k in st
                                                               if k.startswith('mix.')}),
                      use_vemb='vemb.weight' in st,
                      write_scale='mix.0.w_raw' in st,   # timescales only affect init, not the keys
                      gate_ceiling='mix.0.g_max' in st)
    model.load_state_dict(st)
    return model, ck
