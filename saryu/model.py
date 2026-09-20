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

    def _masks(Lc):
        """Index masks for a chunk of Lc timesteps; Lc < C for a partial final chunk."""
        R = Lc * n_h
        eye = torch.eye(R, device=u.device)
        rr = torch.arange(R, device=u.device)[:, None]
        jpos = torch.cat([torch.tensor([-1], device=u.device),
                          (torch.arange(Lc, device=u.device) + 1) * n_h - 1])
        amask = (jpos[None, :] < rr).to(u.dtype)
        rmask = (rr < (torch.arange(Lc, device=u.device)[None, :] + 1) * n_h).to(u.dtype)
        return R, eye, amask, rmask

    # A length that is not a multiple of C ends in a PARTIAL chunk, which needs its own masks.
    # Until 2026-09-16 this reshaped the short slice as if it were full and raised; every length
    # used before then (128, 512, 12, 96, 384) happened to divide by 8, so it never fired.
    full_len = min(C, L)
    full = _masks(full_len)
    for c0 in range(0, L, C):
        Lc = min(C, L - c0)
        R, eye, amask, rmask = full if Lc == full_len else _masks(Lc)
        uc = u[:, c0:c0 + Lc].reshape(B, R, d)
        bc_ = beta[:, c0:c0 + Lc].reshape(B, R)
        ac = a[:, c0:c0 + Lc]
        bb = b[:, c0:c0 + Lc]
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


def sequential_nl(h0, u, beta, a, b, kind='tanh', k=None):
    """h_t = phi(a_t T_t h_{t-1} + b_t) -- NONLINEAR in the state, so no parallel scan exists.

    Why this is the one structural change worth a separate path. Unrolling the linear recurrence
    gives h_T = sum_t (prod_{s>t} A_s) B_t x_t, so the state is always a LINEAR FUNCTIONAL of the
    past: a sketch, not a store. Sketching theory says such an object answers aggregate queries
    cheaply and point queries ('retrieve item i') only under sparsity or with O(n) decoding. Every
    escape this project has found is one of those two, which is not a coincidence.

    Making phi nonlinear breaks the unrolling and the state stops being a sketch. The cost is the
    parallel scan, measured on this code at 3.0x (ctx 128) and 5.4x (ctx 512) of training time --
    and NOTHING at inference, which is already sequential token by token.

    kind='tanh'  generic squashing.
    kind='topk'  keeps the k largest coordinates of the state, which is the sparsity that sketching
                 theory actually asks for rather than a generic nonlinearity.
    """
    B, L, n_h, d = u.shape
    s = h0
    outs = []
    for t in range(L):
        for i in range(n_h):
            ui = u[:, t, i, :]
            s = s - beta[:, t, i:i + 1] * (s * ui).sum(-1, keepdim=True) * ui
        s = a[:, t:t + 1] * s + b[:, t]
        if kind == 'tanh':
            s = torch.tanh(s)
        elif kind == 'topk':
            kk = min(k or d, d)
            thr = s.abs().topk(kk, dim=-1).values[..., -1:]
            s = s * (s.abs() >= thr).to(s.dtype)
        outs.append(s)
    return torch.stack(outs, 1)


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
                 freeze_gate_bias=False, gate_ceiling=False, chunk=CHUNK,
                 carve=None, bind_write=False, sig2=False, sig2_anti=False,
                 gate_cap=0.90, write_topk=None, state_nl=None, state_topk=None,
                 orth_axes=False, beta_init=math.pi, bind_n=1, bind_sign=False):
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
        self.carve, self.bind_write = carve, bind_write
        self.gate_cap = gate_cap
        self.write_topk = write_topk
        self.state_nl, self.state_topk = state_nl, state_topk
        self.orth_axes = orth_axes
        self.beta_init = beta_init
        self.bind_n = bind_n
        if carve is not None:
            assert nh >= 1 + bind_n and 0 < carve < self.dh, (
                'carve needs nh >= 1 + bind_n and 0 < carve < d_h')
            t = torch.zeros(self.dh); t[:carve] = 1.0
            self.register_buffer('track_mask', t)
            self.register_buffer('mem_mask', 1.0 - t)
        self.ln = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, 4, padding=3, groups=d)      # depthwise causal
        self.v_proj = nn.Linear(d, nh * d, bias=False)           # reflections, per head
        self.b_proj = nn.Linear(d, nh * H, bias=True)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.zeros_(self.b_proj.weight)
        # beta = 1 - cos(bias). The default bias = pi gives beta = 2: a PURE REFLECTION, which is
        # orthogonal and norm-preserving, so the transport never removes anything from the state.
        # beta = 1 gives I - u u^T, a PROJECTION that deletes the u component -- and that is exactly
        # the delta rule's erase term S(I - beta k k^T). The delta rule is already inside this
        # operator family; the model simply never goes there. Measured on the 25M checkpoint
        # (trained_geometry.txt): beta median 2.000, min 1.947, and the fraction within 0.2 of the
        # singular point beta = 1 is 0.000 in every layer. Training starts at 2 and stays, plausibly
        # because beta = 1 is where the map is rank-deficient and destroys information, so the
        # gradient treats it as a repeller. beta_init exposes the other regime.
        nn.init.constant_(self.b_proj.bias, beta_init)
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
        # LEVEL-2 TERM. The state above is level 1 of a path signature -- a gated, transported sum
        # of writes -- and the tensor order of a state bounds the order of interaction it can hold,
        # so a level-1 state cannot represent an ordered PAIR (docs/memory_geometry.md, addendum).
        # This adds the second iterated integral per head, A_t = alpha_t A_{t-1} + p_t k_t^T, read
        # by contraction r_t = A_t q_t. Chen's identity makes signature accumulation associative,
        # so this keeps an exact chunk-parallel form; the quadratic form used in forward() is
        # mathematically identical and is simply faster at the lengths we test.
        self.sig2, self.sig2_anti = sig2, sig2_anti
        if sig2:
            self.k_proj = nn.Linear(d, d, bias=False)
            self.q_proj = nn.Linear(d, d, bias=False)
            self.p_proj = nn.Linear(d, d, bias=False)
            self.r_out = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.r_out.weight)      # starts as a no-op, so the level-1 path is intact
            # Its OWN decay, per head, input-independent. The level-1 gate must move fast to track;
            # memory must not. Sharing one gate is what capped the carve experiment at gap 64
            # (evidence/results/carve_test.txt). sigmoid(5) = 0.993, about 150 tokens of retention.
            self.sig_decay = nn.Parameter(torch.full((H,), 5.0))
        self.hnorm = RMSNorm(self.dh)
        self.og = nn.Linear(d, d)                                # silu output gate
        self.out = nn.Linear(d, d)
        self.h0 = nn.Parameter(torch.randn(H, self.dh) * 0.3)
        # SIGN BINDING. One shared projection produces a +-1 vector used to bind the write and to
        # unbind at the read. The point of +-1 over the existing silu output gate is that it is an
        # INVOLUTION: s * (s * c) = c exactly, so a query that reproduces the key's sign vector
        # inverts the binding with no residue. A continuous gate cannot do that at any setting.
        #
        # Cost is O(d) -- signed diagonal matrices, together with permutations, form the
        # hyperoctahedral group B_d, which is the subgroup of O(d) whose elements apply in O(d)
        # rather than O(d^2). Measured against circular convolution at matched width it has the
        # same capacity (d=512: 1.000/1.000/0.958/0.655 at n=8/16/32/64) for a tenth of the time
        # (10.2 vs 122.6 us per bind).
        self.bind_sign = bind_sign
        self.s_proj = nn.Linear(d, d, bias=False) if bind_sign else None
        if bind_sign:
            nn.init.normal_(self.s_proj.weight, std=d ** -0.5)

    def _sign(self, z):
        """Straight-through sign: +-1 forward, identity gradient."""
        x = self.s_proj(z)
        return x + (torch.sign(x) - x).detach()

    def mix(self, z, ve=None):
        """returns u [B,L,H,nh,dh], beta [B,L,H,nh], a [B,L,H], b [B,L,H,dh]"""
        B, L, _ = z.shape
        u = F.normalize(self.v_proj(z).view(B, L, self.H, self.nh, self.dh), dim=-1)
        if self.orth_axes and self.nh > 1:
            # A product of nh Householders is its own inverse ONLY if the axes are mutually
            # orthogonal. v_proj normalises each axis and never orthogonalises them against each
            # other, so T is not an involution and no self-inverse exists to apply at read time.
            # Measured ||(prod H)^2 - I||: 0.259 at nh=2, 0.655 at nh=4, 1.065 at nh=8 for random
            # axes, and exactly 0 for orthogonal ones at every nh.
            #
            # This predicts the flat nh sweep (nh_ratio.txt). Raising nh does two opposing things:
            # it pushes T away from the identity, which is the decorrelation the trace law is about,
            # and simultaneously away from being its own inverse. The two cancel, which is why the
            # constructed ceiling moves 0.347 -> 0.973 while trained accuracy sits at 0.32. Note the
            # hand-built solutions in expressivity.txt used QR axes, so they were exact involutions
            # while every trained model's axes were not.
            #
            # Modified Gram-Schmidt over the nh axes within each head; differentiable.
            cols = []
            for i in range(self.nh):
                v = u[..., i, :]
                for w in cols:
                    v = v - (v * w).sum(-1, keepdim=True) * w
                cols.append(F.normalize(v, dim=-1, eps=1e-6))
            u = torch.stack(cols, dim=-2)
        beta = 1.0 - torch.cos(self.b_proj(z).view(B, L, self.H, self.nh))
        bind = None
        if self.carve is not None:
            # Split each head: the first `carve` dimensions TRACK, the rest REMEMBER.
            # Reflection 0 is the transport, confined to the tracking dimensions, so it acts as
            # the identity on the memory dimensions -- the holonomy reduces and the memory
            # subspace is never scrambled (docs/memory_geometry.md 2.1,
            # evidence/results/memory_mechanisms.txt).
            # Reflection 1 is confined to the memory dimensions and is made INERT as a transport
            # (beta = 0), because its job is to bind the write instead. Parameter count is
            # unchanged and the kernel never sees the difference.
            # bind_n BINDING AXES, not one. A single reflection is worth nothing: measured by
            # construction at dh=128, a key-bound write scores 0.328 at four pairs with one axis
            # and 0.980 with dh/4 of them, against 0.250 chance (evidence/results/write_operator.txt).
            # bind_write shipped with u[..., 1, :] hardcoded, so it bound with exactly one axis and
            # could not have worked at any learning rate -- which is what the three carve+bind runs
            # measured. They are not evidence against key-bound writes.
            # The binding axes are masked into the memory subspace and then orthogonalised AGAINST
            # EACH OTHER there, because masking destroys any orthogonality established earlier and a
            # product of Householders inverts itself only if its axes are mutually orthogonal.
            nb = min(self.bind_n, self.nh - 1)
            ut = F.normalize(u[..., 0, :] * self.track_mask + 1e-8, dim=-1)
            cols = []
            for i in range(1, 1 + nb):
                v = u[..., i, :] * self.mem_mask
                for w in cols:
                    v = v - (v * w).sum(-1, keepdim=True) * w
                cols.append(F.normalize(v + 1e-8, dim=-1))
            bind = (torch.stack(cols, dim=-2), beta[..., 1:1 + nb])
            u = torch.stack([ut] + cols + [u[..., i, :] for i in range(1 + nb, self.nh)], dim=-2)
            beta = torch.stack([beta[..., 0]]
                               + [torch.zeros_like(beta[..., i]) for i in range(1, 1 + nb)]
                               + [beta[..., i] for i in range(1 + nb, self.nh)], dim=-1)
        # The cap is an alpha-underflow guard for the kernel's log-space rescaling. It is also,
        # by Krohn-Rhodes (krohn1965), the thing that removes the architecture's only FLIP-FLOP:
        # g = 1 is the constant map h_t = c_t, and every finite transformation monoid divides an
        # iterated wreath product of flip-flops and prime GROUPS. Reflections supply the groups;
        # a gate that can never reach 1 supplies no flip-flop. Exposed so that can be tested.
        g = ((1.0 - torch.cos(self.g_proj(z))) / 2.0).clamp(max=self.gate_cap)
        if self.g_max is not None:
            g = torch.minimum(g, self.g_max)          # per-head ceiling: retention can only grow
        cz = self.c_proj(z)
        if ve is not None:                      # value embeds into the write
            cz = cz + ve
        c = cz.view(B, L, self.H, self.dh)
        if bind is not None and self.bind_write:
            # Bind the incoming content inside the memory subspace: the write becomes a reflection
            # of c about the token's memory direction, so the pair is stored bound rather than
            # superposed raw. This is the only multiplicative (key x value) interaction in the
            # layer; without it the write is linear in z and cannot associate anything.
            # Apply ALL bind_n reflections, so the binding operator is a product far from the
            # identity rather than a rank-one nudge.
            ums, bb = bind
            for i in range(ums.shape[-2]):
                ui = ums[..., i, :]
                c = c - bb[..., i:i + 1] * (c * ui).sum(-1, keepdim=True) * ui
        if self.write_topk is not None:
            # The read is DIAGONAL: forward() computes out(s * silu(og(z))), an elementwise
            # product of the state with a query-dependent vector. A per-coordinate reweighting can
            # only separate items whose writes occupy DIFFERENT coordinates. Measured on the 25M
            # checkpoint (evidence/results/trained_geometry.txt): the write c has effective support
            # ~30 of dh=93 while the output gate selects ~11, so every item contributes to every
            # coordinate the gate looks at and capacity is about dh/support ~ 3 -- which is where
            # the binding cliff sits. Masking the write to its top-k coordinates makes the codes
            # closer to disjoint. Straight-through so the gradient still reaches the masked units.
            k = min(self.write_topk, self.dh)
            thr = c.abs().topk(k, dim=-1).values[..., -1:]
            c = c * (c.abs() >= thr).to(c.dtype)
        if self.bind_sign:
            c = self._sign(z).view(B, L, self.H, self.dh) * c      # bind by the token's sign vector
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
        if self.state_nl is not None:
            # A nonlinear state has no parallel scan -- see sequential_nl's docstring.
            s = sequential_nl(h0, uf, bf_, af, bbf, self.state_nl, self.state_topk)
        else:
            s = (chunkwise(h0, uf, bf_, af, bbf, self.chunk) if SaryuV3Block.USE_KERNEL
                 else sequential(h0, uf, bf_, af, bbf))
        s = s.view(B, self.H, L, self.dh).permute(0, 2, 1, 3)
        s = self.hnorm(s).reshape(B, L, self.d)
        if self.bind_sign:
            s = s * self._sign(z)          # same map, so the binding inverts exactly
        y = self.out(s * F.silu(self.og(z)))
        return y + self.r_out(self.level2(z)) if self.sig2 else y

    def level2(self, z):
        """The second iterated integral, per head: A_t = alpha A_{t-1} + p_t k_t^T, read A_t q_t.

        Written here in the equivalent quadratic form, which is exact and faster at the lengths
        tested; the point of Chen's identity is that the recurrent and chunk-parallel forms exist
        and agree, not that this particular expansion is the only way to evaluate it."""
        B, L, _ = z.shape
        H, dh = self.H, self.dh
        k = F.normalize(self.k_proj(z).view(B, L, H, dh), dim=-1)
        q = F.normalize(self.q_proj(z).view(B, L, H, dh), dim=-1)
        p = self.p_proj(z).view(B, L, H, dh)
        alpha = torch.sigmoid(self.sig_decay)                       # [H]
        t = torch.arange(L, device=z.device)
        dt = (t[:, None] - t[None, :]).clamp(min=0).to(z.dtype)     # t - s
        w = torch.exp(torch.log(alpha + 1e-8)[:, None, None] * dt[None])
        w = w * (t[None, :] <= t[:, None]).to(z.dtype)[None]        # causal
        r = torch.einsum('bhls,bshd->blhd', torch.einsum('blhd,bshd->bhls', q, k) * w[None], p)
        if self.sig2_anti:
            # Bivector store: A accumulates p k^T - k p^T, so the readout keeps only the skew part
            # -- the Levy area, the antisymmetric half of the level-2 term.
            r = r - torch.einsum('bhls,bshd->blhd',
                                 torch.einsum('blhd,bshd->bhls', q, p) * w[None], k)
        # Normalise by the decay mass actually in the window, NOT by (1 - alpha). With alpha =
        # 0.993 and L = 74 the sum holds ~74 terms, not 1/(1-alpha) = 150, so the old scaling made
        # the readout 0.0058x the scale of z. Against a zero-initialised r_out that starved the
        # level-2 path of gradient and it never learned (evidence/results/sig2_test.txt, first run).
        return (r / w.sum(-1).clamp_min(1e-6).permute(1, 0)[None, :, :, None]).reshape(B, L, self.d)


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
                 gate_w_scale=0.01, freeze_gate_bias=False, gate_ceiling=False, chunk=CHUNK,
                 nh=NH, H=NHH, carve=None, bind_write=False, sig2=False, sig2_anti=False,
                 gate_cap=0.90, write_topk=None, state_nl=None, state_topk=None,
                 orth_axes=False, beta_init=math.pi, bind_n=1, bind_sign=False):
        """nh (reflections per head) and H default to the trained configuration, so the released
        checkpoints load unchanged. They are exposed because nh is the knob the trace law is about
        (evidence/results/trace_law.txt): overlap ~ exp(-2 nh / dh), so nh trades state tracking
        against binding capacity. Nothing could vary it here before, which meant the law had never
        been tested in a trained language model -- see evidence/nh_sweep.py."""
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.vemb = nn.Embedding(vocab, d) if use_vemb else None
        if self.vemb is not None:
            nn.init.normal_(self.vemb.weight, std=0.02)
        self.mix = nn.ModuleList([SaryuV3Block(d, nh=nh, H=H, carve=carve, bind_write=bind_write,
                                               gate_cap=gate_cap, write_topk=write_topk,
                                               state_nl=state_nl, state_topk=state_topk,
                                               orth_axes=orth_axes,
                                               beta_init=beta_init,
                                               bind_n=bind_n,
                                               bind_sign=bind_sign,
                                               sig2=sig2, sig2_anti=sig2_anti,
                                               timescales=timescales, write_scale=write_scale,
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
                      gate_ceiling='mix.0.g_max' in st,
                      # v_proj maps d -> nh*d, so the saved shape names nh; H comes from the
                      # per-head gate. Read off the weights so a non-default sweep reloads.
                      nh=st['mix.0.v_proj.weight'].shape[0] // ck['d'],
                      H=st['mix.0.g_proj.weight'].shape[0])
    model.load_state_dict(st)
    return model, ck
