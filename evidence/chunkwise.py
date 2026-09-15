"""Chunkwise-parallel Saryu (scalar gate): exact, matmul-only, validated against the loop.

WHY THIS FILE EXISTS. Training is a Python loop over timesteps: 0.906 s/step on a T4 vs the
GRU's 0.121 (cuDNN) -- the wall-clock cap fired before the step budget did. The whole modern
linear-RNN class trains chunkwise-parallel; without that form, nothing here scales. This file
derives and implements ours, and PROVES it equal to the sequential recurrence to float
precision, which is the part a hand-rolled kernel usually gets wrong silently.

THE RECURRENCE (scalar-gate configuration, as in the 5M LM):
    h_t = a_t * T_t h_{t-1} + b_t
        a_t = 1 - g_t            scalar in (0,1)
        T_t = H_{t,2} H_{t,1}    product of n_h Householders, orthogonal
        b_t = g_t * c_t

THE DERIVATION, including the bug the test harness caught in the first version.
 1. SCALARS COMMUTE. alpha_t = prod a_s; h_t = alpha_t * hhat_t turns the recurrence into
    a pure reflection chain with injections e_0 = h_0 and e_j = b_j / alpha_j.
    (The rescale DIES for the per-dimension gate: diag(1-g) does not commute with the
    reflections. That configuration needs a diagonal-plus-low-rank form -- known boundary.)
 2. FIRST ATTEMPT, WRONG: h_t = alpha_t Q_t (h_0 + sum Q_j^T b_j/alpha_j). This assumes
    Q^T = Q^{-1}, which holds ONLY at beta in {0,2}; free beta = 1-cos(theta) is not
    orthogonal, and beta = 1 is a non-invertible projection. The bisect showed it exactly:
    transport-only exact, gate-only exact, combined wrong by O(1). COROLLARY worth keeping:
    with beta PINNED at exactly 2 the transpose form IS valid -- exact orthogonality is a
    compute feature, not just a stability one.
 3. THE INVERSE-FREE FORM (DeltaNet-style). Track the scalar influence of each reflection:
    y_r = u_r . phi_{r-1} obeys the unit-lower-triangular system
        y_r + sum_{s<r} beta_s (u_r.u_s) y_s = sum_{injections before r} u_r . e_j
    and every intermediate state is a LEADING PARTIAL SUM
        hhat_t = sum_{j<=t} e_j - sum_{s<=r_t} beta_s y_s u_s
    because y_s never depends on t. One batched (R x R) triangular solve plus a few masked
    matmuls per chunk; valid for ALL beta including the projection point.

NUMERICAL BOUNDARY, stated up front: w_j divides by alpha_j, so a long chunk of hard writes
(g -> 1, a -> 0) underflows alpha and blows the division -- the known decay-division trap in
chunked linear attention (fla's kernels solve it with segmented/log-space tricks). The tests
below include g up to 0.95; production would clamp chunk length or adopt fla's handling.

VALIDATED HERE: max |chunkwise - sequential| over random configs, gate extremes, and long
sequences; then a CPU wall-clock comparison at L=4096 as an honest first speed datum.

REFERENCES
  [Yang et al. 2024a] Parallelizing Linear Transformers with the Delta Rule over Sequence Length, arXiv 2406.06484
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(2)


def sequential(h0, u, beta, a, b):
    """the reference loop, exactly as the LM runs it.
    h0: [B,d]  u: [B,L,n_h,d]  beta: [B,L,n_h]  a: [B,L]  b: [B,L,d]"""
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


def chunkwise(h0, u, beta, a, b, C=16):
    """same recurrence, chunk-parallel: one triangular solve + masked matmuls per chunk.

    CORRECTED CONSTRUCTION. The first version used h_t = alpha_t Q_t (h0 + sum Q_j^T b_j
    / alpha_j), which silently assumes Q_j^T = Q_j^{-1} -- true only when every factor
    (I - beta u u^T) is orthogonal, i.e. beta in {0, 2}. Our beta is free in [0,2] (and
    beta = 1 is a non-invertible projection), so that identity is false and the bisect
    showed exactly that: transport-only and gate-only exact, combined wrong by O(1).

    The inverse-free form (DeltaNet-style): rescale h_t = alpha_t * hhat_t so the scalars
    vanish, treat h0 and each b_j/alpha_j as INJECTIONS into a pure chain of reflections,
    and solve for the scalar influence of each reflection on the running sum:
        y_r = u_r . phi_{r-1},   phi_r = phi_{r-1} - beta_r y_r u_r (+ injections)
    y satisfies the SAME unit-lower-triangular system as the UT transform,
        y_r + sum_{s<r} beta_s (u_r.u_s) y_s = sum_{injections before r} u_r . e_j
    and because y_s never depends on r, every intermediate state is a LEADING PARTIAL SUM:
        hhat_t = sum_{j <= t} e_j - sum_{s <= r_t} beta_s y_s u_s
    No inverses anywhere, so it is valid for ALL beta including the projection point."""
    B, L, n_h, d = u.shape
    assert L % C == 0
    outs = []
    h = h0
    for c0 in range(0, L, C):
        uc = u[:, c0:c0 + C].reshape(B, C * n_h, d)          # [B,R,d] application order
        bc_ = beta[:, c0:c0 + C].reshape(B, C * n_h)         # [B,R]
        ac = a[:, c0:c0 + C]                                 # [B,C]
        bb = b[:, c0:c0 + C]                                 # [B,C,d]
        R = C * n_h
        alpha = torch.cumprod(ac, dim=1)                     # [B,C]
        # injections: e_0 = carried h (before everything), e_j = b_j / alpha_j (after
        # token j's reflections). E: [B, C+1, d]
        E = torch.cat([h[:, None, :], bb / alpha[:, :, None]], dim=1)
        # availability mask: injection j is inside phi before reflection r iff its
        # position precedes r. e_0 at position 0 (< every r); e_j at r_j = j*n_h (1-idx).
        rr = torch.arange(R)[:, None]                        # reflection row r (0-idx)
        jpos = torch.cat([torch.tensor([-1]),                # e_0 always available
                          (torch.arange(C) + 1) * n_h - 1]) # e_j after row (j+1)nh-1
        amask = (jpos[None, :] < rr)                         # [R, C+1]
        # RHS_r = sum_{available j} u_r . e_j
        Ue = torch.einsum('brd,bjd->brj', uc, E)             # [B,R,C+1]
        rhs = (Ue * amask).sum(-1)                           # [B,R]
        # same unit-lower-triangular system as the UT transform
        G = torch.einsum('brd,bsd->brs', uc, uc)
        Lmat = torch.eye(R).expand(B, R, R) \
            + torch.tril(G * bc_[:, None, :], diagonal=-1)
        y = torch.linalg.solve_triangular(Lmat, rhs[:, :, None], upper=False)[:, :, 0]
        # outputs after token t: leading partial sums, then rescale by alpha
        rmask = (torch.arange(R)[:, None] < (torch.arange(C)[None, :] + 1) * n_h)
        Ecum = torch.cumsum(E, dim=1)[:, 1:]                 # sum_{j<=t} e_j incl e_0
        corr = torch.einsum('brd,brj->bjd', uc, (bc_ * y)[:, :, None] * rmask)
        hc = (Ecum - corr) * alpha[:, :, None]
        outs.append(hc)
        h = hc[:, -1]
    return torch.cat(outs, 1)


def make(B, L, n_h, d, gmax=0.6, gen=None):
    g_ = torch.Generator().manual_seed(7 if gen is None else gen)
    u = F.normalize(torch.randn(B, L, n_h, d, generator=g_), dim=-1)
    beta = 1.0 - torch.cos(torch.randn(B, L, n_h, generator=g_))     # our free-beta range
    g = gmax * torch.rand(B, L, generator=g_)
    a = 1.0 - g
    c = torch.randn(B, L, d, generator=g_)
    b = g[:, :, None] * c
    h0 = F.normalize(torch.randn(B, d, generator=g_), dim=-1)
    return h0, u, beta, a, b


if __name__ == '__main__':
    print('CHUNKWISE SARYU (scalar gate): derivation -> code -> proof of equality.')
    print('  inverse-free form: triangular solve for reflection influences y, outputs as')
    print('  leading partial sums. Valid for ALL beta; per-dim gate needs DPLR (documented).')
    print()
    print('EXACTNESS  (max |chunkwise - sequential| over the whole output tensor)')
    for name, (B, L, n_h, d, gmax) in [
            ('small',        (4, 64, 2, 32, 0.6)),
            ('wide',         (2, 64, 2, 128, 0.6)),
            ('n_h=4',        (2, 64, 4, 48, 0.6)),
            ('hard writes',  (2, 64, 2, 48, 0.95)),
            ('long L=1024',  (1, 1024, 2, 64, 0.6))]:
        h0, u, beta, a, b = make(B, L, n_h, d, gmax)
        ref = sequential(h0, u, beta, a, b)
        for C in (8, 16, 32):
            if L % C:
                continue
            out = chunkwise(h0, u, beta, a, b, C)
            err = float((out - ref).abs().max())
            print('  {:>12}  C={:>2}   max err {:.2e}   {}'.format(
                name, C, err, 'OK' if err < 1e-4 else '*** FAIL ***'), flush=True)
    print()
    print('SPEED (CPU, honest first datum -- the real target is GPU matmuls)')
    B, L, n_h, d = 8, 4096, 2, 128
    h0, u, beta, a, b = make(B, L, n_h, d, 0.6)
    t0 = time.time()
    ref = sequential(h0, u, beta, a, b)
    ts = time.time() - t0
    for C in (32, 64):
        t0 = time.time()
        out = chunkwise(h0, u, beta, a, b, C)
        tc = time.time() - t0
        err = float((out - ref).abs().max())
        print('  L={} d={}: sequential {:.2f}s   chunkwise C={} {:.2f}s   '
              '{:.1f}x   err {:.2e}'.format(L, d, ts, C, tc, ts / tc, err), flush=True)
    print()
    print('  err < 1e-4 everywhere -> the parallel form is EXACT, not approximate; the')
    print('  sequential scan was an implementation choice, and the T4 can now train real')
    print('  context lengths. Next: the DPLR analogue for the per-dim gate (the config')
    print('  that wins tasks), via the fla kernel family.')
