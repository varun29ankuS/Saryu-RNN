"""PARITY + DIFFERENTIATOR in one block. Every incumbent feature, then ours on top as flags.

The point is that "ours vs theirs" must differ in ONE thing, not ten. Today's mx1/mx2 runs were
unreadable partly because our block was missing half the family's machinery (head_dim 41 against
their 128-256, no channel gates, no conv, no momentum), so a loss could always be blamed on the
missing parts rather than on the mechanism under test.

INCUMBENT FEATURES, each traced to its source:
  head_dim 128          DeltaNet App. ("head dimension ... set to 128"); GDN-2 uses d_k=d_v=128.
                        We ran 41 in mx1/mx2, below every tested point in the literature.
  expand_v              GDN defaults 2.0 -> a RECTANGULAR d_v x d_k state, 2x the floats per head
                        at equal key dim. We were square.
  short conv per path   DeltaNet Sec.3.4 / GDN: a depthwise causal conv AFTER each of the q/k/v
                        projections, each followed by SiLU. Not one shared conv on the input.
  channel gates         GDN-2 (2605.22791): the scalar beta controls TWO different things - how
                        much to erase on the key side and how much to commit on the value side.
                        Split into channel-wise b_t (d_k) and w_t (d_v).
  Mamba-style decay     A ~ U(0,16), dt log-uniform [1e-3, 0.1], g = -exp(A_log)*softplus(.+dt_bias).
                        A WIDE per-head spread, not one constant. Both tensors are FLAGGED
                        _no_weight_decay; the flag does nothing unless the optimizer reads it
                        (gpu_run.param_groups does - until 2026-09-15 nothing did).
  momentum              MDN (2605.05838, ICML 2026): a momentum state alongside the weight state.
  beta range            (0,1) is the DEFAULT; (0,2) is allow_neg_eigval (2411.12537), opt-in.
  gated output norm     RMSNorm(o) * silu(g), as fla's FusedRMSNormGated.

OURS, both verified absent from all eight fla layers and from FWP / DeltaNet / TTT:
  tied                  k and v from ONE percept map with a lag -> S is a TRANSITION OPERATOR and
                        S^d walks d hops. Untied makes S a lookup table where S.S is meaningless
                        (compose_probe: 0.120 at two hops, chance, vs 1.000 to depth 7 tied).
  n_read                pointer chase q <- normalize(S q). Only coherent when tied.

THEIRS, taken not claimed:
  bias='robust'         MIRAS Eq 17. The attentional bias is ||Sk-v||^2/2 + Delta*||Sk-v||, so the
                        gradient is e + Delta*e/||e||: a constant-magnitude push ON TOP of the
                        least-squares one, which does not vanish as the error shrinks. In bias_fair
                        it stored the most of any objective (||S|| 2.51 vs l2's 1.30) while holding
                        3-hop composition (0.903 vs 0.909). ADDS NO PARAMETERS - Delta is a constant.

BUG FOUND AND FIXED, and it is the reason check 6 now exists. The first version of forward()
built the key by shifting the value stream and then ran the two through SEPARATE short convs
(self.kc and self.vc), and normalised k while leaving v raw. Either alone unties the map: S then
maps kc(shift(p)) -> vc(p), and vc(p_t) is not the key at t+1, which is kc(p_t). Both bugs came
in with the parity features - adding the incumbent's machinery quietly broke OUR mechanism - and
neither changes a shape, raises an error, or stops the loss falling. The tied arm would have gone
to GPU measuring nothing, and "tied doesn't help" would have looked like a result.

THE INVARIANT, now asserted in cpu_precheck check 6: with a pure lag-1 tap, k_t == v_{t-1}
EXACTLY. If a value is not literally the next key, the map is not tied, whatever the flag says.

THE UNIFIED UPDATE, which degenerates correctly to every ancestor:

    S <- D_t * S                                    decay
    u  = (w_t . v_t) - S (b_t . k_t)                gated write minus gated current read
    S <- S + u k_t^T                                rank-1 commit

With b = w = beta (scalar) this IS DeltaNet: S - beta(Sk)k^T + beta v k^T. That equivalence is
asserted numerically in the self-check below rather than assumed.

usage: python experiments/parity_block.py     # runs the degeneration self-checks

REFERENCES
  [Yang et al. 2024a] Parallelizing Linear Transformers with the Delta Rule over Sequence Length, arXiv 2406.06484
  [Yang et al. 2024b] Gated Delta Networks: Improving Mamba2 with Delta Rule, arXiv 2412.06464
  [Hatamizadeh et al. 2026] Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention, arXiv 2605.22791
  [Huang et al. 2026] MDN: Parallelizing Stepwise Momentum for Delta Linear Attention, arXiv 2605.05838
  [Grazzi et al. 2024] Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues, arXiv 2411.12537
  [Behrouz et al. 2025] It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization, arXiv 2504.13173
  [Gu & Dao 2023] Mamba: Linear-Time Sequence Modeling with Selective State Spaces, arXiv 2312.00752
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class ShortConv(nn.Module):
    """Depthwise causal conv + SiLU, applied AFTER a projection. DeltaNet Sec.3.4 / GDN."""

    def __init__(self, d, k=4):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(d, d, k, padding=k - 1, groups=d)

    def forward(self, x):
        L = x.shape[1]
        return F.silu(self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2))


def _try_fla():
    """fla's chunkwise gated delta rule, or None. Verified on a T4 against our own sequential
    recurrence at rel 3.773e-07, with the two-pass chase matching at 9.686e-07, a 76x speedup, and
    gradients reaching the address stream (kaggle/kernel run 2).

    The kernel does not know or care where k and v came from, which is the whole reason this
    works: tying is a lag-weighted sum over the address stream, computed BEFORE the call,
    elementwise and in parallel."""
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        return chunk_gated_delta_rule
    except Exception:       # noqa: BLE001
        return None


_FLA_CHUNK = _try_fla()


def robust_push(u, eta, Delta, eps=1e-4):
    """MIRAS Eq 17's extra term, as a correction to the plain delta-rule update `u`.

    The objective is ||e||^2/2 + Delta*||e||  with  e = Sk - v, so the gradient is
    g = e + Delta*e/||e||. A plain step is u = -eta*e, therefore

        -eta*g = -eta*e - eta*Delta*e/||e|| = u + eta*Delta*u/||u||

    since e and u are antiparallel and the sign cancels in e/||e|| = -u/||u||. The second term
    has CONSTANT magnitude eta*Delta: it does not vanish as the error shrinks, which is the
    whole point (it keeps writing when least-squares has stopped caring) and also the whole risk.

    THE NORM IS PROMOTED, NEVER DEMOTED. fp16 carries no exponent below ~6e-5, so an eps of 1e-12
    flushes to zero there and ||u||=0 makes this 0/0 - measured in cpu_precheck check 3, which is
    the ONLY non-finite cell of eighteen (fp16, ||u||=0, eps=1e-12). That matters because a T4 is
    Turing and has no bf16 tensor cores, so fp16 is exactly what the GPU run would use, and the
    NaN would not appear at step 0 - it appears the first time the memory fits a value exactly.

    This originally read `u.float()`, which promotes fp16/bf16 correctly but silently DEMOTES
    float64 to float32. The precheck's float64 algorithm check then disagreed with numpy at
    2.6e-09 - present at a single step, identical at eps=0 and eps=1e-12, i.e. exactly fp32
    resolution and not the formula at all. promote_types fixes both directions at once.

    eps regularises the UPDATE norm, not the error norm. Since ||u|| = eta*||e||, the exact
    MIRAS denominator ||e||+eps corresponds to ||u||+eps*eta here, so at finite eps this differs
    from Eq 17 by O(eps); the correspondence is exact as eps->0 and is verified there. eps is a
    numerical floor, not part of the objective, and is NOT scaled by eta because eta can be zero
    (and u with it), which would reintroduce the 0/0 this exists to prevent.

    This is module-level so the precheck tests THE SHIPPED PATH, not a retyped copy of it.

    u    (B,H,dv)   eta  (B,H,1)   ->   (B,H,dv)
    """
    acc = torch.promote_types(u.dtype, torch.float32)
    un = u.to(acc).norm(dim=-1, keepdim=True).to(u.dtype)
    return u + eta * Delta * u / (un + eps)


class ParityBlock(nn.Module):
    def __init__(self, d, head_dim=128, expand_v=1.0, conv=True, channel_gates=True,
                 momentum=0.0, neg_eig=False, tied=False, n_read=1, tau=0.25, ntap=4,
                 tap_init='shift', bias='l2', Delta=0.3, robust_eps=1e-4, use_kernel=True):
        super().__init__()
        assert d % head_dim == 0, 'd=%d must be divisible by head_dim=%d' % (d, head_dim)
        assert bias in ('l2', 'robust'), bias
        self.H = d // head_dim
        self.dk = head_dim
        self.dv = int(head_dim * expand_v)
        self.tied, self.n_read, self.tau, self.ntap = tied, n_read, tau, ntap
        self.channel_gates, self.momentum, self.neg_eig = channel_gates, momentum, neg_eig
        self.bias, self.Delta, self.robust_eps = bias, Delta, robust_eps
        self.use_kernel = use_kernel
        #   kept as a GPU tensor and converted only when read: float() here ran a CUDA sync in
        #   every layer of every training step, for a number nothing reads until the end.
        self._S_norm = None
        #   set debug=True to capture the per-step tensors the recurrence ACTUALLY used. Without
        #   this a test can only retype forward()'s own derivation of ke/vw and would share any
        #   mistake with the thing it verifies. cpu_precheck check 2b replays this trace through
        #   an independent numpy implementation of the update.
        self.debug, self.trace = False, None
        H, dk, dv = self.H, self.dk, self.dv

        self.ln = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, H * dk, bias=False)
        self.v_proj = nn.Linear(d, H * dv, bias=False)
        if tied:
            #   ONE map produces both keys and values. v_t = p_t, k_t = lagged blend of p.
            #   Requires dk == dv so a value is a legal key - that IS the mechanism.
            assert dk == dv, 'tied requires expand_v=1.0 so values are legal keys (dk=%d dv=%d)' % (dk, dv)
            self.tap = nn.Parameter(torch.zeros(ntap))
            if tap_init == 'shift':
                with torch.no_grad():
                    self.tap[0] = 4.0
        else:
            self.k_proj = nn.Linear(d, H * dk, bias=False)

        self.qc = ShortConv(H * dk) if conv else None
        #   NO SEPARATE KEY CONV WHEN TIED. A second conv on the key side is a second map, which
        #   is exactly what tying exists to forbid. See the BUG note in the module docstring.
        self.kc = ShortConv(H * dk) if (conv and not tied) else None
        self.vc = ShortConv(H * dv) if conv else None

        self.beta = nn.Linear(d, H, bias=True)
        if channel_gates:
            self.bg = nn.Linear(d, H * dk, bias=True)      # GDN-2 erase gate, key side
            self.wg = nn.Linear(d, H * dv, bias=True)      # GDN-2 write gate, value side

        self.a_proj = nn.Linear(d, H, bias=True)
        A = torch.empty(H).uniform_(0.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A.clamp_min(1e-4)))
        dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True

        self.og = nn.Linear(d, H * dv)
        self.out = nn.Linear(H * dv, d, bias=False)
        self.hnorm = RMSNorm(dv)

    @property
    def last_S_norm(self):
        return float('nan') if self._S_norm is None else float(self._S_norm)

    def kernel_eligible(self, x):
        """Whether the fused path may run. Every clause is a thing the kernel CANNOT express, so
        this is a list of honest restrictions rather than a feature switch:

          debug          the kernel returns no per-step tensors, so cpu_precheck 2b and 6 would
                         silently lose the trace they verify against. Never fuse while tracing.
          channel_gates  the kernel takes a SCALAR beta; GDN-2's channel-wise erase/write gates
                         have no equivalent in it.
          momentum       there is no momentum term in the chunkwise formulation.
          bias=='l2'     THE REAL COST. robust_push modifies u INSIDE the recurrence and the
                         kernel never exposes u, so the MIRAS Eq 17 arm cannot be fused at all and
                         keeps paying the python loop. Worth stating plainly rather than quietly
                         disabling: kernelising is not free across the whole design.
        """
        return (_FLA_CHUNK is not None and x.is_cuda and self.use_kernel
                and not self.debug and not self.channel_gates
                and self.momentum == 0 and self.bias == 'l2')

    def tap_weights(self):
        #   `self.tap` exists ONLY when tied - the untied arm builds k from its own k_proj and has
        #   no lag to weight. Every call site currently guards on `m.tied`, so this was never live,
        #   but a bare AttributeError one unguarded call later would be a puzzle rather than a
        #   message. An audit flagged it; this makes the failure say what it means.
        if not self.tied:
            raise AttributeError(
                'tap_weights() is meaningless on an untied block: there is no tap. The untied arm '
                'derives k from k_proj, so any "tap" reading for it would be an artefact - which '
                'is exactly why shift_arms.tap_report stopped reporting one (see its comment).')
        return F.softmax(self.tap / self.tau, dim=0)

    def forward(self, x):
        B, L, _ = x.shape
        H, dk, dv = self.H, self.dk, self.dv
        z = self.ln(x)

        qs = self.q_proj(z)
        vs = self.v_proj(z)
        if self.qc is not None:
            qs = self.qc(qs)
        if self.vc is not None:
            vs = self.vc(vs)
        q = F.normalize(qs.view(B, L, H, dk), dim=-1)

        if self.tied:
            #   THE percept map, and there is exactly one of it: one projection, one conv, one
            #   normalisation. BOTH k and v are read off THIS tensor, so a value IS a legal key
            #   and S^d walks d hops. Anything that touches one side and not the other - a second
            #   conv, a normalisation applied to k alone - silently unties the map while leaving
            #   every shape and every loss curve looking healthy.
            p = F.normalize(vs.view(B, L, H, dv), dim=-1)
            v = p
            a = self.tap_weights()
            kacc = torch.zeros_like(p)
            for j in range(self.ntap):
                lag = j + 1
                kacc = kacc + a[j] * F.pad(p, (0, 0, 0, 0, lag, 0))[:, :L]
            k = F.normalize(kacc, dim=-1)
        else:
            ks = self.k_proj(z)
            if self.kc is not None:
                ks = self.kc(ks)
            k = F.normalize(ks.view(B, L, H, dk), dim=-1)
            #   the incumbent does NOT normalise v (GDN/DeltaNet normalise q and k only), so the
            #   untied arm keeps it raw. The asymmetry is deliberate: normalising v is REQUIRED
            #   for tying and WRONG for parity, and the two arms must each be faithful.
            v = vs.view(B, L, H, dv)

        bmax = 2.0 if self.neg_eig else 1.0
        beta = bmax * torch.sigmoid(self.beta(z))
        if self.channel_gates:
            bg = torch.sigmoid(self.bg(z).view(B, L, H, dk))
            wg = torch.sigmoid(self.wg(z).view(B, L, H, dv))
        g = -torch.exp(self.A_log) * F.softplus(self.a_proj(z) + self.dt_bias)
        dec = torch.exp(g)

        #   THE FUSED PATH. Measured on a T4 against this very loop: rel 3.773e-07 for one read,
        #   9.686e-07 for the two-pass chase, 76x faster, and gradients reach the address stream
        #   (kaggle/kernel run 2). n_read is repeated kernel CALLS - the kernel returns
        #   o_t = S_t q_t for every t, so feeding o back as the query gives S_t(S_t q_t), which is
        #   exactly what the loop below computes.
        if self.kernel_eligible(x):
            #   AUTOCAST MUST BE OFF HERE AND EVERY INPUT CAST TO ONE DTYPE. Under
            #   autocast(float16) q/k/v come out fp16, but beta and g are built by sigmoid /
            #   softplus / exp, which autocast leaves in fp32. fla forms its A matrix from
            #   beta and g and then does tl.dot(A, v) - fp32 against fp16 - and Triton refuses
            #   to compile: "Both operands must be same dtype. Got fp32 and fp16".
            #
            #   That killed the first P1 run, and NOTHING caught it beforehand because every
            #   kernel check ran OUTSIDE autocast: kaggle/kernel, kaggle/assert and
            #   assert_kernel_matches_loop were all fp32 under no_grad. Component verified,
            #   context missed - the same shape as the untied map and the unbound query.
            #
            #   fp32 is the right target rather than casting down: the bridge measured fp32
            #   chunk_gated_delta_rule matching our own recurrence at rel 3.773e-07, and a T4
            #   is Turing so fp16 buys tensor cores the delta rule's A matrix cannot use safely.
            with torch.amp.autocast('cuda', enabled=False):
                r = q.float()
                kf, vf = k.float(), v.float()
                bf, gf = beta.float(), g.float()
                S_fin = None
                for _ in range(self.n_read):
                    r, S_fin = _FLA_CHUNK(r, kf, vf, g=gf, beta=bf, scale=1.0,
                                          output_final_state=True)
                    if self.n_read > 1:
                        r = F.normalize(r, dim=-1, eps=1e-4)
            if S_fin is not None:
                self._S_norm = S_fin.detach().float().norm(dim=(-2, -1)).mean()
            o = self.hnorm(r).reshape(B, L, H * dv)
            return self.out(o * F.silu(self.og(z)))

        S = torch.zeros(B, H, dv, dk, device=x.device, dtype=x.dtype)
        M = torch.zeros_like(S) if self.momentum > 0 else None
        tr = [] if self.debug else None
        #   the READ side, captured separately so the write-side trace keeps its tuple layout and
        #   cpu_precheck checks 2b and 6 (which index [0] and [1]) are unaffected.
        rd = [] if self.debug else None
        outs = []
        for t in range(L):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]
            bt = beta[:, t].unsqueeze(-1)                                   # (B,H,1)
            if tr is not None:
                tr.append(tuple(u.detach().clone() for u in
                                (kt, vt, bt, dec[:, t])))
            S = dec[:, t].unsqueeze(-1).unsqueeze(-1) * S
            if self.channel_gates:
                ke = bg[:, t] * kt * bt                                     # gated erase key
                vw = wg[:, t] * vt * bt                                     # gated write value
            else:
                ke, vw = kt * bt, vt * bt
            u = vw - torch.einsum('bhvk,bhk->bhv', S, ke)                   # write minus current read
            if self.bias == 'robust':
                u = robust_push(u, bt, self.Delta, self.robust_eps)
            upd = u.unsqueeze(-1) * kt.unsqueeze(-2)
            if M is not None:
                M = self.momentum * M + upd
                S = S + M
            else:
                S = S + upd
            r = qt
            for _ in range(self.n_read):
                r = torch.einsum('bhvk,bhk->bhv', S, r)
                if self.n_read > 1:
                    #   eps=1e-4, NOT F.normalize's default 1e-12. Found by audit; cpu_precheck
                    #   check3 never covered this path, having tested robust_push alone.
                    #
                    #   F.normalize's eps is a CLAMP FLOOR, not an addend: it computes
                    #   x / max(||x||, eps). In fp16, 1e-12 is below the smallest subnormal
                    #   (~6e-8) and flushes to ZERO, so the default clamp is max(||x||, 0) - no
                    #   clamp at all, hence 0/0. 1e-4 is representable in fp16 and actually holds.
                    #
                    #   Measured at fp16: ||S r||=1e-5 gives a non-finite BACKWARD at the default,
                    #   and ||S r||=0 is non-finite both ways; both clean at 1e-4. A realistic
                    #   chase sits at ||S r|| ~ 0.34-0.45, so this is LATENT, not active - it
                    #   fires only when the chase collapses, which is a plausible training state
                    #   and the worst kind to diagnose, since the NaN arrives AFTER the mechanism
                    #   starts working.
                    #
                    #   Because eps clamps rather than adds, the guard is PROVABLY INERT at
                    #   healthy magnitudes: trace_logic 8a compares this against a manual chase at
                    #   the default eps and still reports 0.00e+00, and all 14 links plus
                    #   cpu_precheck 2b/6 return byte-identical numbers to before the change.
                    r = F.normalize(r, dim=-1, eps=1e-4)
            if rd is not None:
                rd.append((S.detach().clone(), qt.detach().clone(), r.detach().clone()))
            outs.append(r)
        #   end-of-sequence state magnitude, logged because the robust bias adds a push that does
        #   NOT vanish as the error shrinks - the failure mode to watch for is unbounded growth.
        self._S_norm = S.detach().float().norm(dim=(-2, -1)).mean()
        if tr is not None:
            self.trace = dict(steps=tr, reads=rd, S=S.detach().clone())
        o = self.hnorm(torch.stack(outs, 1)).reshape(B, L, H * dv)
        return self.out(o * F.silu(self.og(z)))


# ------------------------------------------------------------------ self-checks
def _scalar_delta_reference(S, k, v, beta, dec):
    """Plain DeltaNet/GDN step, written independently: S <- aS - b(Sk)k^T + b v k^T."""
    S = dec * S
    Sk = torch.einsum('bhvk,bhk->bhv', S, k)
    S = S - beta * Sk.unsqueeze(-1) * k.unsqueeze(-2)
    S = S + beta * v.unsqueeze(-1) * k.unsqueeze(-2)
    return S


if __name__ == '__main__':
    torch.manual_seed(0)
    print('PARITY BLOCK - DEGENERATION SELF-CHECKS')
    print('The unified update must reduce EXACTLY to its ancestors, or "parity" is a claim not a fact.')
    print()

    B, H, dk, dv = 2, 2, 8, 8
    S0 = torch.randn(B, H, dv, dk, dtype=torch.float64)
    k = F.normalize(torch.randn(B, H, dk, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, dv, dtype=torch.float64)
    beta = torch.rand(B, H, 1, dtype=torch.float64)
    dec = torch.rand(B, H, 1, 1, dtype=torch.float64)

    ke, vw = k * beta, v * beta
    u = vw - torch.einsum('bhvk,bhk->bhv', dec * S0, ke)
    S_unified = dec * S0 + u.unsqueeze(-1) * k.unsqueeze(-2)
    S_ref = _scalar_delta_reference(S0, k, v, beta.unsqueeze(-1), dec)
    err = float((S_unified - S_ref).abs().max())
    print('  1. channel gates OFF  ->  plain delta rule      max err %.3e   %s'
          % (err, 'EXACT' if err < 1e-12 else '*** DIVERGES ***'))

    bg = torch.ones(B, H, dk, dtype=torch.float64)
    wg = torch.ones(B, H, dv, dtype=torch.float64)
    ke2, vw2 = bg * k * beta, wg * v * beta
    u2 = vw2 - torch.einsum('bhvk,bhk->bhv', dec * S0, ke2)
    S_g = dec * S0 + u2.unsqueeze(-1) * k.unsqueeze(-2)
    err2 = float((S_g - S_ref).abs().max())
    print('  2. channel gates = 1  ->  plain delta rule      max err %.3e   %s'
          % (err2, 'EXACT' if err2 < 1e-12 else '*** DIVERGES ***'))

    print()
    print('  shape / config checks:')
    for cfg in (dict(head_dim=64, expand_v=1.0, tied=False),
                dict(head_dim=64, expand_v=2.0, tied=False),
                dict(head_dim=64, expand_v=1.0, tied=True, n_read=2),
                dict(head_dim=128, expand_v=1.0, tied=True, n_read=3, momentum=0.9)):
        d = 128 if cfg['head_dim'] == 64 else 256
        blk = ParityBlock(d, **cfg)
        x = torch.randn(2, 16, d)
        y = blk(x)
        npar = sum(p.numel() for p in blk.parameters())
        ok = y.shape == x.shape and torch.isfinite(y).all()
        print('    d=%-4d %-56s -> %s  params %7d  %s'
              % (d, str(cfg), tuple(y.shape), npar, 'OK' if ok else '*** BAD ***'))

    print()
    print('  expand_v=2.0 with tied is REFUSED by construction (a value must be a legal key):')
    try:
        ParityBlock(128, head_dim=64, expand_v=2.0, tied=True)
        print('    *** NOT REFUSED - the assert is missing ***')
    except AssertionError as e:
        print('    correctly refused: %s' % e)
