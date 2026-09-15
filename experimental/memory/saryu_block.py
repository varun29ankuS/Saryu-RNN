"""SaryuBlock - the full stack in one trainable block, with its invariants asserted in __main__.

Everything measured in full_stack / bind_ops lived in numpy with hand-set addresses. This is the
same design as a torch module: matrix store, reflection-tracked state, FHRR phase binding, and a
temporal rotation, each behind a flag so any of them can be ablated.

WHAT THE NUMERICS DECIDED, and which is now encoded here rather than argued:

  store beta = 1          full_stack F5: at beta=2 a superseded fact comes back INVERTED (-0.636)
                          and the surviving one is damaged (0.996 -> 0.739). The erase I - 2kk^T
                          reflects along k. Reflections do not belong in an associative store.
  tracker beta = 2        the SAME operation is exactly right here: one reflection can never be
                          an even permutation, which is the whole non-abelian argument, and the
                          tracker reproduced S_5 at 4.88e-15.
  FHRR, not hadamard      bind_ops: phase binding matches UNBOUND capacity at every load (0.766
                          vs content-only 0.769 at N=96) where hadamard costs -0.155. Hadamard
                          preserves coordinate structure so correlated states stay correlated;
                          phase binding gives a shared slot no coordinate to occupy.
  time for episodes       full_stack under FHRR: time buys episodic separation 0.736 -> 0.940 and
                          nothing on capacity. It is in for recovering WHEN a fact held.

THE ONE DESIGN CHOICE THE NUMERICS DID NOT MAKE, and it is what keeps tying alive:

    THE TAP RUNS OVER THE ADDRESS STREAM, NOT THE PERCEPT STREAM.

    a_t = rot( fhrr(p_t, h_t), t )        the address: content bound to state, then clocked
    v_t = a_t
    k_t = sum_j tap_j * a_(t-j-1)

    With a sharp tap k_t = a_(t-1) = v_(t-1) EXACTLY, so a value is literally the next key and
    S^d still walks d hops - whatever binding is layered inside the address. Had the tap run over
    percepts and the binding been applied afterwards, k_t would be fhrr(p_(t-1), h_t): the LAGGED
    content bound to the CURRENT state, which is not any value the store ever wrote. That is the
    key-side-conv bug in a new costume, and it is the reason check 1 below exists.

RUNTIME NOTE ON THE READ-OUT: the value returned by the chase lives in address space and carries
the clock phase. This block un-rotates it by the current t before the output projection, which
handles time. It does NOT un-bind the state - the output projection is left to learn that, which
is a real approximation and is flagged rather than hidden.

usage: cd experiments && python saryu_block.py     # builds nothing, asserts the invariants
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from parity_block import RMSNorm, ShortConv, robust_push


def as_pairs(x):
    h = x.shape[-1] // 2
    return x[..., :h], x[..., h:]


def fhrr_bind(a, b, eps=1e-4):
    """Bind by PHASE ADDITION. Each (re, im) pair is normalised to unit modulus and multiplied,
    so the result depends on both operands' phases and on neither's magnitude. This is the
    operation bind_ops measured as free at every load."""
    ar, ai = as_pairs(a)
    br, bi = as_pairs(b)
    am = torch.sqrt(ar * ar + ai * ai).clamp_min(eps)
    bm = torch.sqrt(br * br + bi * bi).clamp_min(eps)
    ar, ai, br, bi = ar / am, ai / am, br / bm, bi / bm
    return F.normalize(torch.cat([ar * br - ai * bi, ar * bi + ai * br], -1), dim=-1)


def phase_rot(x, t, ang):
    """The temporal code. A rotation IS phase addition, so this is fhrr_bind with a fixed phase -
    which is why content, state and time are one operation applied three times."""
    a, b = as_pairs(x)
    c, s = torch.cos(t * ang), torch.sin(t * ang)
    return torch.cat([a * c - b * s, a * s + b * c], -1)


class SaryuBlock(nn.Module):
    def __init__(self, d, head_dim=64, conv=True, n_read=2, tau=0.25, ntap=4,
                 tap_init='uniform', bias='l2', Delta=0.3, robust_eps=1e-4,
                 use_state=True, use_time=True, nfreq=8, track_beta=2.0):
        super().__init__()
        assert d % head_dim == 0
        assert head_dim % 2 == 0, 'phase binding needs an even head_dim'
        self.H, self.dh = d // head_dim, head_dim
        self.n_read, self.tau, self.ntap = n_read, tau, ntap
        self.use_state, self.use_time, self.track_beta = use_state, use_time, track_beta
        self.bias, self.Delta, self.robust_eps = bias, Delta, robust_eps
        self.debug, self.trace = False, None

        self.ln = nn.LayerNorm(d)
        self.v_proj = nn.Linear(d, d, bias=False)        # THE percept map; k comes from it
        self.q_proj = nn.Linear(d, d, bias=False)
        self.u_proj = nn.Linear(d, d, bias=False)        # drives the reflection tracker
        self.vc = ShortConv(d) if conv else None
        self.qc = ShortConv(d) if conv else None

        self.tap = nn.Parameter(torch.zeros(ntap))
        if tap_init == 'shift':
            with torch.no_grad():
                self.tap[0] = 4.0

        self.beta = nn.Linear(d, self.H, bias=True)
        self.a_proj = nn.Linear(d, self.H, bias=True)
        A = torch.empty(self.H).uniform_(0.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A.clamp_min(1e-4)))
        dt = torch.exp(torch.rand(self.H) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True

        half = head_dim // 2
        ang = torch.rand(nfreq) * (math.pi - 0.3) + 0.15
        self.register_buffer('ang', ang[torch.arange(half) % nfreq])
        self.h0 = nn.Parameter(torch.randn(self.H, head_dim) * 0.1)

        self.og = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
        self.hnorm = RMSNorm(head_dim)

    def tap_weights(self):
        return F.softmax(self.tap / self.tau, dim=0)

    def forward(self, x):
        B, L, _ = x.shape
        H, dh = self.H, self.dh
        z = self.ln(x)

        vs = self.v_proj(z)
        qs = self.q_proj(z)
        if self.vc is not None:
            vs, qs = self.vc(vs), self.qc(qs)
        p = F.normalize(vs.view(B, L, H, dh), dim=-1)
        q0 = F.normalize(qs.view(B, L, H, dh), dim=-1)

        #   the reflection tracker: h_t = h_(t-1) - beta (u.h) u, beta=2 is a Householder.
        hs = None
        if self.use_state:
            u = F.normalize(self.u_proj(z).view(B, L, H, dh), dim=-1)
            h = self.h0.unsqueeze(0).expand(B, H, dh)
            h = F.normalize(h, dim=-1)
            acc = []
            for t in range(L):
                ut = u[:, t]
                h = h - self.track_beta * (ut * h).sum(-1, keepdim=True) * ut
                acc.append(h)
            hs = torch.stack(acc, 1)

        #   THE ADDRESS, and the tap runs over THIS, so a value is exactly the next key.
        tt = torch.arange(L, device=x.device, dtype=x.dtype).view(1, L, 1, 1)
        a = p
        if self.use_state:
            a = fhrr_bind(a, hs)
        if self.use_time:
            a = phase_rot(a, tt, self.ang)
        a = F.normalize(a, dim=-1)

        #   THE QUERY MUST LIVE IN THE SAME SPACE AS THE KEYS. This was the bug the first glimpse
        #   found: k and v carried fhrr_bind + phase_rot while the query came straight off q_proj
        #   unbound, so S mapped bound -> bound and was then interrogated with an unbound vector.
        #   Retrieval could not match by construction, and the binding arms duly collapsed
        #   (mqar 1.000 unbound vs 0.207 with state; chain 0.277 vs 0.027). That was a wiring
        #   fault of mine, not evidence about binding.
        if self.use_state:
            q0 = fhrr_bind(q0, hs)
        if self.use_time:
            q0 = phase_rot(q0, tt, self.ang)
        q0 = F.normalize(q0, dim=-1)

        w = self.tap_weights()
        kacc = torch.zeros_like(a)
        for j in range(self.ntap):
            kacc = kacc + w[j] * F.pad(a, (0, 0, 0, 0, j + 1, 0))[:, :L]
        k, v = F.normalize(kacc, dim=-1), a

        beta = torch.sigmoid(self.beta(z))                      # store stays in (0,1): F5
        g = -torch.exp(self.A_log) * F.softplus(self.a_proj(z) + self.dt_bias)
        dec = torch.exp(g)

        S = torch.zeros(B, H, dh, dh, device=x.device, dtype=x.dtype)
        tr, outs = ([] if self.debug else None), []
        for t in range(L):
            kt, vt, qt = k[:, t], v[:, t], q0[:, t]
            bt = beta[:, t].unsqueeze(-1)
            if tr is not None:
                tr.append(tuple(y.detach().clone() for y in (kt, vt, bt)))
            S = dec[:, t].unsqueeze(-1).unsqueeze(-1) * S
            u_ = vt * bt - torch.einsum('bhvk,bhk->bhv', S, kt * bt)
            if self.bias == 'robust':
                u_ = robust_push(u_, bt, self.Delta, self.robust_eps)
            S = S + u_.unsqueeze(-1) * kt.unsqueeze(-2)
            r = qt
            for _ in range(self.n_read):
                r = torch.einsum('bhvk,bhk->bhv', S, r)
                if self.n_read > 1:
                    #   eps=1e-4 for the reason given in parity_block's read loop. F.normalize's
                    #   eps is a CLAMP FLOOR - x / max(||x||, eps) - and in fp16 the default
                    #   1e-12 sits below the smallest subnormal (~6e-8), flushing to zero, so the
                    #   clamp does nothing and a collapsed chase is 0/0. Latent rather than active
                    #   (a real chase sits at ||S r|| ~ 0.34-0.45), and inert where it does not
                    #   fire: all 9 invariants return byte-identical numbers with it in place.
                    r = F.normalize(r, dim=-1, eps=1e-4)
            if self.use_time:                                   # un-rotate; state is left to `out`
                r = phase_rot(r, -float(t), self.ang)
            outs.append(r)
        if tr is not None:
            self.trace = dict(steps=tr, S=S.detach().clone(),
                              a=a.detach().clone(), h=None if hs is None else hs.detach().clone())
        o = self.hnorm(torch.stack(outs, 1)).reshape(B, L, H * dh)
        return self.out(o * F.silu(self.og(z)))


# ------------------------------------------------------------------ invariants
if __name__ == '__main__':
    torch.manual_seed(0)
    D, HD, L = 64, 32, 12
    OK, BAD = 'OK', '*** FAIL ***'
    res = []

    def rep(n, what, detail, good):
        res.append(good)
        print('  %-4s %-44s %-24s %s' % (n, what, detail, OK if good else BAD))

    def build(**kw):
        torch.manual_seed(0)
        b = SaryuBlock(D, head_dim=HD, **kw).double()
        with torch.no_grad():
            b.tap.copy_(torch.tensor([20., 0., 0., 0.], dtype=torch.float64))
        b.debug = True
        return b

    x = torch.randn(2, L, D, dtype=torch.float64)
    print('SARYU BLOCK INVARIANTS   d=%d head_dim=%d L=%d float64\n' % (D, HD, L))

    print('TYING SURVIVES EVERY BINDING (the tap runs over the address stream)')
    for us, ut, name in ((False, False, 'content only'), (True, False, 'content*state'),
                         (False, True, 'content*time'), (True, True, 'content*state*time')):
        b = build(use_state=us, use_time=ut)
        with torch.no_grad():
            b(x)
        st = b.trace['steps']
        e = max(float((st[t][0] - st[t - 1][1]).abs().max()) for t in range(1, L))
        rep('1', name, 'max |k_t - v_(t-1)| %.1e' % e, e < 1e-12)

    print('\nTHE TRACKER')
    b = build(use_state=True, use_time=True)
    with torch.no_grad():
        b(x)
    h = b.trace['h']
    nrm = float((h.norm(dim=-1) - 1.0).abs().max())
    rep('2a', 'reflections are orthogonal: ||h|| stays 1', 'max dev %.1e' % nrm, nrm < 1e-10)
    moved = float((h[:, 1:] - h[:, :-1]).abs().max())
    rep('2b', 'the state actually moves', 'max step %.3f' % moved, moved > 1e-3)

    print('\nCAUSALITY - k_t must not see its own value')
    x2 = x.clone()
    torch.manual_seed(7)
    x2[:, 5] = torch.randn(2, D, dtype=torch.float64)
    b2 = build(use_state=True, use_time=True)
    with torch.no_grad():
        b2(x2)
    s1, s2 = b.trace['steps'], b2.trace['steps']
    dv = float((s1[5][1] - s2[5][1]).abs().max())
    dk = float((s1[5][0] - s2[5][0]).abs().max())
    dn = float((s1[6][0] - s2[6][0]).abs().max())
    rep('3a', 'changing token t CHANGES v_t', '%.3e' % dv, dv > 1e-6)
    rep('3b', 'changing token t leaves k_t ALONE', '%.3e' % dk, dk < 1e-12)
    rep('3c', 'changing token t CHANGES k_(t+1)', '%.3e' % dn, dn > 1e-6)

    print('\nCOMPOSITION through the block\'s own state, by binding')
    for us, ut, name in ((False, False, 'content only'), (True, False, 'content*state'),
                         (False, True, 'content*time'), (True, True, 'content*state*time')):
        bb = build(use_state=us, use_time=ut)
        with torch.no_grad():
            bb(x)
        S, st = bb.trace['S'], bb.trace['steps']
        cs = []
        for i in range(2, L - 1):
            y = st[i - 2][1]
            for _ in range(2):
                y = F.normalize(torch.einsum('bhvk,bhk->bhv', S, y), dim=-1)
            cs.append(float(F.cosine_similarity(y, st[i][1], dim=-1).mean()))
        print('    %-22s 2-hop cos %.3f' % (name, sum(cs) / len(cs)), flush=True)

    print('\n' + '=' * 70)
    print('%d of %d invariants hold.  %s' % (sum(res), len(res), 'GO' if all(res) else 'DO NOT TRAIN'))
