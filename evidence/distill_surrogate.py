"""Can this recurrence stand in for a softmax attention layer? The prerequisite for any distillation.

WHY THIS BEFORE ANY DISTILLATION RUN. Converting a pretrained transformer into a linear-attention
model is cheap and established: RADLADS (arXiv 2505.03005) reports 350-700M tokens, under 0.005%
of the teacher's training budget, and a 72B conversion for under $2,000. MOHAWK, LoLCATs and
Mamba-in-Llama are variants of the same protocol. That is 600x cheaper than pretraining and is the
only route to a Saryu large enough and long-context enough to test what our CPU harness cannot.

Every one of those protocols works the same way: keep the teacher's MLPs and embeddings, replace
ONLY the attention layers with a student that is a DROP-IN -- same q/k/v/o projections, same head
structure -- and train the student to reproduce what the attention layer did. So the question that
decides whether any of it is possible is not about tokens or budget. It is:

    HOW WELL CAN THE DELTA RULE REPRODUCE THE OUTPUT OF A SOFTMAX ATTENTION LAYER?

If the answer is "poorly", no amount of distillation compute helps and the idea dies here for
about an hour of CPU. If the answer is "well", the protocol is worth building.

WHAT IS HONEST TO SAY UP FRONT. The component that receives this distillation is Saryu's MATRIX
MEMORY -- the delta rule with decay -- which is the part of this architecture that is NOT
distinctive. It is our version of Gated DeltaNet. The Householder transport, which is what makes
this model different, is not attention-shaped and has no q/k/v correspondence, so it cannot
receive a distilled attention layer at all. And efficiency_hybrid.txt measured that same matrix
memory as a 2x efficiency PENALTY when trained from scratch at 5M. So a good result here is
evidence that we can be a conversion target, not that the architecture's distinctive part works.

THE SETUP. A teacher attention layer (GQA, RoPE, QK-norm, output gate -- the AttnBlock this repo
already uses, matching what Qwen3-Next and Kimi Linear ship). A student that reuses the SAME
projections and replaces the softmax attention with the chunk-parallel delta rule. Train only the
student's added parameters plus its projections to match the teacher's output on real hidden
states, and report the fraction of the teacher's output variance left unexplained.

ARMS, so the number means something:
    inherit   student initialised FROM the teacher's q/k/v/o weights (the protocol's own move)
    scratch   the same student, randomly initialised -- isolates what inheriting is worth
    identity  a linear map fitted to the teacher's output, the trivial baseline any method must
              beat. Without it "0.3 relative error" is a number with no scale.

REGISTERED PREDICTIONS, before running:
  P1  inherit beats scratch. If it does not, weight inheritance is not transferring anything and
      the protocol's central assumption fails for this architecture.
  P2  both beat the identity baseline by a wide margin. If a plain linear map is competitive, the
      attention layer being distilled is doing almost nothing and this test is uninformative.
  P3  residual relative error under 0.25 for inherit. That is the rough level at which layerwise
      distillation is reported to recover end-to-end quality after a short finetune; well above
      it means the recurrence cannot represent what the attention head is computing.

FALSIFIER: if inherit cannot get below the identity baseline's error, the delta rule is not an
adequate surrogate for this attention layer and distillation into it is not worth GPU quota.

    python evidence/distill_surrogate.py
Environment: STEPS, SEEDS, D, HEADS, CTX, BS, LR, THREADS.
"""
from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from saryu.model import AttnBlock, SaryuV3Block, ALPHA_FLOOR         # noqa: E402
from saryu.metrics import Run                                        # noqa: E402

STEPS = int(os.environ.get('STEPS', 1500))
SEEDS = int(os.environ.get('SEEDS', 2))
D = int(os.environ.get('D', 256))
HEADS = int(os.environ.get('HEADS', 8))
CTX = int(os.environ.get('CTX', 256))
BS = int(os.environ.get('BS', 8))
LR = float(os.environ.get('LR', 1e-3))
CHUNK = int(os.environ.get('CHUNK', 32))
torch.set_num_threads(int(os.environ.get('THREADS', max(1, (os.cpu_count() or 4) - 2))))


class DeltaSurrogate(nn.Module):
    """A drop-in for AttnBlock that computes the delta rule instead of softmax attention.

    Deliberately the SAME projection shapes as the teacher, because that is what makes weight
    inheritance possible and it is the whole mechanism these protocols rely on. The added
    parameters are only what the recurrence needs and attention does not: an erase strength beta
    per head, and a decay alpha."""

    def __init__(self, d, n_heads, n_kv_heads=None, chunk=CHUNK, dk_mult=1, n_delta=1,
                 l2=True, ogate=False, alog=False):
        """dk_mult widens the state, n_delta writes more than once per token, l2 toggles the
        q/k normalisation. Three separable knobs, because the surrogate's 0.38 residual could
        come from state capacity, update rank, or the normalisation, and they need different
        fixes."""
        super().__init__()
        self.nh, self.hd = n_heads, (d // n_heads) * dk_mult
        self.nkv = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 2)
        self.chunk, self.n_delta, self.l2 = chunk, n_delta, l2
        self.ln = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, n_heads * self.hd, bias=False)
        # n_delta sets of k/v per token: the per-token write becomes rank n_delta instead of
        # rank 1, which is DeltaProduct's move and the same primitive the transport already
        # uses (a PRODUCT of Householder reflections rather than one).
        self.kv_proj = nn.Linear(d, 2 * n_delta * self.nkv * self.hd, bias=False)
        self.o = nn.Linear(n_heads * self.hd, d, bias=False)
        self.b_proj = nn.Linear(d, n_heads * n_delta, bias=True)
        self.a_proj = nn.Linear(d, n_heads * n_delta, bias=True)
        self.qs = nn.Parameter(torch.ones(n_heads))      # learnable scale, used when l2 is off
        # OUTPUT GATE. Qwen3-Next projects a gate z alongside q/k/v and applies a GATED RMSNorm
        # to the delta-rule output; Kimi Linear has KimiLinearRMSNormGated. We had a plain norm
        # and no gate -- the same class of omission as the cell_shootout retraction, which lacked
        # the per-head RMSNorm before the output projection.
        self.ogate = ogate
        self.z_proj = nn.Linear(d, n_heads * self.hd, bias=False) if ogate else None
        self.onorm_w = nn.Parameter(torch.ones(self.hd)) if ogate else None
        # A_log DECAY. Ours is alpha = FLOOR + (1-FLOOR)*sigmoid(proj), zero-initialised, so every
        # head starts at the SAME timescale. Qwen and Kimi use -exp(A_log)*softplus(a + dt_bias)
        # with A_log randomised PER HEAD over roughly (1,16) and dt_bias log-uniform, handing the
        # model a spread of timescales at initialisation. This project tried three times to
        # install a timescale ladder by hand and it collapsed every time
        # (results/gate_timescales.txt). They do not install it; they initialise it.
        self.alog = alog
        if alog:
            # Kimi's init: A_log = log(U(1,16)) with log-uniform dt_bias. Qwen's differs --
            # log(U(0.01,16)) with dt_bias = ones -- and an earlier note here described the two
            # as the same scheme, which they are not.
            A = torch.empty(n_heads).uniform_(1.0, 16.0)
            self.A_log = nn.Parameter(torch.log(A))
            dt = torch.empty(n_heads).uniform_(math.log(1e-3), math.log(1e-1)).exp()
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
            # a_proj's bias is +4.0, which under the SIGMOID parameterisation meant alpha ~ 0.988
            # -- almost no forgetting. Under softplus it means the opposite: softplus(4 + dt_bias)
            # is ~0.2-1.7 instead of ~1e-3-1e-1, giving g in [-23, -2.8] and alpha ~ 0.05. The
            # state would lose 95% of itself every token and the memory would be dead at init.
            # The same constant means opposite things in the two parameterisations, so the
            # bias is set conditionally at the init site below rather than here -- setting it
            # here runs BEFORE that line and is silently overwritten by it.
        nn.init.zeros_(self.b_proj.weight); nn.init.constant_(self.b_proj.bias, 0.0)
        nn.init.zeros_(self.a_proj.weight)
        nn.init.constant_(self.a_proj.bias, 0.0 if alog else 4.0)

    @classmethod
    def from_attention(cls, attn: AttnBlock):
        """Inherit the teacher's projections verbatim -- the protocol's central move."""
        m = cls(attn.o.out_features, attn.nh, attn.nkv)
        if m.hd != attn.hd or m.n_delta != 1:
            raise ValueError('inheritance needs matching head dim and a single delta step')
        with torch.no_grad():
            m.ln.weight.copy_(attn.ln.weight); m.ln.bias.copy_(attn.ln.bias)
            m.q_proj.weight.copy_(attn.q_proj.weight)
            m.kv_proj.weight.copy_(attn.kv_proj.weight)
            m.o.weight.copy_(attn.o.weight)
        return m

    def forward(self, x):
        B, T, _ = x.shape
        h = self.ln(x)
        q = h @ self.q_proj.weight.T
        kv = h @ self.kv_proj.weight.T
        nd = self.n_delta
        q = q.view(B, T, self.nh, self.hd)
        k, v = kv.view(B, T, 2, nd, self.nkv, self.hd).unbind(dim=2)
        if self.nkv != self.nh:                                  # GQA -> expand for the recurrence
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=3)
            v = v.repeat_interleave(rep, dim=3)
        beta = torch.sigmoid(self.b_proj(h)).view(B, T, nd, self.nh) * 2.0
        if self.alog:
            av = self.a_proj(h).view(B, T, nd, self.nh)
            g = -self.A_log.exp()[None, None, None] * F.softplus(av + self.dt_bias)
            # clamped only to keep OUR kernel's separated exp() factors in range. The bound is a
            # property of this formulation, not of the parameterisation: a kernel forming
            # exp(g_i - g_j) directly is bounded by 1 and needs no floor at all.
            alpha = torch.exp(g.clamp(min=-3.0))
        else:
            alpha = ALPHA_FLOOR + (1 - ALPHA_FLOOR) * torch.sigmoid(self.a_proj(h))
            alpha = alpha.view(B, T, nd, self.nh)
        alpha = alpha[..., None].expand(B, T, nd, self.nh, self.hd)
        if self.l2:
            k = F.normalize(k, dim=-1)
            q = F.normalize(q, dim=-1)
        else:
            # magnitude preserved, with one learnable scale per head. The L2 form discards the
            # scale a teacher's projections encode, which is the likeliest reason inheritance
            # bought nothing in the first run of this file.
            k = k / (k.norm(dim=-1, keepdim=True) + 1.0)
            q = q * self.qs[None, None, :, None]
        # nd writes per token: flatten (T, nd) into one stream of length T*nd, which is exactly
        # how the level-1 kernel already handles n_h reflections per step. The chunk kernel is
        # unchanged and stays exact; only the sequence it sees is longer.
        kf = k.reshape(B, T * nd, self.nh, self.hd)
        vf = v.reshape(B, T * nd, self.nh, self.hd)
        bf = beta.reshape(B, T * nd, self.nh)
        af = alpha.reshape(B, T * nd, self.nh, self.hd)
        qf = q[:, :, None].expand(B, T, nd, self.nh, self.hd).reshape(B, T * nd, self.nh, self.hd)
        o = SaryuV3Block._recall_chunk(qf, kf, vf, bf, bf, af, self.chunk)
        o = o.view(B, T, nd, self.nh, self.hd)[:, :, -1]         # read AFTER the last write
        if self.ogate:
            # NORM -> WEIGHT -> GATE. Verified against Qwen3NextRMSNormGated, whose source
            # carries the comment "Norm before gate". The first version here did gate-then-norm
            # (the Mamba-2 order) from memory rather than from the source, and was testing a
            # different operation than the one both shipped models use.
            z = self.z_proj(h).view(B, T, self.nh, self.hd)
            o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * self.onorm_w
            o = o * F.silu(z)
        return self.o(o.reshape(B, T, self.nh * self.hd))


def hidden_states(n, d, seed):
    """Stand-in for a teacher's residual stream: correlated across time, not white noise.

    White noise would make the attention layer's job trivial and the surrogate's job easy, and
    the number would mean nothing. An AR(1) sequence with a slowly varying mean is the cheapest
    thing with the temporal structure attention actually exploits."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, CTX, d, generator=g)
    drift = torch.randn(n, 1, d, generator=g) * 0.5
    for t in range(1, CTX):
        x[:, t] = 0.85 * x[:, t - 1] + 0.53 * x[:, t]
    return x + drift


def rel_err(pred, target):
    return float((pred - target).pow(2).sum() / target.pow(2).sum())


def run(arm, teacher, seed):
    torch.manual_seed(1000 + seed)
    if arm == 'identity':
        student = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D, bias=False))
    elif arm == 'inherit':
        student = DeltaSurrogate.from_attention(teacher)
    else:
        student = DeltaSurrogate(D, HEADS, teacher.nkv)
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.0)
    log = Run(f'ds-{arm}-s{seed}',
              config=dict(arm=f'distill_surrogate/{arm}', d=D, heads=HEADS, ctx=CTX, bs=BS,
                          steps=STEPS, seed=seed, lr=LR, chunk=CHUNK,
                          params=sum(p.numel() for p in student.parameters())))
    xv = hidden_states(BS, D, 999)
    with torch.no_grad():
        yv = teacher(xv)
    best = 1e9
    for s in range(STEPS):
        x = hidden_states(BS, D, 10_000 + s)
        with torch.no_grad():
            y = teacher(x)
        loss = F.mse_loss(student(x), y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        if (s + 1) % 100 == 0:
            with torch.no_grad():
                e = rel_err(student(xv), yv)
            best = min(best, e)
            log.log(s + 1, loss=float(loss), **{'eval/rel_err': e, 'eval/best_rel_err': best})
    log.done()
    return best


def main():
    print(f'DISTILL SURROGATE  d={D} heads={HEADS} ctx={CTX} chunk={CHUNK}, {STEPS} steps, '
          f'{SEEDS} seeds')
    print('question: can the delta rule reproduce a softmax attention layer well enough for')
    print('layerwise distillation? Reported as residual relative error -- fraction of the')
    print("teacher's output energy left unexplained. Lower is better.\n")
    t0 = time.time()
    res = {}
    print(f'{"arm":>10}  best residual relative error per seed')
    print('-' * 52)
    for arm in ('identity', 'scratch', 'inherit'):
        vals = []
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            teacher = AttnBlock(D, n_heads=HEADS)
            for p in teacher.parameters():
                p.requires_grad_(False)
            vals.append(run(arm, teacher, sd))
        res[arm] = vals
        print(f'{arm:>10}  ' + ', '.join(f'{v:.4f}' for v in vals), flush=True)
    print(f'\n{time.time()-t0:.0f}s\n')

    print('READ, against the registered predictions')
    # Compare against a NOISE FLOOR, not by minima. The first version of this READ declared
    # P1 held on 0.3785 against 0.3804 -- a 0.5% gap on per-seed ranges that fully overlap. That
    # is the third variant in this project of one mistake: a summary statistic applied without
    # asking whether the difference exceeds the spread. A gap smaller than the within-arm range
    # is reported as NO EFFECT.
    def spread(a):
        return max(res[a]) - min(res[a])
    i, sc, inh = (sum(res[a]) / len(res[a]) for a in ('identity', 'scratch', 'inherit'))
    floor = max(spread('scratch'), spread('inherit'))
    gap = sc - inh
    verdict = ('NO EFFECT -- gap is inside the seed spread' if abs(gap) <= floor
               else 'HOLDS' if gap > 0 else 'FALSIFIED')
    print(f'  P1 inherit ({inh:.4f}) vs scratch ({sc:.4f}), gap {gap:+.4f}, '
          f'seed spread {floor:.4f}: {verdict}')
    print(f'  P2 both beat the identity baseline ({i:.4f}): '
          f'{"HOLDS" if max(sc, inh) < i else "FALSIFIED -- a linear map is competitive, so this"}'
          + ('' if max(sc, inh) < i else ' test is uninformative'))
    print(f'  P3 inherit below 0.25: {"HOLDS" if inh < 0.25 else "FALSIFIED"} ({inh:.4f})')
    if inh >= i:
        print('\n  FALSIFIER FIRED. The delta rule is not an adequate surrogate for this')
        print('  attention layer, and distillation into it is not worth GPU quota.')
    elif inh < 0.25:
        print('\n  The recurrence can carry what this attention layer computes. Distillation is')
        print('  worth building -- but note what it would prove: that our MATRIX MEMORY is a')
        print('  viable conversion target. The Householder transport is not attention-shaped and')
        print('  receives nothing from this, so it is not evidence for the distinctive part.')
    else:
        print('\n  Between the baseline and the threshold: representable but not cleanly, which')
        print('  is the regime where an end-to-end finetune decides it and layerwise numbers do')
        print('  not. That is a bigger commitment than this test was meant to justify.')


if __name__ == '__main__':
    main()
