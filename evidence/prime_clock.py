"""Does a recurrent state need a CLOCK, and what frequencies should it tick at?

THE GAP. A recurrence knows ORDER by construction -- non-commuting transports are exactly order
sensitivity, which is what the group results in this project are about. It does not obviously know
DISTANCE. The state decays, and decay says "recently", not "47 tokens ago". Transformers get
distance from RoPE, which rotates each 2D pair of dimensions by an angle proportional to position.
We have no positional encoding at all, correctly, because we do not need one for order.

This asks whether we need one for distance, and if so at what frequencies.

THE TASK, chosen to isolate distance and nothing else. A sequence of random tokens, then a marker,
then a distance k. Answer: the token that sat k positions before the marker.

    [t0 t1 t2 ... t_{n-1}]  [SEP]  [DIST_k]   ->   t_{n-k}

Order alone cannot solve it -- you must know HOW FAR BACK. Chance is 1/NVOCAB.

FOUR ARMS, differing only in the phase frequencies of a rotation applied to what gets written:
    none        no clock at all; the control
    geometric   ang_h = base^(-h/H), RoPE's choice
    powers2     ang_h = 2*pi / 2^h, the Clockwork RNN choice (Koutnik et al. 2014)
    primes      ang_h = 2*pi / p_h with p_h the h'th prime

WHY PRIMES MIGHT BEAT POWERS OF TWO, stated precisely so it can be wrong. Phases are cyclic, so a
set of periods disambiguates positions up to their LEAST COMMON MULTIPLE. Powers of two share every
factor: periods 2,4,8,16 give lcm 16, not the product 1024. Co-prime periods 2,3,5,7 give
lcm = product = 210. That is the Chinese Remainder Theorem used as a code, and it is what grid
cells appear to do with near-irrational scale ratios. On this reading Clockwork RNN chose the
spacing that MINIMISES combined range.

NOTE THE SCOPE, because I got this wrong once already. The argument applies to ROTATION, which is
genuinely periodic. It does NOT apply to gate retention, which is decay -- a decaying state does
not wrap around, so there is no period for primes to be co-prime with. Testing this on the gate
bound would have been a category error.

ROUND 2 PREDICTIONS, registered before running:
  P1  any clock beats none.  (Round 1: held clearly, 0.53 to 0.18.)
  P2  golden > ratio32. Ratios 1.618 vs 1.500 at comparable span, so the ONLY systematic
      difference is how badly the ratio can be approximated by a rational. If resonance is the
      mechanism, this is where it shows.
  P3  primes ~ evens. Both are integer sets over the same span; they differ in co-primality and
      little else. If co-primality were the mechanism, primes would win here -- and round 1's
      failed control already argues it is not.

FALSIFIER for P2: golden and ratio32 within seed spread. Then rationality of the ratio is not the
operative quantity either, and frequency choice is an empirical hyperparameter with no principle
behind it. That outcome is worth having: it would close a line rather than leave it open.

WHAT ROUND 1 SHOWED, kept because the ranking was right for the wrong reason.
    clock      best        k=2    k=5    k=11   k=23   k=43
    none    0.141,0.219   0.609  0.156  0.089  0.062  0.036
    geometric 0.289,0.266 0.815  0.461  0.154  0.065  0.036
    powers2 0.422,0.500   0.979  0.812  0.258  0.146  0.076
    primes  0.523,0.539   0.977  0.924  0.612  0.188  0.086
P1 held. Primes led. But P3 -- the control -- FAILED: the arms already differed at k=2, where
nothing has wrapped and lcm cannot apply. So the LCM/CRT account was wrong even though its
prediction of the ranking came out right, which is the most dangerous way to be wrong.

    python evidence/prime_clock.py
Environment: STEPS, SEEDS, LR, NVOCAB, SEQ, THREADS.
"""
from __future__ import annotations

import math
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
from saryu.metrics import Run                                       # noqa: E402

STEPS = int(os.environ.get('STEPS', 3000))
SEEDS = int(os.environ.get('SEEDS', 2))
LR = float(os.environ.get('LR', 1e-3))
NVOCAB = int(os.environ.get('NVOCAB', 32))       # content tokens
SEQ = int(os.environ.get('SEQ', 48))             # content length
D, H, BS = 128, 8, 32
DIST = [2, 5, 11, 23, 43]                        # the distances asked about
SEP = NVOCAB
DTOK = {k: NVOCAB + 1 + i for i, k in enumerate(DIST)}
VOCAB = NVOCAB + 1 + len(DIST)
torch.set_num_threads(int(os.environ.get('THREADS', 4)))
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]


PHI = (1 + 5 ** 0.5) / 2


def freqs(kind, H):
    """Angles as 2*pi / period. What is being varied is the RATIONALITY of the period ratios.

    ROUND 1 GOT THE MECHANISM WRONG, and checking Riemann is what showed it. The LCM story said
    co-prime periods extend unambiguous range, so arms should be equal at short distances where
    nothing has wrapped. They were not (0.609 to 0.979 at k=2), so LCM is not what is operating.

    The zeta connection supplies the right criterion. Montgomery-Dyson-Odlyzko: the zeros of zeta
    have GUE pair correlation -- they REPEL, and are spaced more evenly than random. The
    transferable idea is anti-clustering, not primality. And the known optimum for that is PHI,
    the most irrational number (continued fraction all 1s, so the worst rational approximations),
    which is why it is used for low-discrepancy sampling and has already been proposed for RoPE
    via Weyl's equidistribution theorem.

    The mechanism is ARNOLD TONGUES: rational frequency ratios mode-lock and resonate. Powers of
    two have ratio exactly 2 -- maximally rational, maximally resonant. Primes are a discrete
    approximation to irrationality. Phi is the actual optimum. Resonance degrades a set at EVERY
    scale, which is what round 1 measured and what LCM could not explain.

    Round-1 confounds, both fixed here:
      powers2 had a DEAD HEAD -- ang_0 = 2*pi/2^0 = 2*pi is a full turn, i.e. the identity, so it
        carried no positional information and the arm effectively ran on 7 heads.
      geometric was MIS-SCALED -- 10000^(-h/8) gives periods out to 19,870 tokens, so six of eight
        heads barely rotated over a 50-token sequence. That tested RoPE at the wrong scale, not
        RoPE's choice.
    Every arm below now starts at period 2 and spans a comparable range."""
    if kind == 'none':
        return None
    if kind == 'geometric':      # RoPE's ratio, rescaled to this sequence length
        return torch.tensor([2 * math.pi / (2.0 * (SEQ / 2.0) ** (h / (H - 1))) for h in range(H)])
    if kind == 'ratio32':        # ratio 3/2: RATIONAL, and close to phi so the span is comparable.
        # An earlier version of this arm used 2 * 2**(h/2), whose ratio is sqrt(2) -- IRRATIONAL,
        # i.e. the exact opposite of what the arm is for. Caught by printing the ratios before
        # running. A ratio-2 arm would be maximally rational but its periods reach 256 on a
        # 48-token sequence, which confounds rationality with range; 3/2 keeps the span matched.
        return torch.tensor([2 * math.pi / (2.0 * 1.5 ** h) for h in range(H)])
    if kind == 'golden':         # ratio phi = 1.618: the MOST irrational number
        return torch.tensor([2 * math.pi / (2.0 * PHI ** h) for h in range(H)])
    if kind == 'primes':         # co-prime integers
        return torch.tensor([2 * math.pi / PRIMES[h] for h in range(H)])
    if kind == 'evens':          # THE CO-PRIMALITY CONTROL: same range as primes, gcd 2 throughout
        return torch.tensor([2 * math.pi / e for e in [2, 4, 6, 8, 12, 14, 18, 20]][:H])
    raise ValueError(kind)


class ClockCell(nn.Module):
    """h_t = a*h_{t-1} + g*rot(c_t, t).   A decaying store whose writes carry a phase.

    The rotation is applied per head to 2D pairs, which is RoPE's operation: content written at
    different times lands at different phases, so the read can tell WHEN as well as WHAT."""

    def __init__(self, d, H, kind):
        super().__init__()
        self.H, self.dh, self.kind = H, d // H, kind
        self.ln = nn.LayerNorm(d)
        self.c = nn.Linear(d, d)
        self.q = nn.Linear(d, d)
        self.g = nn.Linear(d, H)
        self.a = nn.Linear(d, H)
        self.out = nn.Linear(d, d)
        f = freqs(kind, H)
        self.register_buffer('ang', f if f is not None else torch.zeros(H))

    def rot(self, x, t):
        """x: [B,H,dh]. Rotate 2D pairs by t*ang_h."""
        if self.kind == 'none':
            return x
        ph = t * self.ang.to(x.device)[None, :, None]              # [1,H,1]
        a, b = x[..., ::2], x[..., 1::2]
        c, s = torch.cos(ph), torch.sin(ph)
        return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)

    def forward(self, x):
        z = self.ln(x)
        B, L, _ = z.shape
        c = self.c(z).view(B, L, self.H, self.dh)
        q = self.q(z).view(B, L, self.H, self.dh)
        g = torch.sigmoid(self.g(z))[..., None]
        a = torch.sigmoid(self.a(z))[..., None]
        h = torch.zeros(B, self.H, self.dh, device=z.device, dtype=z.dtype)
        outs = []
        for t in range(L):
            h = a[:, t] * h + g[:, t] * self.rot(c[:, t], float(t))
            # read by UN-rotating with the query's own phase, so a query can ask for a phase
            outs.append(h)
        o = torch.stack(outs, 1)
        return self.out((o * q).reshape(B, L, -1))


class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.cell = ClockCell(D, H, kind)
        self.ffn = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2 * D), nn.GELU(),
                                 nn.Linear(2 * D, D))
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB)

    def forward(self, x):
        h = self.emb(x)
        h = h + self.cell(h)
        h = h + self.ffn(h)
        return self.head(self.lnf(h))


def make(rng, k):
    toks = [int(v) for v in rng.integers(0, NVOCAB, size=SEQ)]
    return toks + [SEP, DTOK[k]], toks[SEQ - k]


def batch(rng, bs, k=None):
    ks = [k if k is not None else int(rng.choice(DIST)) for _ in range(bs)]
    o = [make(rng, kk) for kk in ks]
    return torch.tensor([a for a, _ in o]), torch.tensor([b for _, b in o])


def train(kind, seed):
    torch.manual_seed(seed)
    m = Net(kind)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(1000 + seed)
    log = Run(f'clk-{kind}-s{seed}',
              config=dict(arm=f'prime_clock/{kind}', kind=kind, seq=SEQ, dists=DIST,
                          params=sum(p.numel() for p in m.parameters()), steps=STEPS, seed=seed,
                          chance_top1=round(1 / NVOCAB, 4)))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 250 == 0:
            ev = np.random.default_rng(7); a = 0.
            m.eval()
            with torch.no_grad():
                for _ in range(4):
                    x2, y2 = batch(ev, BS)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train(); best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    # per-distance breakdown: P3 says the arms should agree at short k and separate at long k
    per_k = {}
    m.eval()
    with torch.no_grad():
        for k in DIST:
            ev = np.random.default_rng(11); a = 0.
            for _ in range(6):
                x2, y2 = batch(ev, BS, k=k)
                a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 6
            per_k[k] = a
    log.done()
    return best, per_k


def main():
    print(f'PRIME CLOCK  seq {SEQ}, distances {DIST}, vocab {NVOCAB}, chance {1/NVOCAB:.3f}, '
          f'{STEPS} steps, {SEEDS} seeds')
    print('powers2 periods share every factor (lcm 2,4,8,16 = 16); primes do not (lcm = product)\n')
    t0 = time.time()
    res = {}
    print(f'{"clock":>10}  best per seed      ' + '  '.join(f'k={k:<5}' for k in DIST))
    print('-' * 74)
    want = os.environ.get('ARMS', 'none,geometric,ratio32,golden,primes,evens').split(',')
    for kind in want:
        r = [train(kind, s) for s in range(SEEDS)]
        res[kind] = r
        pk = {k: float(np.mean([x[1][k] for x in r])) for k in DIST}
        print(f'{kind:>10}  ' + ', '.join(f'{b:.3f}' for b, _ in r) + '      '
              + '  '.join(f'{pk[k]:.3f}  ' for k in DIST), flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ')
    print('  P1 any clock > none, else the recurrence already has distance and this is moot.')
    print('  P2 primes >= geometric > powers2 at the LONGEST distances (lcm bounds range).')
    print('  P3 all equal at the SHORTEST distance -- the control saying any effect is about')
    print('     range and not about rotation helping in general.')
    print('  FALSIFIED if primes and powers2 are within seed spread everywhere: lcm is not the')
    print('     operative quantity and frequency choice is a tuned knob, not a number-theory one.')


if __name__ == '__main__':
    main()
