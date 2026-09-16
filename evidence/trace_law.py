"""Binding capacity is set by the TRACE of the transport, and the number of reflections sets it.

THE IDENTITY. For an orthogonal map g acting on the sphere, the expected overlap of a random unit
vector with its image is the normalised trace:

    E[ v . g v ] = tr(g) / d

so "how much does binding hide the value" IS a character. Writing an element of O(d) by its d/2
rotation angles, tr(g) = sum 2 cos(theta_i):

    one Householder reflection   one angle = pi, the rest 0   ->  tr = d - 2,  overlap = 1 - 2/d
    a phase rotation             every angle nonzero          ->  tr ~ 0,      overlap ~ 0
    a permutation                trace = number of fixed points, typically O(1)

A single reflection is therefore the LEAST hiding non-trivial element of the whole orthogonal group,
which is why reflect-to-bind fails (evidence/results/recall_by_construction.txt) and why it fails
WORSE as the head gets wider: 1 - 2/d -> 1.

THE CONSEQUENCE. A product of k independent random reflections multiplies the expected trace by
(1 - 2/d) each time, so

    overlap(k) ~ exp(-2k / d)

Binding needs overlap near 0, i.e. k ~ d/2. Tracking needs k = 1 or 2, because a near-identity nudge
is exactly what composes across tokens without destroying the state. DeltaProduct (2502.10297)
introduced n_h, reflections per token, as a STATE-TRACKING expressivity knob; it is simultaneously
the BINDING-CAPACITY knob, and the two ends of it are different jobs.

Saryu runs n_h = 2 at head dimension 40 (5M) and 93 (25M), i.e. 2k/d ~ 0.04: as far into the
tracking regime as the parameterisation allows, which is why it composes perfectly and stores
nothing.

WHAT THIS SCRIPT MEASURES. Overlap and recall capacity against k, so the law can be checked rather
than believed, and the crossover located for a given head dimension.

    python evidence/trace_law.py
Environment: D (head dimension), KS, PAIRS, TRIALS, SEED.

REFERENCES
  [Siems et al. 2025] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products,
      arXiv 2502.10297
  [Grazzi et al. 2024] Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues,
      arXiv 2411.12537
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

D = int(os.environ.get('D', 64))
KS = [int(x) for x in os.environ.get('KS', '1,2,4,8,16,32,48').split(',')]
PAIRS = [int(x) for x in os.environ.get('PAIRS', '4,8').split(',')]
TRIALS = int(os.environ.get('TRIALS', 150))


def bind(us, v):
    """Apply a product of reflections. Each is an involution, so unbinding is the same product in
    reverse order -- no inverse needs to be computed or stored."""
    for u in us:
        v = v - 2 * (u * v).sum(-1, keepdim=True) * u
    return v


def main():
    gen = torch.Generator().manual_seed(int(os.environ.get('SEED', 0)))
    print(f'binding by k reflections, head dimension d = {D}')
    print(f'law: overlap ~ exp(-2k/d); binding needs overlap ~ 0, so k ~ d/2 = {D // 2}\n')
    head = f'{"k":>4} {"overlap":>9} {"exp(-2k/d)":>11}'
    print(head + ''.join(f'{p:>8}-pair' for p in PAIRS))
    print('-' * (len(head) + 13 * len(PAIRS)))
    for k in KS:
        ov = []
        for _ in range(TRIALS * 2):
            us = F.normalize(torch.randn(k, D, generator=gen), dim=-1)
            v = F.normalize(torch.randn(D, generator=gen), dim=-1)
            ov.append(float(F.cosine_similarity(bind(us, v), v, dim=0)))
        row = []
        for n in PAIRS:
            hit = 0
            for _ in range(TRIALS):
                keys = F.normalize(torch.randn(n, k, D, generator=gen), dim=-1)
                vals = F.normalize(torch.randn(n, D, generator=gen), dim=-1)
                state = torch.zeros(D)
                for i in range(n):
                    state = state + bind(keys[i], vals[i])
                j = int(torch.randint(0, n, (1,), generator=gen))
                read = bind(list(reversed(list(keys[j]))), state)
                hit += int(int((vals @ read).argmax()) == j)
            row.append(hit / TRIALS)
        print(f'{k:>4} {sum(ov)/len(ov):>9.3f} {math.exp(-2*k/D):>11.3f}'
              + ''.join(f'{v:>13.3f}' for v in row))
    print('\nchance is 1/pairs.  Saryu uses k = 2, i.e. the far tracking end of this curve.')
    print('Cost: k reflections are O(k d) per token, so k ~ d/2 costs O(d^2) -- matrix-memory')
    print('compute with vector-memory STATE, which is the trade this buys.')


if __name__ == '__main__':
    main()
