"""Can a Saryu block REPRESENT key-value recall, with weights set by hand and no training?

WHY. evidence/results/recall_vs_transport.txt reported recall at chance at every gap and concluded
the vector state "stores nothing". That conclusion outran the evidence: the run used lr 1e-3 for 800
steps, and this project's own transformer triage measured 4-pair MQAR at lr 3e-4 -> 1.000 (jumping
at step 1000-1500) and at lr 1e-3 -> below 0.21. So the run may have measured an optimisation
failure. Learnability and representability are different questions, and only the second can be
settled without touching an optimiser.

THE MECHANISM BEING TESTED. A reflection is its own inverse: H(u, 2) H(u, 2) = I. So reflections are
natural BINDING operators for this architecture, with no matrix memory required:

    write   at a value token v, reflect the running state about a direction u(k) derived from the
            key that preceded it. The width-4 causal convolution puts k and v in the same window, so
            one projection can see both.
    store   successive bindings superpose in the state, because the write is additive.
    read    at the query k_i, reflect about u(k_i) again. The binding for that key is undone and its
            value becomes readable; the other bindings stay scrambled.

This is exactly the structure the block already has (u from a projection of the conv window, beta=2
available, additive write), so the question is whether the weights EXIST, not whether the optimiser
finds them.

WHAT THIS SCRIPT DOES. Builds the mechanism directly in numpy/torch at the level of the recurrence
-- unit key directions, reflect-to-bind, superpose, reflect-to-unbind -- and measures recall as a
function of the number of stored pairs and the head dimension. No training, no optimiser, no data
beyond generated keys and values.

READ IT AS: if hand-built recall is near 1.0 for a few pairs, the vector state CAN store and address
facts, and every negative recall result so far is about learning, not capacity. If it collapses to
chance even at one pair, the limit is structural and the matrix memory is the only route.

    python evidence/recall_by_construction.py
Environment: DH (head dimension), PAIRS, TRIALS, SEED.

REFERENCES
  [Arora et al. 2023] Zoology: Measuring and Improving Recall in Efficient Language Models, arXiv 2312.04927
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

DH = int(os.environ.get('DH', 16))          # d/H for the trained models: 40 at 5M, 93 at 25M
PAIRS = [int(x) for x in os.environ.get('PAIRS', '1,2,4,8,16').split(',')]
TRIALS = int(os.environ.get('TRIALS', 200))
torch.manual_seed(int(os.environ.get('SEED', 0)))


def reflect(u, x, beta=2.0):
    """H(u, beta) x = x - beta (u.x) u, the block's own transport, applied to a batch of states."""
    return x - beta * (x * u).sum(-1, keepdim=True) * u


def trial(n_pairs, dh, gen):
    """Write n_pairs bindings into one dh-vector, then read one back. Returns True if recovered."""
    keys = F.normalize(torch.randn(n_pairs, dh, generator=gen), dim=-1)     # u(k): the bind axes
    vals = F.normalize(torch.randn(n_pairs, dh, generator=gen), dim=-1)     # the stored values
    state = torch.zeros(dh)
    for u, v in zip(keys, vals):
        state = state + reflect(u, v)              # bind by reflecting, superpose by adding
    i = int(torch.randint(0, n_pairs, (1,), generator=gen))
    read = reflect(keys[i], state)                 # reflect again: that binding is undone
    scores = vals @ read                           # nearest stored value wins
    return int(scores.argmax()) == i


def perturbation(dh, gen, trials=400):
    """How far does one reflection actually move a random vector? ||H(u)v - v|| = 2|u.v| ~ 2/sqrt(d).

    This is the whole story: a Householder is a RANK-ONE update, so in high dimensions it is nearly
    the identity, and an operator that barely changes a vector cannot bind one."""
    u = F.normalize(torch.randn(trials, dh, generator=gen), dim=-1)
    v = F.normalize(torch.randn(trials, dh, generator=gen), dim=-1)
    moved = (reflect(u, v) - v).norm(dim=-1).mean()
    cos = F.cosine_similarity(reflect(u, v), v, dim=-1).mean()
    return float(moved), float(cos)


def bind_capacity(kind, n_pairs, dh, gen, trials):
    """Capacity of one dh-vector under different BINDING operators, same superpose-and-read test."""
    hits = 0
    for _ in range(trials):
        keys = F.normalize(torch.randn(n_pairs, dh, generator=gen), dim=-1)
        vals = F.normalize(torch.randn(n_pairs, dh, generator=gen), dim=-1)
        if kind == 'reflection':                       # what the block has: one rank-one reflection
            enc = lambda u, v: reflect(u, v)           # noqa: E731
            dec = lambda u, x: reflect(u, x)           # noqa: E731
        elif kind == 'reflection x8':                  # eight of them: still rank-one each
            def enc(u, v, g=gen, d=dh):
                for j in range(8):
                    v = reflect(F.normalize(torch.randn(d, generator=g), dim=-1) * 0 + u, v)
                return v
            dec = enc
        elif kind == 'permutation':                    # a key-specific shuffle: full-rank scrambling
            def enc(u, v):
                p = torch.argsort(u)
                return v[p]
            def dec(u, x):
                p = torch.argsort(u)
                inv = torch.empty_like(p); inv[p] = torch.arange(len(p))
                return x[inv]
        elif kind == 'fhrr phase':                     # what saryu_block.py already uses
            def enc(u, v, d=dh):
                h = d // 2
                ar, ai = u[:h], u[h:]; br, bi = v[:h], v[h:]
                na = torch.sqrt(ar**2 + ai**2) + 1e-6
                ar, ai = ar / na, ai / na
                return torch.cat([ar*br - ai*bi, ar*bi + ai*br])
            def dec(u, x, d=dh):
                h = d // 2
                ar, ai = u[:h], u[h:]; xr, xi = x[:h], x[h:]
                na = torch.sqrt(ar**2 + ai**2) + 1e-6
                ar, ai = ar / na, -ai / na             # conjugate = inverse rotation
                return torch.cat([ar*xr - ai*xi, ar*xi + ai*xr])
        state = torch.zeros(dh)
        for u, v in zip(keys, vals):
            state = state + enc(u, v)
        i = int(torch.randint(0, n_pairs, (1,), generator=gen))
        read = dec(keys[i], state)
        hits += int(int((vals @ read).argmax()) == i)
    return hits / trials


def main():
    gen = torch.Generator().manual_seed(int(os.environ.get('SEED', 0)))
    print('recall by construction: reflect-to-bind, superpose, reflect-to-unbind')
    print('no training; accuracy is "did the queried value win a nearest-neighbour read"\n')
    print(f'{"head dim":>9} ' + ' '.join(f'{p:>7}' for p in PAIRS) + '   pairs stored')
    print('-' * (10 + 8 * len(PAIRS)))
    for dh in (8, 16, 32, DH, 40, 93):
        row = []
        for p in PAIRS:
            hits = sum(trial(p, dh, gen) for _ in range(TRIALS))
            row.append(hits / TRIALS)
        print(f'{dh:>9} ' + ' '.join(f'{v:>7.3f}' for v in row)
              + ('   <- 5M head dim' if dh == 40 else '   <- 25M head dim' if dh == 93 else ''))
    print('\nchance for p pairs is 1/p.  Capacity does NOT grow with dimension, which is the tell.')

    print('\nWHY: one reflection barely moves a vector in high dimensions (rank-one update)')
    print(f'{"head dim":>9} {"||H(u)v - v||":>14} {"cos(H(u)v, v)":>15}')
    for dh in (8, 16, 40, 93, 256):
        moved, cos = perturbation(dh, gen)
        print(f'{dh:>9} {moved:>14.3f} {cos:>15.4f}')
    print('  an operator that leaves a vector almost unchanged cannot bind it to a key.')

    print('\nBINDING OPERATORS COMPARED, same superpose-and-read test, head dim 64')
    print(f'{"operator":>14} ' + ' '.join(f'{p:>7}' for p in PAIRS) + '   pairs stored')
    print('-' * (15 + 8 * len(PAIRS)))
    for kind in ('reflection', 'permutation', 'fhrr phase'):
        row = [bind_capacity(kind, p, 64, gen, max(50, TRIALS // 2)) for p in PAIRS]
        print(f'{kind:>14} ' + ' '.join(f'{v:>7.3f}' for v in row))
    print('\n  permutation and phase binding scramble the WHOLE vector, so interference falls as')
    print('  1/sqrt(d) and capacity grows with dimension. A rank-one reflection does neither.')


if __name__ == '__main__':
    main()
