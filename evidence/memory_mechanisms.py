"""Which memory mechanism survives being transported? By construction, no training, CPU seconds.

WHY THIS EXISTS. recall_by_construction.py bound a key to a value, superposed, and unbound
IMMEDIATELY. That is not the situation the recurrence is in: between write and read, every
intervening token applies its own transport to the whole state. Leaving that step out is why the
by-construction numbers looked survivable while the trained model failed (nh_sweep_shortgap.txt).

So this bench puts the scrambling back in, and compares the candidates from docs/memory_geometry.md
on equal terms:

    write   state <- superposition of n bound (key, value) pairs
    scramble  apply `gap` random transports, each a product of 2 reflections, as the model does
    read    recover value j, scored by argmax over the n values

THE UNIFYING QUESTION. Binding survives if and only if the intervening transport COMMUTES with the
unbind operator. Writing A for the accumulated transport and U_j for the unbind of key j, a read
gives U_j A h where correctness needs A U_j h. Every mechanism below is a different way of making
those equal -- or of failing to.

MECHANISMS
  reflect-naive     bind and unbind by reflections about the key. What Saryu has. [A, U] != 0.
  reflect-equiv     unbind about the TRANSPORTED key, A u. Commutes by construction, but needs A,
                    which a vector state does not have. An oracle: the upper bound for this family.
  carved            reflections confined to a tracking subspace W; memory lives in W-perp, which
                    every transport fixes. Commutation by disjoint support (docs 2.1).
  outer-equiv       matrix memory M = sum v k^T, conjugated by the transport, read with A k.
                    Equivariant, so the rotation cancels (docs 2.3).
  outer-naive       the same matrix memory read with the UNtransported key: the control that shows
                    conjugation is not automatically benign.
  bivector          antisymmetric outer product, the Clifford grade-2 store (docs 2.4).

Chance is 1/n. Capacity and persistence are reported separately because the trained model fails on
both axes independently (nh_sweep_shortgap.txt).

    python evidence/memory_mechanisms.py
Environment: D, PAIRS, GAPS, TRIALS, KBIND, SEED.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

D = int(os.environ.get('D', 64))
PAIRS = [int(x) for x in os.environ.get('PAIRS', '1,2,4,8').split(',')]
GAPS = [int(x) for x in os.environ.get('GAPS', '0,4,16,64').split(',')]
TRIALS = int(os.environ.get('TRIALS', 200))
KBIND = int(os.environ.get('KBIND', 2))          # reflections per bind; 2 = the shipped n_h
SEED = int(os.environ.get('SEED', 0))


def rand_unit(gen, *shape):
    return F.normalize(torch.randn(*shape, generator=gen), dim=-1)


def reflect(us, v):
    """Apply a product of Householder reflections (beta = 2). Each is an involution."""
    for u in us:
        v = v - 2 * (u @ v) * u
    return v


def reflect_mat(us, d):
    """The same product as an explicit matrix, so it can be composed and inspected."""
    M = torch.eye(d)
    for u in us:
        M = M - 2 * torch.outer(u, u) @ M
    return M


def transport(gen, d, gap, sub=None):
    """`gap` tokens, each applying 2 random reflections -- the shipped configuration.

    If `sub` is given, the reflection directions are confined to that subspace, so the transport
    acts as the identity on its orthogonal complement."""
    A = torch.eye(d)
    for _ in range(gap):
        us = rand_unit(gen, 2, d)
        if sub is not None:
            us = F.normalize(us @ sub @ sub.T, dim=-1)
        A = reflect_mat(list(us), d) @ A
    return A


# ------------------------------------------------------------------ mechanisms
def run_reflect(gen, d, n, gap, equivariant):
    keys, vals = rand_unit(gen, n, KBIND, d), rand_unit(gen, n, d)
    h = torch.zeros(d)
    for i in range(n):
        h = h + reflect(list(keys[i]), vals[i])
    A = transport(gen, d, gap)
    h = A @ h
    j = int(torch.randint(0, n, (1,), generator=gen))
    us = [A @ u for u in keys[j]] if equivariant else list(keys[j])
    read = reflect(list(reversed(us)), h)
    ref = (A @ vals.T).T if equivariant else vals
    return int((ref @ read).argmax()) == j


def run_carved(gen, d, n, gap):
    """Half the dimensions track, half remember. Transports touch only the tracking half."""
    m = d // 2
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen))
    W, Mem = Q[:, :m], Q[:, m:]                      # tracking and memory subspaces
    # keys and values live in the memory subspace
    keys = F.normalize(torch.randn(n, KBIND, m, generator=gen), dim=-1) @ Mem.T
    vals = F.normalize(torch.randn(n, m, generator=gen), dim=-1) @ Mem.T
    h = torch.zeros(d)
    for i in range(n):
        h = h + reflect(list(keys[i]), vals[i])
    A = transport(gen, d, gap, sub=W)                # confined to the tracking subspace
    h = A @ h
    j = int(torch.randint(0, n, (1,), generator=gen))
    read = reflect(list(reversed(list(keys[j]))), h)
    return int((vals @ read).argmax()) == j


def run_outer(gen, d, n, gap, equivariant):
    keys, vals = rand_unit(gen, n, d), rand_unit(gen, n, d)
    M = sum(torch.outer(vals[i], keys[i]) for i in range(n))
    A = transport(gen, d, gap)
    M = A @ M @ A.T                                  # conjugation: how a matrix state transports
    q = A @ keys[None].squeeze(0).T                  # transported keys
    j = int(torch.randint(0, n, (1,), generator=gen))
    read = M @ (q[:, j] if equivariant else keys[j])
    ref = (A @ vals.T).T if equivariant else vals
    return int((ref @ read).argmax()) == j


def run_bivector(gen, d, n, gap):
    """Clifford grade 2: the antisymmetric part of the outer product, conjugated and read
    equivariantly. Antisymmetry is what makes the store order-sensitive."""
    keys, vals = rand_unit(gen, n, d), rand_unit(gen, n, d)
    B = sum(torch.outer(vals[i], keys[i]) - torch.outer(keys[i], vals[i]) for i in range(n))
    A = transport(gen, d, gap)
    B = A @ B @ A.T
    j = int(torch.randint(0, n, (1,), generator=gen))
    read = B @ (A @ keys[j])
    ref = (A @ vals.T).T
    return int((ref @ read).argmax()) == j


MECHS = {
    'reflect-naive': lambda g, d, n, gap: run_reflect(g, d, n, gap, False),
    'reflect-equiv': lambda g, d, n, gap: run_reflect(g, d, n, gap, True),
    'carved': run_carved,
    'outer-naive': lambda g, d, n, gap: run_outer(g, d, n, gap, False),
    'outer-equiv': lambda g, d, n, gap: run_outer(g, d, n, gap, True),
    'bivector': run_bivector,
}


def main():
    print(f'MEMORY MECHANISMS UNDER TRANSPORT  d={D}, bind uses {KBIND} reflections, '
          f'{TRIALS} trials')
    print('write n pairs -> apply `gap` token transports (2 reflections each) -> read one back\n')
    for n in PAIRS:
        print(f'  {n} pair(s), chance {1/n:.3f}')
        head = f'    {"mechanism":<16}' + ''.join(f'{"gap " + str(g):>10}' for g in GAPS)
        print(head); print('    ' + '-' * (len(head) - 4))
        for name, fn in MECHS.items():
            gen = torch.Generator().manual_seed(SEED)
            row = []
            for gap in GAPS:
                hit = sum(fn(gen, D, n, gap) for _ in range(TRIALS))
                row.append(hit / TRIALS)
            print(f'    {name:<16}' + ''.join(f'{v:>10.3f}' for v in row))
        print()

    print('READING IT')
    print('  reflect-naive is what the architecture has. If it falls with gap while')
    print('  reflect-equiv holds, the failure is the missing inverse, not the store.')
    print('  carved holding at every gap means disjoint support is enough to protect memory,')
    print('  and it is the one change that needs no new state.')
    print('  outer-naive vs outer-equiv separates "a matrix memory helps" from "a matrix memory')
    print('  is automatically transport-proof" -- conjugation is only benign if the read is')
    print('  transported too.')


if __name__ == '__main__':
    main()
