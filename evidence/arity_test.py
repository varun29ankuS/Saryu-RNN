"""Is the limit arity or capacity? Hold the state size fixed and change only the factorisation.

THE CLAIM UNDER TEST. Today's finding is that a vector state binds one association because
retrieval is a single matched filter: the state is a sum, h = sum_i A_i c_i, and at read time one
operator must annihilate every non-matching term. If that is right, the fix is not a BIGGER state
but a PRODUCT state -- n independent factors, each separately addressable, giving n filters.

THE CONTROL THAT MAKES IT A TEST. Both arms get exactly D numbers of state.

    single    one factor of dimension D. All n pairs are superposed into it and read with one
              operator. This is what the recurrence does.
    product   n slots of dimension D/n. One pair per slot, read by comparing the query against
              every slot. This is what a matrix state does -- rows instead of a sum.

Same memory, same precision, same everything except whether the state is one factor or n. If the
product arm wins, the limit is ARITY. If both fail together as n grows, the limit is CAPACITY and
today's conclusion is wrong.

Note the product arm is HANDICAPPED on capacity: its slots are D/n wide, so each individual slot is
n times smaller than the single arm's whole state. If it still wins, the result is not a capacity
artefact in its favour.

REGISTERED PREDICTION, before running, at D = 64:
  P1  single  ~1.0 at n=2, then collapses to ~1/n for n >= 3    (one filter, one bit)
  P2  product stays high across n=2,3,4,8 despite narrower slots (n filters)
  P3  the gap widens with n

FALSIFIER: product degrades at the same n as single. Then factorisation is not what matters and the
arity account is wrong.

    python evidence/arity_test.py
Environment: D, PAIRS, TRIALS, KBIND, SEED.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

D = int(os.environ.get('D', 64))
PAIRS = [int(x) for x in os.environ.get('PAIRS', '2,3,4,8').split(',')]
TRIALS = int(os.environ.get('TRIALS', 400))
KBIND = int(os.environ.get('KBIND', 2))     # reflections per bind; 2 = the shipped n_h
SEED = int(os.environ.get('SEED', 0))


def unit(gen, *shape):
    return F.normalize(torch.randn(*shape, generator=gen), dim=-1)


def reflect(us, v):
    for u in us:
        v = v - 2 * (u @ v) * u
    return v


def single(gen, n, d):
    """One factor of dimension d. All pairs superposed; retrieve with the key's own operator."""
    keys = unit(gen, n, KBIND, d)
    vals = unit(gen, n, d)
    h = torch.zeros(d)
    for i in range(n):
        h = h + reflect(list(keys[i]), vals[i])
    j = int(torch.randint(0, n, (1,), generator=gen))
    read = reflect(list(reversed(list(keys[j]))), h)       # the single matched filter
    return int((vals @ read).argmax()) == j


def product(gen, n, d):
    """n slots of dimension d//n: one pair per slot, retrieved by comparing the query to each.

    This is a matrix state: keys live in separate rows and are never summed, so the query can be
    matched against all of them at once."""
    ds = max(2, d // n)                                     # each slot is NARROWER than d
    kslot = unit(gen, n, ds)                                # the key stored in each slot
    vslot = unit(gen, n, ds)
    j = int(torch.randint(0, n, (1,), generator=gen))
    q = kslot[j]                                            # query the j-th key
    sim = kslot @ q                                         # compare against EVERY slot at once
    read = (F.softmax(sim * 8.0, dim=0)[:, None] * vslot).sum(0)
    return int((vslot @ read).argmax()) == j


def hashed(gen, n, d, H=8):
    """Content-selected slot: slot = hash(key) mod H. The router WITHOUT key-comparison.

    This is what a write-time router on the shipped architecture would be: the model has H = 8
    per-head gates (model.py:164), so a content-selected write sends each key to one of H blocks.
    Unlike `product` there is no stored key to compare against -- the address is a function of the
    key alone, so two keys landing on the same head SUPERPOSE and neither can be recovered cleanly.

    This is the arm that decides whether a router is a real third option or just a broken matrix
    state. Collisions are pure birthday statistics: P(no collision) = H!/((H-n)! H^n), which is
    0.41 at n = 4 and 0.002 at n = 8 for H = 8, no matter how good the hash is."""
    ds = max(2, d // H)
    vals = unit(gen, n, ds)
    slot_of = torch.randint(0, H, (n,), generator=gen)       # an arbitrary (ideal) hash
    store = torch.zeros(H, ds)
    for i in range(n):
        store[slot_of[i]] += vals[i]                         # collisions SUPERPOSE
    j = int(torch.randint(0, n, (1,), generator=gen))
    read = store[slot_of[j]]                                 # address by the query's own hash
    return int((vals @ read).argmax()) == j


def nested(gen, n, d, levels=2):
    """Hierarchical addressing: m groups x m slots, m = ceil(n ** (1/levels)).

    Each block stays d//m wide no matter how large n is -- depth is spent instead of width,
    which is the trade a flat partition cannot make. Retrieval is `levels` sequential
    comparisons of m items each, rather than one comparison of n."""
    m = max(2, math.ceil(n ** (1.0 / levels)))
    ds = max(2, d // m)                                     # block width, INDEPENDENT of n
    # a key per level: the query is routed group-by-group, then to a slot
    addr = [unit(gen, m, ds) for _ in range(levels)]        # level keys
    slots = unit(gen, m ** levels, ds)                      # the stored values
    j = int(torch.randint(0, min(n, m ** levels), (1,), generator=gen))
    digits, t = [], j
    for _ in range(levels):
        digits.append(t % m); t //= m
    # Digits are extracted little-endian (t % m first), so the index must be rebuilt the same
    # way. Rebuilding big-endian reverses the address and only lands when the digits coincide.
    picked = 0
    for lv in range(levels):                                # one selection per level
        q = addr[lv][digits[lv]]
        sim = F.softmax((addr[lv] @ q) * 8.0, dim=0)
        picked += int(sim.argmax()) * (m ** lv)
    read = slots[picked]
    return int((slots[:min(n, m ** levels)] @ read).argmax()) == j


def main():
    print(f'ARITY vs CAPACITY  total state D = {D} for BOTH arms, {TRIALS} trials, '
          f'{KBIND} reflections per bind')
    print('single  = one factor of D, all pairs summed, one matched filter')
    print(f'product = n slots of D/n (narrower!), one pair each, compared all at once\n')
    print(f'{"pairs":>6} {"chance":>8} {"single":>9} {"hashed":>9} {"product":>9} {"nested":>9}   '
          f'{"P(no clash)":>11}')
    print('-' * 72)
    for n in PAIRS:
        g1 = torch.Generator().manual_seed(SEED)
        g2 = torch.Generator().manual_seed(SEED)
        g3 = torch.Generator().manual_seed(SEED)
        g4 = torch.Generator().manual_seed(SEED)
        s = sum(single(g1, n, D) for _ in range(TRIALS)) / TRIALS
        p = sum(product(g2, n, D) for _ in range(TRIALS)) / TRIALS
        nst = sum(nested(g3, n, D) for _ in range(TRIALS)) / TRIALS
        hsh = sum(hashed(g4, n, D) for _ in range(TRIALS)) / TRIALS
        pc = math.prod((8 - i) / 8 for i in range(n)) if n <= 8 else 0.0
        print(f'{n:>6} {1/n:>8.3f} {s:>9.3f} {hsh:>9.3f} {p:>9.3f} {nst:>9.3f}   '
              f'{pc:>11.3f}')
    print('\nREAD')
    print('  single high at n=2 then collapsing to 1/n  -> one filter, one bit: ARITY')
    print('  product staying high on NARROWER slots     -> factorisation is what matters')
    print('  both collapsing together                   -> capacity, and the arity account fails')


if __name__ == '__main__':
    main()
