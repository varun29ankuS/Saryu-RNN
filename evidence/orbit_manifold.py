"""Are stored items POINTS or ORBITS? Manifold radius of a single item's contribution.

THE IDEA. Item i enters the state as b_i and is then multiplied by M_{i->T} = prod_{s>i} a_s T_s,
a product of rotations determined by WHATEVER TOKENS HAPPEN TO FOLLOW IT. So across sequences the
same (key, value) pair does not land at a point -- it sweeps an ORBIT. And beta is pinned at 2.000
in trained models (trained_geometry.txt), which makes every transport exactly norm-preserving, so
that orbit lies on a SPHERE: a closed manifold, not a contracting one.

Manifold capacity theory (Gardner; Chung, Lee & Sompolinsky) says linear separability collapses as
manifold RADIUS grows. The relevant question is therefore not how many vectors fit in dh dimensions
-- exponentially many -- but how many ORBITS can be separated.

It also gives the flat n_h sweep a reading that nothing else has. More reflections bind harder AND
smear the orbit wider, because there is more rotation between write and read. Fewer do neither.
A genuine trade with an optimum, rather than a knob that should have monotonically helped.

HOW THE CONTRIBUTION IS ISOLATED, exactly. The recurrence is LINEAR in the writes: h_T = sum_i
M_i b_i. So changing only item i's value from v to v' changes the state by exactly M_i (b(v)-b(v')).
A finite difference therefore isolates that one item's contribution with no approximation, which is
the one place the linear-state property is useful rather than limiting.

WHAT IS MEASURED, over many random contexts (other pairs, fillers and query all resampled):
    R_M   spread of the normalised contribution direction about its mean.
          0 = the item always lands in the same direction (a POINT).
          ~1.4 = directions are mutually orthogonal (fully smeared over the sphere).
    D_M   participation ratio of the residual covariance: how many directions the orbit occupies.
    |mu|  length of the mean direction. 1 = perfectly concentrated, 0 = no preferred direction.

REGISTERED PREDICTIONS, before running:
  P1  n=2, which this model solves, has a SMALLER R_M than n=4, which it never solves.
  P2  R_M grows with n_h, since more reflections mean more rotation between write and read.
  P3  if R_M is already near the fully-smeared value at n=2 -- the configuration that WORKS -- then
      orbit spread is not what separates the two regimes and the manifold account fails.

FALSIFIER: R_M is the same at n=2 and n=4. Then the geometry of the stored item is not what changes
across the wall, and this is another falsified geometric story -- the fifth in
docs/memory_geometry.md.

    python evidence/orbit_manifold.py
Environment: NPAIRS, NHS, GAP, STEPS, D, H, NL, SEED, CTX, THREADS.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                     # noqa: E402
from binding_long import Muon                                        # noqa: E402

PAIRS = [int(x) for x in os.environ.get('NPAIRS', '2,4').split(',')]
NHS = [int(x) for x in os.environ.get('NHS', '2,8').split(',')]
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 3000))
D = int(os.environ.get('D', 128))
H = int(os.environ.get('H', 8))
NL = int(os.environ.get('NL', 2))
SEED = int(os.environ.get('SEED', 0))
CTX = int(os.environ.get('CTX', 400))      # how many random contexts define the orbit
NENT = 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 2)))


def build(rng, n, k0, v0):
    """A sequence whose FIRST pair is pinned to (k0, v0); everything after is resampled."""
    pool = np.setdiff1d(np.arange(NENT), np.array([k0, v0]))
    ent = rng.choice(pool, size=2 * (n - 1), replace=False)
    ks = [k0] + [int(x) for x in ent[:n - 1]]
    vs = [v0] + [int(x) for x in ent[n - 1:]]
    rest = np.setdiff1d(pool, ent)
    seq = []
    for k, v in zip(ks, vs):
        seq += [k, v]
    seq += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, n))
    seq += [SEP, ks[i]]
    return seq


@torch.no_grad()
def state_at_query(m, x):
    h = m.emb(x)
    ve = m.vemb(x) if m.vemb is not None else None
    for mix, ffn in zip(m.mix, m.ffn):
        h = h + mix(h, ve)
        h = h + ffn(h)
    return m.lnf(h)[:, -1]


def train(n, nh, seed):
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT + 2, D, NL, nh=nh, H=H)
    p = list(m.parameters())
    opt = Muon(p, lr=3e-4)
    rng = np.random.default_rng(1000 + seed)
    for s in range(STEPS):
        ent = rng.choice(NENT, size=(32, 2 * n))
        xs, ys = [], []
        for row in ent:
            u = np.unique(row)
            if len(u) < 2 * n:
                u = rng.choice(NENT, size=2 * n, replace=False)
            ks, vs = u[:n], u[n:]
            seq = []
            for k, v in zip(ks, vs):
                seq += [int(k), int(v)]
            rest = np.setdiff1d(np.arange(NENT), u)
            seq += [int(z) for z in rng.choice(rest, size=GAP, replace=True)]
            i = int(rng.integers(0, n))
            seq += [SEP, int(ks[i])]
            xs.append(seq); ys.append(int(vs[i]))
        x, y = torch.tensor(xs), torch.tensor(ys)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(p, 1.0); opt.step()
    m.eval()
    return m


def orbit(m, n, seed=31337):
    """Finite-difference one item's contribution across many following-contexts."""
    rng = np.random.default_rng(seed)
    dirs = []
    for _ in range(CTX):
        k0 = int(rng.integers(0, NENT))
        v0, v1 = rng.choice(np.setdiff1d(np.arange(NENT), [k0]), size=2, replace=False)
        s0 = build(rng, n, k0, int(v0))
        s1 = list(s0); s1[1] = int(v1)                 # change ONLY that item's value
        a = state_at_query(m, torch.tensor([s0, s1]))
        d = a[0] - a[1]                                # = M_i (b(v0) - b(v1)), exactly
        dirs.append(F.normalize(d, dim=-1))
    U = torch.stack(dirs)
    mu = U.mean(0)
    R = float(torch.sqrt(((U - mu) ** 2).sum(-1).mean()))
    C = (U - mu).T @ (U - mu) / len(U)
    lam = torch.linalg.eigvalsh(C).clamp_min(0)
    DM = float(lam.sum() ** 2 / (lam ** 2).sum())
    return R, DM, float(mu.norm())


def main():
    print(f'ORBIT MANIFOLD  d={D} H={H} nl={NL}, {STEPS} steps, {CTX} contexts per measurement')
    print('an item enters as b_i then is rotated by every following token; across contexts it')
    print('sweeps an orbit. R_M ~ 0 means a POINT, R_M ~ 1.41 means fully smeared (orthogonal).\n')
    print(f'{"pairs":>6} {"nh":>4} {"acc":>7} {"R_M":>7} {"D_M":>7} {"|mu|":>7}')
    print('-' * 44)
    for nh in NHS:
        for n in PAIRS:
            m = train(n, nh, SEED)
            rng = np.random.default_rng(7)
            xs, ys = [], []
            for _ in range(128):
                k0 = int(rng.integers(0, NENT))
                v0 = int(rng.choice(np.setdiff1d(np.arange(NENT), [k0])))
                s = build(rng, n, k0, v0)
                xs.append(s)
                ys.append(s[1] if s[-1] == s[0] else None)
            keep = [(a, b) for a, b in zip(xs, ys) if b is not None]
            acc = float('nan')
            if keep:
                with torch.no_grad():
                    xt = torch.tensor([a for a, _ in keep])
                    yt = torch.tensor([b for _, b in keep])
                    acc = float((m(xt)[:, -1].argmax(-1) == yt).float().mean())
            R, DM, mn = orbit(m, n)
            print(f'{n:>6} {nh:>4} {acc:>7.3f} {R:>7.3f} {DM:>7.1f} {mn:>7.3f}', flush=True)
    print('\nREAD')
    print('  R_M smaller where the model WORKS (n=2) -> orbit spread is what the wall is about')
    print('  R_M the same at n=2 and n=4             -> falsified, like the rest of the geometry')


if __name__ == '__main__':
    main()
