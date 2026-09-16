"""Does the character diagnostic identify the learned quotient ACROSS runs, or only once?

WHY THIS EXISTS. Paper B derives which character norms a 4-dimensional representation of S_4 can
have (evidence/verify_paperB.py part 3): 2 for a faithful one, 3/4/5 for kernel V_4, 8/10/16 for
kernel A_4, 16 for the trivial one. Those are facts about S_4. But the paper could point to only ONE
measured run confirming a quotient value, which made the character section its weakest claim -- a
derived table with a single data point under it.

THE TEST. For each seed, two independent readings of the same trained model:

    from the ERRORS     coset consistency identifies the kernel N: the smallest normal subgroup
                        whose cosets the model's mistakes respect. Needs the group table.
    from the TRACES     <chi,chi> = (1/|G|) sum_g tr(T_g)^2, and the multiplicities m_i by
                        orthogonality against the character table. Needs no labels and no errors.

They are computed from different things. If the diagnostic works, the trace reading must land in the
set of values the error reading allows -- for every run, not one.

TWO CONDITIONS, both measured rather than assumed:
  homomorphism   <chi,chi> is a character norm only if T_a T_b = T_ab. A run with a large group-law
                 violation is not a representation and its trace sum means nothing. Reported per run.
  full rank      <chi,chi> = 2 certifies faithfulness only at full rank: at d = 3 the value 2 means
                 kernel V_4, at d = 2 kernel A_4. So the effective rank of the state is measured too.

Falsifier, registered before running: a run that is flat, has a group-law violation below HOM_TOL,
has full effective rank, and whose <chi,chi> lies more than 0.25 from every value its
coset-identified kernel permits.

    python evidence/character_table.py
Environment: SEEDS, STEPS, NH, D, LAM, HOM_TOL, THREADS.
"""
from __future__ import annotations

import itertools
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

NH = int(os.environ.get('NH', 4))
D = int(os.environ.get('D', 4))
STEPS = int(os.environ.get('STEPS', 2000))
BS = 256
TRAIN_L = 12
LAM = float(os.environ.get('LAM', 1.0))
HOM_TOL = float(os.environ.get('HOM_TOL', 0.02))
SEEDS = [int(s) for s in os.environ.get('SEEDS', ','.join(str(i) for i in range(16))).split(',')]
torch.set_num_threads(int(os.environ.get('THREADS', 8)))

# ---------------------------------------------------------------------------- S_4 and its algebra
ELS = sorted(itertools.permutations(range(4)))
IX = {p: i for i, p in enumerate(ELS)}
N_G = 24
TAB = torch.tensor([[IX[tuple(b[a[i]] for i in range(4))] for b in ELS] for a in ELS])
INV = [IX[tuple(sorted(range(4), key=lambda i: p[i]))] for p in ELS]


def cycle_type(p):
    seen, t = set(), []
    for i in range(len(p)):
        if i in seen:
            continue
        c, j = 0, i
        while j not in seen:
            seen.add(j); j = p[j]; c += 1
        t.append(c)
    return tuple(sorted(t, reverse=True))


CT = [cycle_type(p) for p in ELS]
# The character table of S_4, indexed by cycle type. Five classes, five irreps.
CHARS = {
    'trivial':  {(1, 1, 1, 1): 1, (2, 1, 1): 1, (2, 2): 1, (3, 1): 1, (4,): 1},
    'sign':     {(1, 1, 1, 1): 1, (2, 1, 1): -1, (2, 2): 1, (3, 1): 1, (4,): -1},
    '2dim':     {(1, 1, 1, 1): 2, (2, 1, 1): 0, (2, 2): 2, (3, 1): -1, (4,): 0},
    'standard': {(1, 1, 1, 1): 3, (2, 1, 1): 1, (2, 2): -1, (3, 1): 0, (4,): -1},
    'std*sign': {(1, 1, 1, 1): 3, (2, 1, 1): -1, (2, 2): -1, (3, 1): 0, (4,): 1},
}
IRREP_DIM = {'trivial': 1, 'sign': 1, '2dim': 2, 'standard': 3, 'std*sign': 3}
# Kernels as explicit subsets, so intersections are computed and not reasoned about.
V4 = {IX[p] for p in ELS if cycle_type(p) in {(1, 1, 1, 1), (2, 2)}}
A4 = {IX[p] for p in ELS if cycle_type(p) in {(1, 1, 1, 1), (2, 2), (3, 1)}}
S4 = set(range(N_G))
IRREP_KER = {'trivial': S4, 'sign': A4, '2dim': V4, 'standard': {IX[tuple(range(4))]},
             'std*sign': {IX[tuple(range(4))]}}
NORMALS = [('1', {IX[tuple(range(4))]}), ('V_4', V4), ('A_4', A4), ('S_4', S4)]


def allowed_norms(ker_set):
    """Every <chi,chi> a 4-dimensional rep of S_4 with exactly this kernel can have."""
    out = set()
    names = list(CHARS)
    for mult in itertools.product(range(5), repeat=5):
        if sum(m * IRREP_DIM[n] for m, n in zip(mult, names)) != D:
            continue
        used = [n for m, n in zip(mult, names) if m]
        if not used:
            continue
        k = set.intersection(*(IRREP_KER[n] for n in used))
        if k == ker_set:
            out.add(sum(m * m for m in mult))
    return sorted(out)


# ---------------------------------------------------------------------------- the model
class Reflect(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.v = nn.Parameter(torch.randn(n, NH, d) * 0.5)
        self.b = nn.Parameter(torch.full((n, NH), 2.0))
        self.h0 = nn.Parameter(torch.randn(d) * 0.3)
        self.out = nn.Linear(d, n)
        self.d = d

    def T(self, h, tok):
        V, B = self.v[tok], self.b[tok]
        for i in range(NH):
            u = F.normalize(V[:, i], dim=-1)
            h = h - B[:, i:i + 1] * (h * u).sum(-1, keepdim=True) * u
        return h

    def forward(self, x):
        h = self.h0[None].expand(x.shape[0], -1)
        o = []
        for t in range(x.shape[1]):
            h = self.T(h, x[:, t])
            o.append(h)
        return self.out(torch.stack(o, 1))


def batch(bs, L, gen):
    x = torch.randint(0, N_G, (bs, L), generator=gen)
    acc = x[:, 0].clone()
    ys = [acc.clone()]
    for t in range(1, L):
        acc = TAB[acc, x[:, t]]
        ys.append(acc.clone())
    return x, torch.stack(ys, 1)


def transport_matrices(m):
    """T_g as an explicit d x d matrix, by pushing the basis through. Exact, not sampled."""
    eye = torch.eye(m.d)
    return torch.stack([m.T(eye, torch.full((m.d,), g)).T for g in range(N_G)])


def measure(m):
    M = transport_matrices(m)                                  # (24, d, d)
    chi = torch.stack([M[g].trace() for g in range(N_G)])
    norm = float((chi ** 2).mean())                            # <chi,chi> = (1/|G|) sum chi(g)^2
    mult = {}
    for name, tbl in CHARS.items():
        mult[name] = float(sum(tbl[CT[g]] * chi[g] for g in range(N_G)) / N_G)
    return chi, norm, mult


def effective_rank(m, gen):
    """How many dimensions the state actually uses. <chi,chi>=2 certifies faithfulness only at d=4."""
    x, _ = batch(256, TRAIN_L, gen)
    with torch.no_grad():
        h = m.h0[None].expand(x.shape[0], -1)
        hs = []
        for t in range(x.shape[1]):
            h = m.T(h, x[:, t]); hs.append(h)
        H = torch.cat(hs, 0)
    s = torch.linalg.svdvals(H - H.mean(0, keepdim=True))
    return int((s > 0.01 * s[0]).sum()), s


def coset_scores(m, gen):
    """Identify the kernel from the model's ERRORS: the smallest N whose cosets they respect."""
    with torch.no_grad():
        x, y = batch(512, 96, gen)
        pred = m(x)[:, -1].argmax(-1)
    true = y[:, -1]
    out = {}
    for name, S in NORMALS:
        same = [int(int(TAB[INV[int(p)], int(t)]) in S) for p, t in zip(pred, true)]
        out[name] = sum(same) / len(same)
    return out, float((pred == true).float().mean())


def run(seed):
    torch.manual_seed(seed)
    m = Reflect(N_G, D)
    op = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(500 + seed)
    for _ in range(STEPS):
        x, y = batch(BS, TRAIN_L, g)
        hh = torch.randn(64, D)
        a = torch.randint(0, N_G, (64,)); b = torch.randint(0, N_G, (64,))
        law = (m.T(m.T(hh, a), b) - m.T(hh, TAB[a, b])).norm(dim=1).mean() / D ** 0.5
        loss = F.cross_entropy(m(x).reshape(-1, N_G), y.reshape(-1)) + LAM * law
        op.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); op.step()

    m.eval()
    ge = torch.Generator().manual_seed(99)
    with torch.no_grad():
        accs = {}
        for L in (12, 96):
            gl = torch.Generator().manual_seed(7 + L)
            x, y = batch(512, L, gl)
            accs[L] = float((m(x)[:, -1].argmax(-1) == y[:, -1]).float().mean())
        hh = torch.randn(512, D)
        a = torch.randint(0, N_G, (512,)); b = torch.randint(0, N_G, (512,))
        viol = float((m.T(m.T(hh, a), b) - m.T(hh, TAB[a, b])).norm(dim=1).mean() / D ** 0.5)
        _, norm, mult = measure(m)
        rank, _ = effective_rank(m, ge)
        cos, raw = coset_scores(m, ge)
    return dict(acc12=accs[12], acc96=accs[96], viol=viol, norm=norm, mult=mult,
                rank=rank, cos=cos, raw=raw)


def main():
    allowed = {name: allowed_norms(S) for name, S in NORMALS}
    print('DOES THE CHARACTER NORM AGREE WITH THE KERNEL THE ERRORS IDENTIFY?')
    print(f'S_4 word problem, n_h={NH}, d={D}, {STEPS} steps, {len(SEEDS)} seeds, '
          f'homomorphism tolerance {HOM_TOL}\n')
    print('Allowed <chi,chi> by kernel, derived from the character table (no data involved):')
    for name, _ in NORMALS:
        print(f'   kernel {name:>3}: {allowed[name]}')
    print()
    hdr = (f'{"sd":>3} {"L=12":>6} {"L=96":>6} {"flat":>5} {"g-law":>7} {"rank":>5} '
           f'{"<X,X>":>7} {"ker(errors)":>12} {"allowed":>14}  verdict')
    print(hdr); print('-' * len(hdr))
    rows = []
    for sd in SEEDS:
        t0 = time.time()
        r = run(sd)
        ker = next((n for n, _ in NORMALS if r['cos'][n] > 0.99), None)
        flat = abs(r['acc12'] - r['acc96']) < 0.05
        hom = r['viol'] < HOM_TOL
        al = allowed.get(ker, [])
        dev = min((abs(r['norm'] - v) for v in al), default=float('nan'))
        if not hom:
            verd = 'not a hom.'
        elif ker is None:
            verd = 'no kernel'
        elif dev < 0.25:
            verd = 'AGREES'
        else:
            verd = 'VIOLATION'
        rows.append(dict(sd=sd, ker=ker, hom=hom, flat=flat, dev=dev, verd=verd, **r))
        print(f'{sd:>3} {r["acc12"]:>6.3f} {r["acc96"]:>6.3f} {str(flat):>5} {r["viol"]:>7.4f} '
              f'{r["rank"]:>5} {r["norm"]:>7.3f} {str(ker):>12} {str(al):>14}  {verd}'
              f'   ({time.time()-t0:.0f}s)', flush=True)

    print('\nMULTIPLICITIES (m_i by orthogonality; a real representation gives near-integers)')
    h2 = (f'{"sd":>3} {"trivial":>8} {"sign":>7} {"2dim":>7} {"standard":>9} {"std*sign":>9} '
          f'{"sum m^2":>8}  kernel implied by m')
    print(h2); print('-' * len(h2))
    for r in rows:
        mu = r['mult']
        used = [n for n in CHARS if abs(mu[n]) > 0.5]
        ki = set.intersection(*(IRREP_KER[n] for n in used)) if used else None
        kn = next((n for n, S in NORMALS if S == ki), '?') if ki is not None else '-'
        print(f'{r["sd"]:>3} {mu["trivial"]:>8.3f} {mu["sign"]:>7.3f} {mu["2dim"]:>7.3f} '
              f'{mu["standard"]:>9.3f} {mu["std*sign"]:>9.3f} '
              f'{sum(v*v for v in mu.values()):>8.3f}  {kn}')

    hom = [r for r in rows if r['hom'] and r['ker'] is not None]
    agree = [r for r in hom if r['verd'] == 'AGREES']
    viol = [r for r in hom if r['verd'] == 'VIOLATION']
    print(f'\nSCORED: {len(hom)}/{len(rows)} runs are homomorphisms (g-law < {HOM_TOL}) with an '
          f'identified kernel.')
    print(f'  the trace reading agrees with the error reading on {len(agree)}/{len(hom)}')
    print(f'  registered falsifier (a scored run deviating > 0.25): {len(viol)} hits -> '
          f'{"NOT falsified" if not viol else "FALSIFIED"}')
    byk = {}
    for r in hom:
        byk.setdefault(r['ker'], []).append(r['norm'])
    print('\n  measured <chi,chi> by kernel, which is the coverage the paper was missing:')
    for n, _ in NORMALS:
        vs = byk.get(n, [])
        print(f'    kernel {n:>3}: allowed {str(allowed[n]):>14}   measured '
              f'{[round(v, 3) for v in vs] if vs else "(no run landed here)"}')


if __name__ == '__main__':
    main()
