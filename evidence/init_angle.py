"""Does the near-involution bias of high-dim random init hurt state tracking?

Measured: two random unit vectors in R^d are near-orthogonal, so the product of their
reflections is a rotation by ~pi -- an involution. ||T^2 - I|| / ||T - I|| falls from
0.73 at d=4 to 0.11 at d=256. Wider init = stronger order-2 bias.

If that bias matters, widening the state should HURT, and setting the initial angle away
from orthogonal should undo it. Rotation angle = 2 * angle(u1, u2), so <u1,u2> = cos(phi)
gives rotation 2*phi; phi = pi/2 (random) -> pi (involution), phi = pi/3 -> 2pi/3 (order 3).
"""
import itertools, math, time, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
K, TRAIN_L, BS, STEPS = 4, 12, 256, 1500
EVAL_L = [12, 24, 48, 96]
els = list(itertools.permutations(range(K)))
idx = {p: i for i, p in enumerate(els)}
TAB = torch.tensor([[idx[tuple(b[a[i]] for i in range(K))] for b in els] for a in els])
N = len(els)

class Model(nn.Module):
    def __init__(s, d, phi):
        super().__init__()
        v = torch.randn(N, 2, d)
        if phi is not None:                      # force angle(u1,u2) = phi
            u1 = F.normalize(v[:, 0], dim=-1)
            w = v[:, 1] - (v[:, 1] * u1).sum(-1, keepdim=True) * u1
            w = F.normalize(w, dim=-1)
            v[:, 1] = math.cos(phi) * u1 + math.sin(phi) * w
        s.v = nn.Parameter(v)
        s.h0 = nn.Parameter(torch.randn(d) * 0.3)
        s.d = d
        s.out = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, N))
    def T(s, h, tok):
        V = s.v[tok]
        for i in range(2):
            u = F.normalize(V[:, i], dim=-1)
            h = h - 2.0 * (h * u).sum(-1, keepdim=True) * u
        return h
    def forward(s, x):
        h = s.h0[None].expand(x.shape[0], -1); o = []
        for t in range(x.shape[1]):
            h = s.T(h, x[:, t]); o.append(h)
        return s.out(torch.stack(o, 1))
    def involution_err(s):
        """||T^2 - I|| / ||T - I||, averaged over tokens. 0 = pure involution."""
        with torch.no_grad():
            I = torch.eye(s.d); rs = []
            for g in range(N):
                cols = []
                for j in range(s.d):
                    e = I[j][None]
                    cols.append(s.T(e, torch.tensor([g]))[0])
                T = torch.stack(cols, 1)
                rs.append(((T @ T - I).norm() / (T - I).norm()).item())
            return sum(rs) / len(rs)

def batch(bs, L, g):
    x = torch.randint(0, N, (bs, L), generator=g)
    acc = x[:, 0].clone(); ys = [acc.clone()]
    for t in range(1, L):
        acc = TAB[acc, x[:, t]]; ys.append(acc.clone())
    return x, torch.stack(ys, 1)

def run(d, phi, seed):
    torch.manual_seed(seed); m = Model(d, phi)
    inv0 = m.involution_err()
    op = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=.01)
    g = torch.Generator().manual_seed(500 + seed)
    for _ in range(STEPS):
        x, y = batch(BS, TRAIN_L, g)
        loss = F.cross_entropy(m(x).reshape(-1, N), y.reshape(-1))
        op.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); op.step()
    ge = torch.Generator().manual_seed(99); accs = []
    with torch.no_grad():
        for L in EVAL_L:
            a = 0.
            for _ in range(4):
                x, y = batch(256, L, ge)
                a += (m(x)[:, -1].argmax(-1) == y[:, -1]).float().mean().item()
            accs.append(a / 4)
    return accs, inv0, m.involution_err()

if __name__ == '__main__':
    print('DOES HIGH-DIM INIT BIAS TRANSPORTS TOWARD ORDER 2, AND DOES IT MATTER?')
    print(f'S_{K}, n_h=2, trained at L={TRAIN_L}, evaluated at {EVAL_L}. chance={1/N:.3f}\n')
    hdr = (f'{"d":>5} {"init":>10} {"sd":>3} ' + ' '.join(f'{("L="+str(L)):>7}' for L in EVAL_L)
           + f' {"inv@init":>9} {"inv@end":>8}')
    print(hdr); print('-' * len(hdr), flush=True)
    for d in [8, 32, 128]:
        for name, phi in [('random', None), ('phi=pi/3', math.pi / 3)]:
            for seed in [0, 1]:
                t0 = time.time()
                accs, i0, i1 = run(d, phi, seed)
                print(f'{d:>5} {name:>10} {seed:>3} ' + ' '.join(f'{x:>7.3f}' for x in accs)
                      + f' {i0:>9.3f} {i1:>8.3f}   ({time.time()-t0:.0f}s)', flush=True)
    print('\n  inv = ||T^2 - I|| / ||T - I||.  0 = pure involution, ~1 = generic element.')
    print('  If the bias matters: random init degrades as d grows; phi=pi/3 does not.')
