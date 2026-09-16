"""Three groups, three different predictions, from pure algebra.

Observed on S_4: every outcome is flat in length and sits on 1/|N| for N a normal
subgroup -- 1.000, 0.250, 0.083, 0.042. If that is real structure and not four
coincidences, the RUNG SET is a property of the group, which the model cannot see.

  S_4  order 24. Normals 1, V_4, A_4, S_4      -> rungs 1.000 0.250 0.083 0.042
       BOTH extensions SPLIT (S_4 = V_4 : S_3, and A_4 -> S_4 -> Z_2 split by a
       transposition), so a section exists: a stuck seed is an OPTIMISER failure.
  Q_8  order 8. Hamiltonian -- every subgroup normal: 1, Z_2, three Z_4, Q_8
                                              -> rungs 1.000 0.500 0.250 0.125
       1 -> Z_2 -> Q_8 -> V_4 is NON-SPLIT (Q_8 has a unique element of order 2, so
       it contains no V_4). No section exists: that plateau is an ALGEBRAIC obstruction.
  A_5  order 60. SIMPLE: normals are only 1 and A_5 -> rungs 1.000 and 0.017 ONLY.
       No intermediate plateau can exist.

Falsifier: any stable, length-flat accuracy on A_5 strictly between 0.017 and 1.000.

Kernel is measured directly, not assumed: cluster the learned transports and read
|ker| = |G| / (number of distinct transports). Predicted accuracy is then 1/|ker|.
"""
import itertools, time, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
NH, BS, STEPS, TRAIN_L = 4, 256, 2000, 12

def s4():
    els = list(itertools.permutations(range(4))); ix = {p: i for i, p in enumerate(els)}
    T = [[ix[tuple(b[a[i]] for i in range(4))] for b in els] for a in els]
    return 'S_4', torch.tensor(T), 4, {1: 1.0, 4: .25, 12: 1/12, 24: 1/24}

def a5():
    def par(p): return sum(1 for i in range(5) for j in range(i+1,5) if p[i]>p[j]) % 2
    els = [p for p in itertools.permutations(range(5)) if par(p) == 0]
    ix = {p: i for i, p in enumerate(els)}
    T = [[ix[tuple(b[a[i]] for i in range(5))] for b in els] for a in els]
    return 'A_5', torch.tensor(T), 5, {1: 1.0, 60: 1/60}

def q8():
    BM = [[(0,0),(1,0),(2,0),(3,0)], [(1,0),(0,1),(3,0),(2,1)],
          [(2,0),(3,1),(0,1),(1,0)], [(3,0),(2,0),(1,1),(0,1)]]
    T = [[0]*8 for _ in range(8)]
    for x in range(8):
        for y in range(8):
            rb, f = BM[x % 4][y % 4]
            T[x][y] = rb + 4 * ((x//4 + y//4 + f) % 2)
    return 'Q_8', torch.tensor(T), 4, {1: 1.0, 2: .5, 4: .25, 8: .125}

class Model(nn.Module):
    def __init__(s, N, d):
        super().__init__()
        s.v = nn.Parameter(torch.randn(N, NH, d) * 0.5)
        s.b = nn.Parameter(torch.full((N, NH), 2.0))
        s.h0 = nn.Parameter(torch.randn(d) * 0.3)
        s.out = nn.Linear(d, N); s.N, s.d = N, d
    def T(s, h, tok):
        V = s.v[tok]; B = s.b[tok]
        for i in range(NH):
            u = F.normalize(V[:, i], dim=-1)
            h = h - B[:, i:i+1] * (h * u).sum(-1, keepdim=True) * u
        return h
    def forward(s, x):
        h = s.h0[None].expand(x.shape[0], -1); o = []
        for t in range(x.shape[1]):
            h = s.T(h, x[:, t]); o.append(h)
        return s.out(torch.stack(o, 1))

def make(TAB, N):
    def batch(bs, L, g):
        x = torch.randint(0, N, (bs, L), generator=g)
        acc = x[:, 0].clone(); ys = [acc.clone()]
        for t in range(1, L):
            acc = TAB[acc, x[:, t]]; ys.append(acc.clone())
        return x, torch.stack(ys, 1)
    return batch

def kernel_size(m, N):
    """cluster transports; |ker| = |G| / #distinct"""
    with torch.no_grad():
        h = torch.randn(400, m.d)
        M = torch.stack([m.T(h, torch.full((400,), g)).mean(0) for g in range(N)])
        D = torch.cdist(M[None], M[None])[0] / (m.d ** .5)
        thr = 0.05
        lab = [-1]*N; c = 0
        for i in range(N):
            if lab[i] >= 0: continue
            for j in range(N):
                if D[i, j] < thr: lab[j] = c
            c += 1
        return N / max(c, 1), c

def run(name, TAB, d, rungs, seeds=(0,1,2,3)):
    N = TAB.shape[0]; batch = make(TAB, N)
    print(f'\n{name}  |G|={N}  d={d}  n_h={NH}   predicted rungs: '
          + '  '.join(f'1/{k}={v:.3f}' for k, v in rungs.items()))
    print(f'  {"sd":>3} {"L=12":>7} {"L=96":>7} {"flat?":>6} {"|ker|":>6} {"1/|ker|":>8} {"nearest rung":>13}', flush=True)
    for seed in seeds:
        t0 = time.time(); torch.manual_seed(seed); m = Model(N, d)
        op = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=.01)
        g = torch.Generator().manual_seed(500 + seed)
        for _ in range(STEPS):
            x, y = batch(BS, TRAIN_L, g)
            hl = 0.
            hh = torch.randn(64, d); a = torch.randint(0,N,(64,)); b = torch.randint(0,N,(64,))
            hl = (m.T(m.T(hh,a),b) - m.T(hh,TAB[a,b])).norm(dim=1).mean() / (d**.5)
            loss = F.cross_entropy(m(x).reshape(-1,N), y.reshape(-1)) + 1.0 * hl
            op.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); op.step()
        ge = torch.Generator().manual_seed(99); acc = {}
        with torch.no_grad():
            for L in (12, 96):
                a_ = 0.
                for _ in range(4):
                    x, y = batch(256, L, ge)
                    a_ += (m(x)[:, -1].argmax(-1) == y[:, -1]).float().mean().item()
                acc[L] = a_/4
        ks, nc = kernel_size(m, N)
        flat = abs(acc[12]-acc[96]) < 0.05
        near = min(rungs.values(), key=lambda v: abs(v-acc[96]))
        print(f'  {seed:>3} {acc[12]:>7.3f} {acc[96]:>7.3f} {str(flat):>6} {ks:>6.1f} '
              f'{1/max(ks,1e-9):>8.3f} {near:>13.3f}   ({time.time()-t0:.0f}s)', flush=True)

if __name__ == '__main__':
    print('RUNG STRUCTURE IS A PROPERTY OF THE GROUP, NOT THE MODEL')
    print('Falsifier: a stable length-flat accuracy on A_5 strictly between 0.017 and 1.000.')
    for f in (q8, s4, a5):
        run(*f())
