"""Two things nobody seems to try, both implied by what we measured.

Measured: training at FIXED length selects a length-specific shortcut that cross-entropy
cannot distinguish from the homomorphism (trained: 0.777 at L=12, 0.048 at L=96).

Candidate 1  MIXED-LENGTH TRAINING. If the shortcut is length-specific, training on several
             lengths at once should break it -- no constraint, no group table, no spectral
             machinery. The homomorphism becomes the only thing that fits all lengths.

Candidate 2  WEAK READOUT. Ours is a 2-layer MLP (128 hidden) -- ample capacity to decode a
             messy state and HIDE the transport's failure. A linear readout forces the
             transport to form the clean quotient itself.

2x2, 4 seeds each, evaluated far outside any training length.
"""
import itertools, time, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)
K, BS, STEPS = 4, 256, 2500
EVAL_L = [12, 24, 48, 96]
els = list(itertools.permutations(range(K))); idx = {p: i for i, p in enumerate(els)}
TAB = torch.tensor([[idx[tuple(b[a[i]] for i in range(K))] for b in els] for a in els]); N = len(els)

class Model(nn.Module):
    def __init__(s, weak):
        super().__init__()
        s.v = nn.Parameter(torch.randn(N, 3, K) * 0.5)
        s.b = nn.Parameter(torch.full((N, 3), 2.0))
        s.h0 = nn.Parameter(torch.randn(K) * 0.3)
        s.out = nn.Linear(K, N) if weak else nn.Sequential(
            nn.Linear(K, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, N))
    def T(s, h, tok):
        V = s.v[tok]; B = s.b[tok]
        for i in range(3):
            u = F.normalize(V[:, i], dim=-1)
            h = h - B[:, i:i+1] * (h * u).sum(-1, keepdim=True) * u
        return h
    def forward(s, x):
        h = s.h0[None].expand(x.shape[0], -1); o = []
        for t in range(x.shape[1]): h = s.T(h, x[:, t]); o.append(h)
        return s.out(torch.stack(o, 1))
    def gl(s):
        with torch.no_grad():
            e = 0.
            for _ in range(10):
                h = torch.randn(300, K); a = torch.randint(0, N, (300,)); b = torch.randint(0, N, (300,))
                e = max(e, (s.T(s.T(h, a), b) - s.T(h, TAB[a, b])).norm(dim=1).mean().item() / K**.5)
            return e

def batch(bs, L, g):
    x = torch.randint(0, N, (bs, L), generator=g)
    acc = x[:, 0].clone(); ys = [acc.clone()]
    for t in range(1, L): acc = TAB[acc, x[:, t]]; ys.append(acc.clone())
    return x, torch.stack(ys, 1)

def run(mixed, weak, seed):
    torch.manual_seed(seed); m = Model(weak)
    op = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=.01)
    g = torch.Generator().manual_seed(500 + seed)
    LS = [4, 6, 8, 10, 12] if mixed else [12]
    for i in range(STEPS):
        L = LS[i % len(LS)]
        x, y = batch(BS, L, g)
        loss = F.cross_entropy(m(x).reshape(-1, N), y.reshape(-1))
        op.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); op.step()
    ge = torch.Generator().manual_seed(99); accs = []
    with torch.no_grad():
        for L in EVAL_L:
            a = 0.
            for _ in range(6):
                x, y = batch(256, L, ge); a += (m(x)[:, -1].argmax(-1) == y[:, -1]).float().mean().item()
            accs.append(a / 6)
    return accs, m.gl()

print('WHAT NOBODY TRIES: mixed-length training and a weak readout')
print(f'S_{K}, |G|={N}. Mixed trains on lengths 4-12; fixed trains only on 12.')
print(f'Evaluated at {EVAL_L}; chance {1/N:.3f}. Reference: exact solution = 1.000 everywhere.\n')
print(f'{"length":>8} {"readout":>9} {"solved":>7} ' + ' '.join(f'{("L="+str(L)):>7}' for L in EVAL_L) + f' {"grp-law":>8}')
print('-' * 66, flush=True)
for mixed in [False, True]:
    for weak in [False, True]:
        res = [run(mixed, weak, s) for s in range(4)]
        mean = [sum(r[0][i] for r in res)/4 for i in range(4)]
        gl = sum(r[1] for r in res)/4
        solved = sum(r[0][-1] > 0.9 for r in res)
        print(f'{"mixed" if mixed else "fixed":>8} {"linear" if weak else "MLP":>9} {solved:>4}/4 '
              + ' '.join(f'{v:>7.3f}' for v in mean) + f' {gl:>8.4f}', flush=True)
