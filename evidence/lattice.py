"""Does the model land on the NORMAL SUBGROUP LATTICE of S_4?

Observed accuracies across every run on this task, all flat in length (= homomorphisms):
    1.000   0.255   0.083   0.042
Normal subgroups of S_4 are 1, V_4 (4), A_4 (12), S_4 (24). A model that learns G/N gets
the coset right and guesses inside it, so its ceiling is 1/|N|:
    1/1 = 1.000   1/4 = 0.250   1/12 = 0.083   1/24 = 0.042
Exact match on all four rungs. This generalises the measured abelian ceiling 1/|[G,G]|,
which for S_4 is 1/|A_4| = 0.083 -- one particular rung, not a separate law.

Coincidence of four numbers, or real structure? Decisive test: if the model learned G/N,
its errors must respect cosets -- predicted and true element in the SAME coset of N,
essentially always. A model that is merely bad has no reason to respect any coset.

Reports, for each trained model, the fraction of predictions landing in the correct coset
of each candidate N. The learned kernel is the smallest N scoring ~1.0.
"""
import itertools, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
K, TRAIN_L, BS, STEPS = 4, 12, 256, 2500
els = list(itertools.permutations(range(K)))
idx = {p: i for i, p in enumerate(els)}
TAB = torch.tensor([[idx[tuple(b[a[i]] for i in range(K))] for b in els] for a in els])
N = len(els)

def parity(p):
    return sum(1 for i in range(K) for j in range(i+1, K) if p[i] > p[j]) % 2
A4 = [i for i, p in enumerate(els) if parity(p) == 0]
V4 = [idx[p] for p in [(0,1,2,3), (1,0,3,2), (2,3,0,1), (3,2,1,0)]]
SUBS = {'1 (faithful)': [idx[(0,1,2,3)]], 'V_4': V4, 'A_4': A4, 'S_4': list(range(N))}

def cosets(H):
    """partition G into left cosets gH; return label per element"""
    lab = [-1]*N; c = 0
    for g in range(N):
        if lab[g] >= 0: continue
        for h in H: lab[TAB[g, h].item()] = c
        c += 1
    return torch.tensor(lab), c

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
        for t in range(x.shape[1]):
            h = s.T(h, x[:, t]); o.append(h)
        return s.out(torch.stack(o, 1))
    def hom_loss(s, n=64):
        h = torch.randn(n, K)
        a = torch.randint(0, N, (n,)); b = torch.randint(0, N, (n,))
        return (s.T(s.T(h, a), b) - s.T(h, TAB[a, b])).norm(dim=1).mean() / (K ** .5)
    def faith_loss(s, n=128):
        h = torch.randn(n, K)
        M = torch.stack([s.T(h, torch.full((n,), g)) for g in range(N)]).mean(1)
        D = torch.cdist(M[None], M[None])[0] / (K ** .5)
        return F.relu(0.2 - D[~torch.eye(N, dtype=bool)]).mean()

def batch(bs, L, g):
    x = torch.randint(0, N, (bs, L), generator=g)
    acc = x[:, 0].clone(); ys = [acc.clone()]
    for t in range(1, L):
        acc = TAB[acc, x[:, t]]; ys.append(acc.clone())
    return x, torch.stack(ys, 1)

def train(a_hom, c_faith, seed):
    torch.manual_seed(seed); m = Model(True)
    op = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=.01)
    g = torch.Generator().manual_seed(500 + seed)
    for _ in range(STEPS):
        x, y = batch(BS, TRAIN_L, g)
        loss = F.cross_entropy(m(x).reshape(-1, N), y.reshape(-1))
        if a_hom > 0:   loss = loss + a_hom * m.hom_loss()
        if c_faith > 0: loss = loss + c_faith * m.faith_loss()
        op.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); op.step()
    return m

if __name__ == '__main__':
    print('DOES THE MODEL LAND ON THE NORMAL SUBGROUP LATTICE OF S_4?\n')
    print('If it learned G/N, predictions must land in the CORRECT COSET of N almost always.')
    print('Random guessing inside a coset of size |N| gives raw accuracy 1/|N|.\n')
    LAB = {}
    for nm, H in SUBS.items():
        LAB[nm] = cosets(H)
        print(f'  {nm:>12}: |N|={len(H):>2}  {LAB[nm][1]:>2} cosets  predicted ceiling 1/|N| = {1/len(H):.3f}')
    print()
    hdr = f'{"hom":>5} {"faith":>6} {"sd":>3} {"raw acc":>8} ' + ' '.join(f'{n:>14}' for n in SUBS)
    print(hdr); print('-'*len(hdr), flush=True)
    ge = torch.Generator().manual_seed(99)
    for a_hom, c_faith, seed in [(1.,0.,0), (1.,0.,2), (1.,1.,1), (1.,0.,3), (1.,0.,1)]:
        m = train(a_hom, c_faith, seed)
        with torch.no_grad():
            x, y = batch(2048, 48, ge)
            pred = m(x)[:, -1].argmax(-1); true = y[:, -1]
            raw = (pred == true).float().mean().item()
            row = []
            for nm in SUBS:
                lab, _ = LAB[nm]
                row.append((lab[pred] == lab[true]).float().mean().item())
        print(f'{a_hom:>5.1f} {c_faith:>6.1f} {seed:>3} {raw:>8.3f} '
              + ' '.join(f'{v:>14.3f}' for v in row), flush=True)
    print('\n  Read: the learned kernel is the SMALLEST N whose coset score is ~1.000.')
    print('  If raw acc ~ 1/|N| for that same N, the lattice reading is confirmed.')
