"""Is the ACTIVATION ON BETA what gates the non-abelian regime?

Our transports carry no activation -- the algebra lives in the recurrence. But real DeltaNet
sets beta = 2*sigmoid(.), and that RANGE decides the eigenvalue structure of I - beta u u^T:

    beta < 1   contraction          eigenvalue in (0,1)
    beta = 1   projection           eigenvalue 0, rank-deficient
    beta = 2   reflection           eigenvalue -1

Reflections are what generate non-abelian structure, and fla's `allow_neg_eigval` flag is
literally the factor of 2 on the sigmoid. So the activation's range may gate which groups
are representable at all -- an architectural claim, not a tuning detail.

    sig2    beta = 2*sigmoid(x)   in (0,2)   reflections reachable   [fla default]
    sig1    beta = 1*sigmoid(x)   in (0,1)   contractions only
    fixed2  beta = 2 constant                pure reflections, no learning
    free    beta unconstrained               no activation at all

Prediction: sig1 cannot reach the faithful rung (no reflections), and should pile up on
lower rungs. If the reachable rungs track the beta range, the activation determines the
lattice position -- which would make it the load-bearing choice, above objective design.

REFERENCES
  [Grazzi et al. 2024] Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues, arXiv 2411.12537
  [Yang et al. 2024a] Parallelizing Linear Transformers with the Delta Rule over Sequence Length, arXiv 2406.06484
"""
import itertools, time, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
K, BS, STEPS, TRAIN_L = 4, 256, 2000, 12
els = list(itertools.permutations(range(K))); ix = {p: i for i, p in enumerate(els)}
TAB = torch.tensor([[ix[tuple(b[a[i]] for i in range(K))] for b in els] for a in els])
N = len(els)

def cycle_type(p):
    seen, ct = set(), []
    for i in range(K):
        if i in seen: continue
        c, j = 0, i
        while j not in seen:
            seen.add(j); j = p[j]; c += 1
        ct.append(c)
    return tuple(sorted(ct, reverse=True))
CT = [cycle_type(p) for p in els]
CLASSES = sorted(set(CT)); CLS_OF = torch.tensor([CLASSES.index(c) for c in CT])
ORDER = [(1,1,1,1), (2,1,1), (2,2), (3,1), (4,)]
IDX = [CLASSES.index(c) for c in ORDER]
RAW = {'trivial':[1,1,1,1,1], 'sign':[1,-1,1,1,-1], '2dim':[2,0,2,-1,0],
       'std':[3,1,-1,0,-1], 'std_sgn':[3,-1,-1,0,1]}
CHI = {}
for k, v in RAW.items():
    row = torch.zeros(len(CLASSES))
    for j, c in enumerate(IDX): row[c] = v[j]
    CHI[k] = torch.tensor([row[CLS_OF[g]] for g in range(N)])

class Model(nn.Module):
    def __init__(s, mode, d=4, nh=3):
        super().__init__()
        s.v = nn.Parameter(torch.randn(N, nh, d) * 0.5)
        s.raw = nn.Parameter(torch.zeros(N, nh))     # sigmoid(0)=0.5 -> beta=1 at init
        s.h0 = nn.Parameter(torch.randn(d) * 0.3)
        s.out = nn.Linear(d, N); s.nh, s.d, s.mode = nh, d, mode
    def beta(s, tok):
        r = s.raw[tok]
        if s.mode == 'sig2':   return 2.0 * torch.sigmoid(r)
        if s.mode == 'sig1':   return 1.0 * torch.sigmoid(r)
        if s.mode == 'fixed2': return torch.full_like(r, 2.0)
        return r + 1.0                                # free: unconstrained, init at 1
    def T(s, h, tok):
        V = s.v[tok]; B = s.beta(tok)
        for i in range(s.nh):
            u = F.normalize(V[:, i], dim=-1)
            h = h - B[:, i:i+1] * (h * u).sum(-1, keepdim=True) * u
        return h
    def forward(s, x):
        h = s.h0[None].expand(x.shape[0], -1); o = []
        for t in range(x.shape[1]):
            h = s.T(h, x[:, t]); o.append(h)
        return s.out(torch.stack(o, 1))
    def hom_loss(s, n=64):
        h = torch.randn(n, s.d)
        a = torch.randint(0, N, (n,)); b = torch.randint(0, N, (n,))
        return (s.T(s.T(h, a), b) - s.T(h, TAB[a, b])).norm(dim=1).mean() / s.d ** .5
    def mults(s):
        with torch.no_grad():
            I = torch.eye(s.d)
            cols = torch.stack([s.T(I[j][None].expand(N, -1), torch.arange(N))
                                for j in range(s.d)], -1)
            chi = cols.diagonal(dim1=-2, dim2=-1).sum(-1)
            return {k: float((chi * CHI[k]).sum() / N) for k in CHI}

def batch(bs, L, g):
    x = torch.randint(0, N, (bs, L), generator=g)
    acc = x[:, 0].clone(); ys = [acc.clone()]
    for t in range(1, L):
        acc = TAB[acc, x[:, t]]; ys.append(acc.clone())
    return x, torch.stack(ys, 1)

def run(mode, seed):
    torch.manual_seed(seed); m = Model(mode)
    op = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=.01)
    g = torch.Generator().manual_seed(500 + seed)
    for _ in range(STEPS):
        x, y = batch(BS, TRAIN_L, g)
        loss = F.cross_entropy(m(x).reshape(-1, N), y.reshape(-1)) + 1.0 * m.hom_loss()
        op.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); op.step()
    ge = torch.Generator().manual_seed(99); accs = []
    with torch.no_grad():
        for L in (12, 96):
            a = 0.
            for _ in range(4):
                x, y = batch(256, L, ge)
                a += (m(x)[:, -1].argmax(-1) == y[:, -1]).float().mean().item()
            accs.append(a / 4)
        bmean = float(m.beta(torch.arange(N)).mean())
    return accs, m.mults(), bmean

if __name__ == '__main__':
    print('DOES THE BETA ACTIVATION GATE THE NON-ABELIAN REGIME?', flush=True)
    print(flush=True)
    print('  I - beta u u^T :  beta<1 contraction | beta=1 projection | beta=2 reflection',
          flush=True)
    print('  reflections generate non-abelian structure; fla\'s allow_neg_eigval IS the 2x.',
          flush=True)
    print('  rungs 1.000 / 0.250 / 0.083 / 0.042', flush=True)
    print(flush=True)
    print(f'{"beta":>7} {"sd":>3} {"L=12":>7} {"L=96":>7} {"rung":>7} {"mean b":>7} '
          f'{"max3dim":>8}', flush=True)
    print('-' * 56, flush=True)
    RUNG = [1.0, .25, 1/12, 1/24]
    for mode in ['sig2', 'sig1', 'fixed2', 'free']:
        solved = 0
        for seed in range(4):
            t0 = time.time(); accs, mu, bm = run(mode, seed)
            m3 = max(mu['std'], mu['std_sgn']); solved += accs[-1] > 0.9
            near = min(RUNG, key=lambda v: abs(v - accs[-1]))
            print(f'{mode:>7} {seed:>3} {accs[0]:>7.3f} {accs[1]:>7.3f} {near:>7.3f} '
                  f'{bm:>7.3f} {m3:>8.3f}   ({time.time()-t0:.0f}s)', flush=True)
        print(f'{mode:>7}  == {solved}/4 solved at L=96', flush=True)
        print(flush=True)
    print('  If sig1 (no reflections) cannot reach 1.000 while sig2 can, the ACTIVATION', flush=True)
    print('  RANGE -- not the objective -- is what gates the reachable lattice position.', flush=True)
