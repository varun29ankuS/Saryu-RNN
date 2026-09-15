"""WHY linear transport can reset: the trivial summand.

Predicted linear would fail on a RESET token (a linear map's only fixed point is 0).
It did not -- best seed 0.946 at L=96, and affine was WORSE. The reasoning was wrong.

Permutations of (a,b,c,d) all lie on the affine hyperplane sum = a+b+c+d, so the
projection onto the all-ones direction is CONSTANT over every reachable state. A rank-1
projection onto that normal is therefore an exact constant map. The permutation rep is
trivial (+) standard, and the flip-flop lives in the TRIVIAL summand -- free of charge.

Test of the mechanism, two ways:
  (a) MEASURE: for a trained linear model, is there a direction v with v.h nearly constant
      across reachable states? Report min over v of std(v.h)/||h||, via the smallest
      singular value of the CENTRED state cloud. Near 0 => states lie on a hyperplane.
  (b) REMOVE IT: project the state to sum-zero after every step, killing the trivial
      summand. Prediction: linear now FAILS and affine still works.
Falsified if linear still resets with the trivial summand removed.
"""
import itertools, time, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
K, NH, BS, STEPS, TRAIN_L = 4, 3, 256, 2000, 12
EVAL_L = [12, 48, 96]
els = list(itertools.permutations(range(K)))
idx = {p: i for i, p in enumerate(els)}
Gn = len(els)
TAB = torch.tensor([[idx[tuple(b[a[i]] for i in range(K))] for b in els] for a in els])
RESET, NTOK, E = Gn, Gn + 1, idx[tuple(range(K))]
ONES = torch.ones(K) / K ** .5

class Model(nn.Module):
    def __init__(s, affine, sumzero):
        super().__init__()
        s.v = nn.Parameter(torch.randn(NTOK, NH, K) * 0.5)
        s.b = nn.Parameter(torch.full((NTOK, NH), 2.0))
        s.bias = nn.Parameter(torch.zeros(NTOK, K)) if affine else None
        s.h0 = nn.Parameter(torch.randn(K) * 0.3)
        s.out = nn.Linear(K, Gn); s.affine, s.sumzero = affine, sumzero
    def T(s, h, tok):
        V = s.v[tok]; B = s.b[tok]
        for i in range(NH):
            u = F.normalize(V[:, i], dim=-1)
            h = h - B[:, i:i+1] * (h * u).sum(-1, keepdim=True) * u
        if s.affine: h = h + s.bias[tok]
        if s.sumzero: h = h - (h * ONES).sum(-1, keepdim=True) * ONES   # kill trivial summand
        return h
    def forward(s, x, collect=False):
        h = s.h0[None].expand(x.shape[0], -1)
        if s.sumzero: h = h - (h * ONES).sum(-1, keepdim=True) * ONES
        o = []
        for t in range(x.shape[1]):
            h = s.T(h, x[:, t]); o.append(h)
        H = torch.stack(o, 1)
        return H if collect else s.out(H)

def batch(bs, L, g, p_reset=0.15):
    x = torch.randint(0, Gn, (bs, L), generator=g)
    x[torch.rand(bs, L, generator=g) < p_reset] = RESET
    acc = torch.full((bs,), E); ys = []
    for t in range(L):
        tok = x[:, t]
        acc = torch.where(tok == RESET, torch.full_like(acc, E), TAB[acc, tok.clamp(max=Gn-1)])
        ys.append(acc.clone())
    return x, torch.stack(ys, 1)

def run(affine, sumzero, seed):
    torch.manual_seed(seed); m = Model(affine, sumzero)
    op = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=.01)
    g = torch.Generator().manual_seed(500 + seed)
    for _ in range(STEPS):
        x, y = batch(BS, TRAIN_L, g)
        loss = F.cross_entropy(m(x).reshape(-1, Gn), y.reshape(-1))
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
        # (a) flatness of the state cloud: smallest singular value of centred states
        x, _ = batch(512, 48, ge)
        H = m(x, collect=True).reshape(-1, K)
        Hc = H - H.mean(0)
        sv = torch.linalg.svdvals(Hc) / (Hc.shape[0] ** .5)
        flat = (sv[-1] / sv[0]).item()
    return accs, flat

if __name__ == '__main__':
    print('DOES LINEAR RESET COME FROM THE TRIVIAL SUMMAND?\n')
    print(f'S_{K} + RESET (p=0.15). chance={1/Gn:.3f}')
    print('flatness = smallest/largest singular value of the centred state cloud.')
    print('  near 0 => states lie on a hyperplane => a rank-1 projection is a constant map.\n')
    hdr = f'{"transport":>10} {"summand":>9} {"sd":>3} ' + ' '.join(f'{("L="+str(L)):>7}' for L in EVAL_L) + f' {"flatness":>9}'
    print(hdr); print('-' * len(hdr), flush=True)
    for sumzero in [False, True]:
        for affine in [False, True]:
            for seed in [0, 2]:
                t0 = time.time(); accs, flat = run(affine, sumzero, seed)
                print(f'{"affine" if affine else "linear":>10} '
                      f'{"sum-zero" if sumzero else "full":>9} {seed:>3} '
                      + ' '.join(f'{a:>7.3f}' for a in accs)
                      + f' {flat:>9.4f}   ({time.time()-t0:.0f}s)', flush=True)
    print('\n  Prediction: with the trivial summand removed (sum-zero), linear loses the')
    print('  reset and affine does not. Falsified if linear still resets.')
