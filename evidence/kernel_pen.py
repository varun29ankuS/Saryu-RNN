"""TABLE-FREE faithfulness by counting the KERNEL, not by separating pairs.

Seven of eight objectives tonight were gamed or harmful. The two that worked were the braid
relations (many equations, hard to satisfy vacuously) and the multiplicity penalty (needs
the character table). The failures share a shape:

    margin hinge      |G|^2 all-pairs separation -> over-constrains, FIGHTS the algebra
    Hopfield          same, on states           -> grp-law up 35x
    <chi,chi>         ONE scalar                -> gamed by leaving the representation
                                                   manifold (seed 1: chi-norm 1.976, acc 0.045)

The fix is targeting. For a HOMOMORPHISM, faithfulness is a statement about the kernel alone:
    T_g = T_h  <=>  T_{g h^-1} = I
so |G|-1 conditions against the IDENTITY are equivalent to |G|^2 pairwise ones -- but far
more targeted, and each is pinned to a fixed reference rather than to the other transports.

Table-free because we need not know WHICH token is the identity: in a faithful rep exactly
ONE transport may sit near I. Count them softly and penalise the excess.

    s_g     = exp(-||T_g h - h||^2 / (sigma^2 d))     ~1 when T_g ~ I
    kernel  = relu(sum_g s_g - 1)                      excess beyond the identity itself

Three variants, since the soft-count shape is the whole question:
    excess   relu(sum s_g - 1)          penalise only the excess
    soft     sum s_g                    penalise all near-identity mass
    second   second-largest s_g         penalise the runner-up directly (most targeted)
"""
import itertools, time, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
K, BS, STEPS, TRAIN_L = 4, 256, 2500, 12
SIGMA = 0.5
els = list(itertools.permutations(range(K))); ix = {p: i for i, p in enumerate(els)}
TAB = torch.tensor([[ix[tuple(b[a[i]] for i in range(K))] for b in els] for a in els])
N = len(els); E = ix[tuple(range(K))]

class Model(nn.Module):
    def __init__(s, d=4, nh=3):
        super().__init__()
        s.v = nn.Parameter(torch.randn(N, nh, d) * 0.5)
        s.b = nn.Parameter(torch.full((N, nh), 2.0))
        s.h0 = nn.Parameter(torch.randn(d) * 0.3)
        s.out = nn.Linear(d, N); s.nh, s.d = nh, d
    def T(s, h, tok):
        V = s.v[tok]; B = s.b[tok]
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
    def near_identity(s, n=128):
        """s_g in [0,1]: how close T_g is to the identity map. No table needed."""
        h = torch.randn(n, s.d)
        dev = torch.stack([(s.T(h, torch.full((n,), g)) - h).pow(2).mean()
                           for g in range(N)])                    # (N,)
        return torch.exp(-dev / (SIGMA ** 2))
    def kernel_loss(s, kind):
        sg = s.near_identity()
        if kind == 'excess': return F.relu(sg.sum() - 1.0)
        if kind == 'soft':   return sg.sum()
        if kind == 'second': return sg.topk(2).values[1]          # runner-up only
        return torch.zeros(())
    def kernel_size(s):
        with torch.no_grad():
            return int((s.near_identity() > 0.5).sum())

def batch(bs, L, g):
    x = torch.randint(0, N, (bs, L), generator=g)
    acc = x[:, 0].clone(); ys = [acc.clone()]
    for t in range(1, L):
        acc = TAB[acc, x[:, t]]; ys.append(acc.clone())
    return x, torch.stack(ys, 1)

def run(kind, seed):
    torch.manual_seed(seed); m = Model()
    op = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=.01)
    g = torch.Generator().manual_seed(500 + seed)
    for _ in range(STEPS):
        x, y = batch(BS, TRAIN_L, g)
        loss = F.cross_entropy(m(x).reshape(-1, N), y.reshape(-1)) + 1.0 * m.hom_loss()
        if kind != 'none': loss = loss + 1.0 * m.kernel_loss(kind)
        op.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); op.step()
    ge = torch.Generator().manual_seed(99); a = 0.
    with torch.no_grad():
        for _ in range(4):
            x, y = batch(256, 96, ge)
            a += (m(x)[:, -1].argmax(-1) == y[:, -1]).float().mean().item()
    return a / 4, m.hom_loss(300).item(), m.kernel_size()

if __name__ == '__main__':
    print('TABLE-FREE FAITHFULNESS: count the kernel instead of separating pairs')
    print()
    print(f'S_{K}. For a homomorphism, faithful <=> only the identity maps to I.')
    print('|G|-1 constraints against a FIXED reference, not |G|^2 mutual ones.')
    print('Reference: hom only 3/6; character-table multiplicity 5/6 (needs the table).')
    print()
    hdr = f'{"arm":>8} {"sd":>3} {"L=96":>7} {"grp-law":>8} {"|ker|":>6}'
    print(hdr); print('-' * len(hdr), flush=True)
    for kind in ['none', 'excess', 'soft', 'second']:
        solved = 0
        for seed in range(6):
            t0 = time.time(); acc, gl, ks = run(kind, seed)
            solved += acc > 0.9
            print(f'{kind:>8} {seed:>3} {acc:>7.3f} {gl:>8.4f} {ks:>6}   ({time.time()-t0:.0f}s)',
                  flush=True)
        print(f'{"":>8}     -> {solved}/6 solved at L=96', flush=True)
        print(flush=True)
    print('  |ker| = transports within the identity basin. 1 = faithful.')
