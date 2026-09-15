"""Does non-abelian transport buy anything?  v2 -- rebuilt around a PROVABLE ceiling.

THE FUNDAMENTAL FIX
-------------------
An abelian transport gives  h_out = (prod_i T_i) h_0  with the product
order-independent. So its output can only ever be a function of the MULTISET of
input tokens. That is not a capacity limit -- it is an impossibility.

Therefore the best accuracy ANY abelian-transport model can reach is exactly

    CEILING = sum_multiset P(multiset) * max_g P(product = g | multiset)

which we compute by exhaustive enumeration. This converts the experiment from
"did it train?" (which sank v1) into "did it exceed a proven bound?".

  Z_6  is abelian  -> product IS determined by the multiset -> ceiling = 1.000
  S_3  is not      -> ceiling < 1.000, computed below

MATCHED CONTROLS
  both groups have order 6, same vocab, same readout width, same everything
  diag_rot   norm-preserving diagonal rotation  -- ABELIAN (this is LRU/S4-style,
             replacing v1's tanh gain which decayed the state to 6e-3 and voided the run)
  givens1    single fixed Givens pairing        -- ABELIAN (rotations, but commuting)
  butterfly  alternating Givens pairings        -- NON-ABELIAN
  xfmr       transformer + RoPE                 -- sees positions, NOT multiset-bound
"""
import itertools, math, sys, time, collections, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(3)

D, L, STEPS, BS = 64, 8, 6000, 64
SEEDS = [0, 1, 2]

# ---------------------------------------------------------------- groups (order 6)
PERM3 = list(itertools.permutations(range(3)))          # |S_3| = 6
NG = 6
IDX = {p: i for i, p in enumerate(PERM3)}
S3 = torch.tensor([[IDX[tuple(PERM3[b][PERM3[a][i]] for i in range(3))]
                    for b in range(NG)] for a in range(NG)])
Z6 = torch.tensor([[(a + b) % NG for b in range(NG)] for a in range(NG)])
TAB = {'s3': S3, 'z6': Z6}

def product(seq, tab):
    a = seq[0]
    for t in seq[1:]: a = tab[a][t]
    return a

def ceiling(group):
    """exact best-possible accuracy for any function of the MULTISET"""
    tab = TAB[group].tolist()
    by_ms = collections.defaultdict(collections.Counter)
    for seq in itertools.product(range(NG), repeat=L):
        by_ms[tuple(sorted(seq))][product(seq, tab)] += 1
    tot = NG ** L
    return sum(c.most_common(1)[0][1] for c in by_ms.values()) / tot

def make(bs, group, g):
    x = torch.randint(0, NG, (bs, L), generator=g)
    tab = TAB[group]
    acc = x[:, 0].clone()
    for t in range(1, L): acc = tab[acc, x[:, t]]
    return x, acc

# ---------------------------------------------------------------- transports
def pairs(d, layer, alternate):
    off = 1 if (alternate and layer % 2) else 0
    return [(i, (i + 1) % d) for i in range(off, d - (0 if off else 1), 2)]

class Transport(nn.Module):
    def __init__(s, kind, nl=3):
        super().__init__(); s.kind = kind
        if kind == 'diag_rot':                      # norm-preserving abelian rotation
            s.ang = nn.Embedding(NG, D // 2); nn.init.normal_(s.ang.weight, 0, 1.0)
        else:
            s.pl = [pairs(D, l, kind == 'butterfly') for l in range(nl)]
            s.ang = nn.Embedding(NG, sum(len(p) for p in s.pl))
            nn.init.normal_(s.ang.weight, 0, 1.0)
            s.idx = [torch.tensor([i for i, _ in p]) for p in s.pl]
            s.jdx = [torch.tensor([j for _, j in p]) for p in s.pl]
    def forward(s, h, tok):
        if s.kind == 'diag_rot':
            th = s.ang(tok); c, sn = torch.cos(th), torch.sin(th)
            e, o = h[:, 0::2], h[:, 1::2]
            r = torch.empty_like(h); r[:, 0::2] = e * c - o * sn; r[:, 1::2] = e * sn + o * c
            return r
        a = s.ang(tok); off = 0
        for pl, idx, jdx in zip(s.pl, s.idx, s.jdx):
            th = a[:, off:off + len(pl)]; off += len(pl)
            c, sn = torch.cos(th), torch.sin(th)
            hi, hj = h[:, idx], h[:, jdx]
            h = h.clone(); h[:, idx] = hi * c - hj * sn; h[:, jdx] = hi * sn + hj * c
        return h

class Scan(nn.Module):
    def __init__(s, kind):
        super().__init__(); s.t = Transport(kind)
        s.h0 = nn.Parameter(torch.randn(D) / D ** .5)
        s.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(),
                              nn.Linear(4 * D, 4 * D), nn.GELU(), nn.Linear(4 * D, NG))
    def forward(s, x):
        h = s.h0[None].expand(x.shape[0], -1)
        for t in range(x.shape[1]): h = s.t(h, x[:, t])
        return s.out(h)

class Xfmr(nn.Module):
    def __init__(s, H=4, NL=3):
        super().__init__(); s.H, s.dh = H, D // H
        s.e = nn.Embedding(NG, D)
        s.bl = nn.ModuleList([nn.ModuleDict(dict(
            qkv=nn.Linear(D, 3 * D, bias=False), p=nn.Linear(D, D, bias=False),
            n1=nn.LayerNorm(D), n2=nn.LayerNorm(D),
            m=nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)))) for _ in range(NL)])
        s.nf = nn.LayerNorm(D); s.head = nn.Linear(D, NG)
        s.register_buffer('f', 10000. ** (-torch.arange(0, s.dh, 2).float() / s.dh))
    def forward(s, x):
        B, T = x.shape
        a = torch.arange(T, dtype=torch.float)[:, None] * s.f[None]
        c, sn = torch.cos(a), torch.sin(a)
        def rot(t):
            e, o = t[..., 0::2], t[..., 1::2]
            r = torch.empty_like(t); r[..., 0::2] = e * c - o * sn; r[..., 1::2] = e * sn + o * c
            return r
        z = s.e(x)
        for b in s.bl:
            q, k, v = [t.view(B, T, s.H, s.dh).transpose(1, 2) for t in b['qkv'](b['n1'](z)).chunk(3, -1)]
            y = F.scaled_dot_product_attention(rot(q), rot(k), v)
            z = z + b['p'](y.transpose(1, 2).reshape(B, T, D))
            z = z + b['m'](b['n2'](z))
        return s.head(s.nf(z))[:, -1]

def run(kind, group, seed):
    torch.manual_seed(seed)
    m = Xfmr() if kind == 'xfmr' else Scan(kind)
    op = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(op, STEPS)
    g = torch.Generator().manual_seed(500 + seed)
    for _ in range(STEPS):
        x, y = make(BS, group, g)
        loss = F.cross_entropy(m(x), y)
        op.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); op.step(); sch.step()
    m.eval(); ge = torch.Generator().manual_seed(99); acc = 0.
    with torch.no_grad():
        for _ in range(20):
            x, y = make(256, group, ge)
            acc += (m(x).argmax(-1) == y).float().mean().item()
    return acc / 20

if __name__ == '__main__':
    cz, cs = ceiling('z6'), ceiling('s3')
    print(f'MATCHED-GROUP TEST v2 | |Z_6| = |S_3| = {NG} | seq len {L} | {len(SEEDS)} seeds')
    print(f'chance = {1/NG:.3f}   ({STEPS} steps, d={D})\n')
    print(f'PROVEN CEILING for any abelian transport (best function of the multiset):')
    print(f'   Z_6 (abelian)      : {cz:.4f}   <- multiset determines the product')
    print(f'   S_3 (non-abelian)  : {cs:.4f}   <- ORDER matters; abelian cannot exceed this\n', flush=True)
    print(f'{"transport":>10} {"abelian?":>9} {"Z_6":>16} {"S_3":>16}   verdict')
    print('-' * 72, flush=True)
    lab = {'diag_rot': 'yes', 'givens1': 'yes', 'butterfly': 'NO', 'xfmr': 'n/a'}
    for kind in ['diag_rot', 'givens1', 'butterfly', 'xfmr']:
        r = {}
        for grp in ['z6', 's3']:
            a = [run(kind, grp, s) for s in SEEDS]
            mu = sum(a) / len(a)
            sd = (sum((v - mu) ** 2 for v in a) / max(1, len(a) - 1)) ** .5
            r[grp] = (mu, sd)
        v = 'EXCEEDS abelian ceiling' if r['s3'][0] > cs + 0.02 else 'at/below ceiling'
        print(f'{kind:>10} {lab[kind]:>9} {r["z6"][0]:>9.3f}+-{r["z6"][1]:<5.3f} '
              f'{r["s3"][0]:>9.3f}+-{r["s3"][1]:<5.3f}   {v}', flush=True)
    print(f'\nAnything above {cs:.3f} on S_3 is doing something abelian transport provably cannot.', flush=True)
