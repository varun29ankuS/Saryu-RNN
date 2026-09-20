"""Does sign binding survive training? The cheapest binding operator, tested against the only
intervention that has ever moved this wall.

THE OPERATOR. Bind the write by a +-1 vector and unbind at the read with the SAME projection.
+-1 matters because it is an INVOLUTION: s * (s * c) = c exactly, so a query reproducing the key's
sign vector inverts the binding with no residue. The existing silu output gate is continuous and
cannot do that at any setting.

WHY THIS ONE. Signed diagonal matrices together with permutations form the hyperoctahedral group
B_d = (Z/2)^d x| S_d -- the subgroup of O(d) whose elements apply in O(d) rather than O(d^2). So it
is not one option among many; it is the boundary of cheap norm-preserving binding. Measured against
circular convolution at matched width, same capacity for a tenth of the time:

    d=512   n=8    n=16   n=32   n=64    us/bind
    sign   1.000  1.000  0.958  0.655     10.2
    conv   1.000  0.998  0.965  0.645    122.6

THE PRIOR, AND IT IS BAD. Two constructions in this project predicted ~0.97 and trained to ~0.31
(the nh ladder, the key-bound write). Constructions here have no demonstrated predictive value.
Twenty architectural interventions have not moved 4-pair recall off ~0.32. The ONLY thing that has
is a pair-count curriculum: 8/16 seeds above 0.5 against 0/8 (Fisher p ~ 0.01), while still failing
the registered 0.8 bar.

So this is a 2x2, not a demonstration. sign binding is tested WITH and WITHOUT the curriculum,
because this project's own evidence says effects here are combinatorial (key-bound write alone
0.282, orthogonal axes alone 0.250, both together 0.980).

REGISTERED PREDICTIONS, before running:
  P1  curriculum main effect reappears in this config (it vanished at H=1/nh=17, so it may be
      specific to H=8/dh=16 -- that is worth knowing either way).
  P2  sign binding alone beats its control.
  P3  sign + curriculum is the best cell.

FALSIFIER: sign binding is no better than its control in either curriculum condition. Then the
cheapest binding operator does not survive gradient descent, straight-through sign is not
trainable here, and the construction-to-training gap claims a third victim.

Both thresholds declared in advance: 0.8 (registered) and 0.5 (where the curriculum effect lives).

    python evidence/sign_bind.py
Environment: NPAIRS, GAP, STEPS, LR, OPT, D, H, NL, NH, SEEDS, THREADS.
"""
from __future__ import annotations
import itertools, os, sys, time
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                     # noqa: E402
from saryu.metrics import Run                                        # noqa: E402
from binding_long import Muon                                        # noqa: E402

TARGET = int(os.environ.get('NPAIRS', 4)); GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 5000)); BS = 32
LR = float(os.environ.get('LR', 3e-4)); OPT = os.environ.get('OPT', 'muon')
D = int(os.environ.get('D', 128)); H = int(os.environ.get('H', 8))
NL = int(os.environ.get('NL', 2)); NH = int(os.environ.get('NH', 2))
SEEDS = int(os.environ.get('SEEDS', 3)); NENT = 64; SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 4)))

def make(rng, n):
    e = rng.choice(NENT, size=2 * n, replace=False); ks, vs = e[:n], e[n:]
    rest = np.setdiff1d(np.arange(NENT), e); s = []
    for k, v in zip(ks, vs): s += [int(k), int(v)]
    s += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, n)); s += [SEP, int(ks[i])]
    return s, int(vs[i])

def batch(rng, bs, n):
    o = [make(rng, n) for _ in range(bs)]
    return torch.tensor([a for a, _ in o]), torch.tensor([b for _, b in o])

def train_one(sign, cur, seed):
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT + 2, D, NL, nh=NH, H=H, bind_sign=sign)
    p = list(m.parameters())
    opt = (torch.optim.AdamW(p, lr=LR, weight_decay=0.01) if OPT == 'adamw' else Muon(p, lr=LR))
    rng = np.random.default_rng(1000 + seed); ev = np.random.default_rng(7)
    log = Run(f'sb-{"sign" if sign else "plain"}-{cur}-s{seed}',
              config=dict(arm=f'sign_bind/{"sign" if sign else "plain"}-{cur}', pairs=TARGET,
                          bind_sign=sign, curriculum=cur, d=D, H=H, nh=NH, nl=NL, lr=LR,
                          opt=OPT, steps=STEPS, seed=seed, chance_top1=round(1 / TARGET, 4)))
    best = 0.0
    for s in range(STEPS):
        n = TARGET if cur == 'uniform' else min(TARGET, 2 + int(3.0 * s / STEPS))
        x, y = batch(rng, BS, n)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(p, 1.0); opt.step()
        if s == 0 or (s + 1) % 250 == 0:
            m.eval()
            with torch.no_grad():
                a = 0.
                for _ in range(4):
                    x2, y2 = batch(ev, BS, TARGET)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train(); best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best, 'cur/pairs': n})
    log.done(); return best

def main():
    print(f'SIGN BINDING  {TARGET} pairs, d={D} H={H} dh={D//H} nh={NH}, {STEPS} steps, '
          f'{SEEDS} seeds, chance 0.250')
    print('+-1 binding is an exact involution; the silu gate is not. 2x2 against the curriculum.\n')
    print(f'{"bind":>6} {"cur":>8} {">0.8":>6} {">0.5":>6}  best per seed')
    print('-' * 52)
    t0 = time.time(); rows = []
    for sign, cur in itertools.product((False, True), ('uniform', 'slow')):
        b = [train_one(sign, cur, s) for s in range(SEEDS)]
        rows.append((sign, cur, b))
        print(f'{("sign" if sign else "plain"):>6} {cur:>8} {sum(x>0.8 for x in b):>3}/{SEEDS} '
              f'{sum(x>0.5 for x in b):>3}/{SEEDS}  ' + ', '.join(f'{x:.3f}' for x in b), flush=True)
    print(f'\n{time.time()-t0:.0f}s\nMAIN EFFECTS (mean best)')
    for lab, f in (('plain', lambda r: not r[0]), ('sign', lambda r: r[0]),
                   ('uniform', lambda r: r[1] == 'uniform'), ('slow', lambda r: r[1] == 'slow')):
        v = [x for r in rows if f(r) for x in r[2]]
        print(f'  {lab:>8}: {np.mean(v):.3f}')
    print('\nREAD')
    print('  sign beats plain in either column -> the cheapest binding operator TRAINS')
    print('  no difference                     -> straight-through sign is not trainable here and')
    print('                                       the construction-to-training gap claims a third')


if __name__ == "__main__":
    main()
