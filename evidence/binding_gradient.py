"""At the converged point, does the loss gradient carry ANY information about binding?

THE QUESTION. The converged model is rank one: rows of the key->value assignment are identical to
four decimals, so the readout ignores the query key entirely (assignment_matrix.txt). Architecture
changes do not move it and seeds do not separate them (multiseed.txt). Two very different
explanations remain, and they are distinguishable by one measurement:

  FLAT / SYMMETRIC   the loss gradient is orthogonal to every direction that would create
                     key-dependence. Then SGD gets no signal at all, escaping is a bifurcation
                     rather than a descent step, and this is a landscape fact no optimiser setting
                     fixes.
  MISALIGNED / STIFF the binding direction IS loss-improving, but the Euclidean gradient barely
                     points along it. Then it is a conditioning problem and a preconditioner
                     (natural gradient, Muon-style) should help.

HOW. Define a binding objective measured on the same batches:

    B = mean_i [ P(value of pair i | key of pair i) - mean_{j != i} P(value of pair j | key i) ]

B is exactly "diagonal mass minus off-diagonal mass" of the assignment matrix, so raising B IS
creating key-dependence, and B = 0 for any rank-one readout no matter what prior it uses. Then at
the trained parameters compute

    g_L = grad of the training cross-entropy
    g_B = grad of B

and report their norms and the cosine between -g_L (the descent direction) and g_B (the direction
that builds binding). A cosine near zero is the first explanation; a clearly positive cosine with a
small |g_B| relative to |g_L| is the second.

CONTROL. The same quantities are reported at INITIALISATION. If the cosine is near zero both before
and after training, the binding direction was never visible to the loss at any point, which is a
stronger statement than it being invisible only at the end.

    python evidence/binding_gradient.py
Environment: PAIRS, GAP, STEPS, LR, SEED, THREADS, PROBE_BATCHES.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from saryu.model import SaryuV3LM                                     # noqa: E402

PAIRS = int(os.environ.get('PAIRS', 4))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 3000))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
SEED = int(os.environ.get('SEED', 0))
PROBE = int(os.environ.get('PROBE_BATCHES', 24))
D, NL, NENT = 128, 2, 64
SEP, PAD = NENT, NENT + 1
SEQ = 2 * PAIRS + GAP + 2
torch.set_num_threads(int(os.environ.get('THREADS', 8)))


def make(rng):
    ent = rng.choice(NENT, size=2 * PAIRS, replace=False)
    ks, vs = ent[:PAIRS], ent[PAIRS:]
    rest = np.setdiff1d(np.arange(NENT), ent)
    seq = []
    for k, v in zip(ks, vs):
        seq += [int(k), int(v)]
    seq += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, PAIRS))
    seq += [sep_(), int(ks[i])]
    return seq, [int(v) for v in vs], i


def sep_():
    return SEP


def batch(rng, bs):
    xs, vs, ii = zip(*(make(rng, ) for _ in range(bs)))
    return torch.tensor(xs), torch.tensor(vs), torch.tensor(ii)


def flat_grad(loss, params, retain=False):
    gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(gs, params)])


def objectives(m, x, vs, ii):
    """Returns (cross-entropy, binding score B). Both differentiable."""
    logits = m(x)[:, -1]
    p = F.softmax(logits, -1)
    tgt = vs[torch.arange(len(ii)), ii]
    ce = F.cross_entropy(logits, tgt)
    mass = p.gather(1, vs)                                   # [B, PAIRS] mass on each pair's value
    mass = mass / mass.sum(1, keepdim=True).clamp_min(1e-9)  # restricted to the values present
    corr = mass.gather(1, ii[:, None]).squeeze(1)
    off = (mass.sum(1) - corr) / max(PAIRS - 1, 1)
    return ce, (corr - off).mean()


def probe(m, tag):
    params = [p for p in m.parameters() if p.requires_grad]
    rng = np.random.default_rng(4242)
    gL = torch.zeros(sum(p.numel() for p in params))
    gB = torch.zeros_like(gL)
    bval = 0.0
    for _ in range(PROBE):
        x, vs, ii = batch(rng, BS)
        ce, B = objectives(m, x, vs, ii)
        gL += flat_grad(ce, params, retain=True) / PROBE   # B still needs the graph
        gB += flat_grad(B, params) / PROBE
        bval += float(B) / PROBE
    nL, nB = gL.norm().item(), gB.norm().item()
    cos = float(torch.dot(-gL, gB) / (nL * nB + 1e-12))
    # how much of the descent direction lies along the binding direction
    frac = float(torch.dot(-gL, gB / (nB + 1e-12)) / (nL + 1e-12))
    print(f'  {tag:<14} B={bval:+.4f}  |gL|={nL:.4e}  |gB|={nB:.4e}  '
          f'cos(-gL,gB)={cos:+.4f}')
    return cos, nL, nB, bval


def main():
    print(f'BINDING GRADIENT  {PAIRS} pairs, gap {GAP}, {STEPS} steps, lr {LR:g}, seed {SEED}')
    print('B = diagonal minus off-diagonal mass of the assignment matrix; B = 0 for ANY rank-one '
          'readout.\n')
    torch.manual_seed(SEED)
    m = SaryuV3LM(NENT + 2, D, NL)
    print('gradient alignment:')
    c0 = probe(m, 'at init')

    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
    rng = np.random.default_rng(1000 + SEED)
    t0 = time.time()
    for s in range(STEPS):
        x, vs, ii = batch(rng, BS)
        ce, _ = objectives(m, x, vs, ii)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if (s + 1) % max(1, STEPS // 3) == 0:
            print(f'    step {s+1}/{STEPS} ce {float(ce):.3f} {time.time()-t0:.0f}s', flush=True)
    c1 = probe(m, 'after training')

    print('\nREADING IT')
    print(f'  cos at init      {c0[0]:+.4f}')
    print(f'  cos after        {c1[0]:+.4f}')
    print(f'  B after          {c1[3]:+.4f}   (0 = rank one, no key-dependence)')
    if abs(c1[0]) < 0.02:
        print('\n  The descent direction is essentially ORTHOGONAL to the binding direction.')
        print('  The loss gradient carries almost no information about key-dependence, so SGD is')
        print('  not failing to follow a signal -- there is no signal to follow. Escaping is a')
        print('  bifurcation, and no learning rate or optimiser setting addresses that.')
    elif c1[0] > 0.02:
        print('\n  Binding IS loss-improving and the descent direction points partly along it.')
        print('  Then this is conditioning, not geometry: the step along that direction is simply')
        print('  small, and a preconditioner should recover it.')
    else:
        print('\n  The descent direction points AWAY from binding: at this point the loss is')
        print('  actively served by staying key-independent.')


if __name__ == '__main__':
    main()
