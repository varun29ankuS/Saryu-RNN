"""Binding is loss-improving and the gradient points at it. So does it ever arrive?

WHERE THIS COMES FROM. binding_gradient.txt measured, at the converged point, the cosine between the
descent direction and the direction that builds key-dependence:

    at init         B = -0.0006   cos(-gL, gB) = +0.80
    after training  B = +0.0095   cos(-gL, gB) = +0.75

So the landscape is NOT flat in the binding direction and this is NOT a symmetry-breaking
bifurcation -- the gradient is strongly aligned with binding the whole way. Yet 3000 steps of Adam
move B from 0 to 0.0095 while the cross-entropy falls 4.16 -> 2.95, i.e. essentially all of the
loss improvement is bought by the value set and the positional prior, and almost none by binding.

That leaves stiffness: the direction is right, the progress along it is minuscule. Two consequences
are testable, and this script tests both on the EASIEST case (2 pairs), because if binding does not
arrive there it will not arrive anywhere.

  1. does B grow if simply given far more steps?     -> arm `adamw`, 20k steps, B logged throughout
  2. does a preconditioner recover it?                -> arm `muon`, orthogonalised momentum on
                                                         every 2-D parameter, which rescales exactly
                                                         the stiff directions a Euclidean step
                                                         under-weights

REGISTERED PREDICTIONS, before running. B = 0 means a rank-one, key-independent readout; B = 1 is
perfect binding. Accuracy at 2 pairs is 1/2 whenever B = 0.

  P1  adamw: B rises monotonically and passes 0.05 by 20k steps (slow but real progress)
  P2  muon reaches a higher B than adamw at equal steps
  P3  if either arm passes B = 0.3, accuracy leaves 0.5 -- binding and accuracy move together

FALSIFIER: B flat at ~0.01 for both arms through 20k steps. Then the direction being aligned does
not mean it is reachable, and "conditioning" is the wrong diagnosis too -- what would remain is that
the architecture's readout cannot express key-dependent retrieval at all, which is a claim about the
model and not about the optimiser.

    python evidence/binding_long.py
Environment: PAIRS, GAP, STEPS, LR, ARMS, SEED, EVERY, THREADS.
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
from saryu.metrics import Run                                         # noqa: E402

PAIRS = int(os.environ.get('PAIRS', 2))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 20000))
EVERY = int(os.environ.get('EVERY', 1000))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 1e-3))
SEED = int(os.environ.get('SEED', 0))
D, NL, NENT = 128, 2, 64
SEP = NENT
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
    seq += [SEP, int(ks[i])]
    return seq, [int(v) for v in vs], i


def batch(rng, bs):
    xs, vs, ii = zip(*(make(rng) for _ in range(bs)))
    return torch.tensor(xs), torch.tensor(vs), torch.tensor(ii)


def scores(m, x, vs, ii):
    logits = m(x)[:, -1]
    tgt = vs[torch.arange(len(ii)), ii]
    ce = F.cross_entropy(logits, tgt)
    p = F.softmax(logits, -1).gather(1, vs)
    p = p / p.sum(1, keepdim=True).clamp_min(1e-9)
    corr = p.gather(1, ii[:, None]).squeeze(1)
    off = (p.sum(1) - corr) / max(PAIRS - 1, 1)
    acc = (logits.argmax(-1) == tgt).float().mean()
    return ce, (corr - off).mean(), acc


@torch.no_grad()
def ns_orth(G, steps=5):
    """Newton-Schulz orthogonalisation, the core of Muon: replace G by the orthogonal factor of its
    polar decomposition, so every singular direction takes an equally sized step."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if transposed else X).to(G.dtype)


class Muon:
    """Orthogonalised momentum on 2-D parameters; plain AdamW-style on the rest."""

    def __init__(self, params, lr, mom=0.95, wd=0.01):
        self.p2 = [p for p in params if p.ndim == 2]
        self.p1 = [p for p in params if p.ndim != 2]
        self.lr, self.mom, self.wd = lr, mom, wd
        self.buf = {id(p): torch.zeros_like(p) for p in self.p2}
        self.adam = torch.optim.AdamW(self.p1, lr=lr, weight_decay=wd)

    def zero_grad(self):
        for p in self.p2 + self.p1:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for p in self.p2:
            if p.grad is None:
                continue
            b = self.buf[id(p)]
            b.mul_(self.mom).add_(p.grad)
            g = ns_orth(b)
            scale = (max(p.shape[0], p.shape[1]) ** 0.5)
            p.mul_(1 - self.lr * self.wd).add_(g, alpha=-self.lr * scale)
        self.adam.step()


def run(arm):
    torch.manual_seed(SEED)
    m = SaryuV3LM(NENT + 2, D, NL)
    params = list(m.parameters())
    opt = (torch.optim.AdamW(params, lr=LR, weight_decay=0.01) if arm == 'adamw'
           else Muon(params, lr=LR))
    rng = np.random.default_rng(1000 + SEED)
    log = Run(f'bind-{arm}-p{PAIRS}', config=dict(arm=arm, pairs=PAIRS, gap=GAP, lr=LR,
                                                  steps=STEPS, chance_top1=round(1 / NENT, 4),
                                                  chance_loss=round(float(np.log(NENT)), 3)))
    print(f'\n{arm}: {STEPS} steps, lr {LR:g}, {PAIRS} pairs   (B=0 means rank one, '
          f'accuracy {1/PAIRS:.3f})')
    print(f'{"step":>7} {"ce":>7} {"B":>9} {"acc":>7}')
    t0 = time.time()
    ev = np.random.default_rng(99)
    for s in range(STEPS):
        x, vs, ii = batch(rng, BS)
        ce, B, _ = scores(m, x, vs, ii)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if s == 0 or (s + 1) % EVERY == 0:
            m.eval()
            with torch.no_grad():
                cs, bs_, as_ = 0., 0., 0.
                for _ in range(8):
                    x2, v2, i2 = batch(ev, BS)
                    c2, b2, a2 = scores(m, x2, v2, i2)
                    cs += float(c2) / 8; bs_ += float(b2) / 8; as_ += float(a2) / 8
            m.train()
            log.log(s + 1, loss=cs, **{'eval/binding_B': bs_, 'eval/top1': as_})
            print(f'{s+1:>7} {cs:>7.3f} {bs_:>+9.4f} {as_:>7.3f}', flush=True)
    log.done()
    print(f'  {time.time()-t0:.0f}s')
    return bs_, as_


def main():
    print(f'DOES BINDING EVER ARRIVE?  {PAIRS} pairs, gap {GAP}, d={D}, {NL} layers')
    res = {}
    for arm in os.environ.get('ARMS', 'adamw,muon').split(','):
        res[arm] = run(arm)
    print('\nFINAL   ' + '   '.join(f'{k}: B={v[0]:+.4f} acc={v[1]:.3f}' for k, v in res.items()))
    print('\nREGISTERED')
    for k, (B, a) in res.items():
        print(f'  {k:<6} P1/P2 B > 0.05: {"HOLDS" if B > 0.05 else "FAILS"}   '
              f'P3 acc left {1/PAIRS:.3f}: {"yes" if abs(a - 1/PAIRS) > 0.05 else "no"}')
    if all(v[0] < 0.05 for v in res.values()):
        print('\n  FALSIFIER TRIGGERED: B stays ~0 for every arm. The binding direction being')
        print('  aligned with descent does not make it reachable, and conditioning is not the')
        print('  explanation either. What remains is that this readout cannot express')
        print('  key-dependent retrieval -- a claim about the architecture, not the optimiser.')


if __name__ == '__main__':
    main()
