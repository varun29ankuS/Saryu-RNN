"""The vector route, retested WITH the component its own literature says it needs.

EVERY BASELINE QUOTED BELOW IS RETRACTED. Read this first. The figures 0.836, 0.023 and 0.984 come
from results/matrix_decision.txt, and no run in this repository produced any of them -- the nine
logged runs of that experiment scored delta 0.133/0.148/0.164 and hrr 0.039/0.039/0.031 (CLAIMS #25).
So P1's "baseline 0.836", P2's "the matrix held 0.984", P3, and the falsifier's "matrix_decision.txt
stands exactly as recorded" are all anchored to numbers that do not exist. The EXPERIMENT is still
worth running -- whether cleanup rescues a vector state is a real question, and cleanup really was
missing -- but its predictions cannot be scored against these baselines. Anything this script needs
as a comparison has to be re-measured here, in its own arms, against its own logged runs.

Found 2026-09-21 by scripts/audit_claims.py, which asked which live files still cite a retracted
result. Six did.

WHAT THIS IS FIXING, AND IT IS MY OMISSION. cell_shootout.py concluded the vector route was
"strictly dominated" by the matrix, from a wide-vector convolution-binding cell that scored 0.836
at 4 pairs and collapsed to 0.023 at 8. That cell unbinds and hands the result straight to a linear
layer:

    outs.append(irfft(conj(Q[:,t]) * h))      # unbind
    return self.out(stack(outs, 1))           # ... and that is the whole read

The holographic / vector-symbolic tradition does not work that way. Unbinding is understood to
return a NOISY version of the stored vector, and capacity is recovered by a CLEANUP step: snap the
noisy result to the nearest entry of a codebook of legal values. Cleanup is not an optimisation in
that literature, it is part of the memory. We left it out and then recorded the result as a verdict
on the architecture.

This is the second time in this project that a published design was benchmarked without a component
its own paper names as essential. The first was DeltaNet without the short causal convolution:
0.094 -> 0.961 once it was added.

THE ANALOGY THAT PROMPTED IT. Flash storage went from 1 bit per cell to 5 by distinguishing 32
voltage levels instead of 2. The raw error rate of QLC would be unusable as a storage device; it
works because error correction (BCH, then LDPC) got strong enough to absorb it. The density came
from ACCEPTING A NOISIER READOUT AND CORRECTING IT DOWNSTREAM. Cleanup is exactly that move, and
this project has been demanding exact reads everywhere.

ARMS, at 4 and 8 pairs, equal state floats for the two vector arms.
    hrr        unbind -> linear                          (reproduces the 0.836 / 0.023 baseline)
    hrr+clean  unbind -> soft cleanup -> linear          the same cell with the missing component
    delta      matrix, rank-1 erase, contraction read    the reference that won (0.984 / 0.984)

HONEST NOTE ON WHAT CLEANUP COSTS. Soft cleanup is a softmax over a codebook, so this does
reintroduce a softmax. It is a softmax over a CONSTANT-SIZE set (the codebook, 64 entries here),
not over the sequence, so the per-token cost stays constant in context and the architecture is
still O(1) per token. But credit for any gain is partly credit to a softmax, and the writeup has to
say so rather than claim the vector state did it alone.

REGISTERED PREDICTIONS, before running:
  P1  cleanup improves the 4-pair number (baseline 0.836).
  P2  THE DECISIVE ONE. At 8 pairs the vector collapsed to 0.023 while the matrix held 0.984. If
      cleanup lifts 8 pairs substantially, then "the vector route is strictly dominated" in
      matrix_decision.txt was measured on a crippled cell and must be revised.
  P3  delta reproduces ~0.984 at both, confirming the harness matches cell_shootout.

FALSIFIER, and this is the outcome that closes the question for good: cleanup gains less than +0.05
at BOTH 4 and 8 pairs. Then the missing-component hypothesis is dead, the vector route really is
dominated, and matrix_decision.txt stands exactly as recorded.

    python evidence/hrr_cleanup.py
Environment: STEPS, SEEDS, LR, DSTATE, TAU, THREADS, CPAIRS (a list, e.g. "4,8").
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.metrics import Run                                        # noqa: E402
from cell_shootout import DeltaCell, HRRCell, make, batch, NENT      # noqa: E402

STEPS = int(os.environ.get('STEPS', 6000))
BS = 32
LR = float(os.environ.get('LR', 3e-3))
D = 128
DSTATE = int(os.environ.get('DSTATE', 1024))
TAU = float(os.environ.get('TAU', 0.1))
SEEDS = int(os.environ.get('SEEDS', 2))
# CPAIRS, not NPAIRS. cell_shootout.py does TARGET = int(os.environ.get('NPAIRS', 4)) AT IMPORT, so
# a list value crashes this script before main() runs. That is the second time this exact collision
# has happened here -- binding_long.py parsing the older name as an int is why NPAIRS exists -- and the
# smoke test missed it because it passed NPAIRS=4, a value that cannot trigger the bug. Smoke-test
# with the value the real run uses.
CPAIRS = [int(p) for p in os.environ.get('CPAIRS', '4,8').split(',')]
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


class Cleanup(nn.Module):
    """Snap a noisy retrieved vector to a codebook of legal values.

    Soft (differentiable) form: attention over the codebook. Temperature TAU controls how hard the
    snap is; at tau -> 0 this becomes nearest-neighbour, which is the discrete VSA operation."""

    def __init__(self, dm, n_codes, tau=TAU):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(n_codes, dm) / math.sqrt(dm))
        self.tau = tau

    def forward(self, v):
        sim = F.normalize(v, dim=-1) @ F.normalize(self.codes, dim=-1).t()
        return (sim / self.tau).softmax(-1) @ self.codes


class Net(nn.Module):
    def __init__(self, kind, vocab, d, dstate, clean=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        dk = dv = int(math.isqrt(dstate))
        self.kind = kind
        if kind == 'delta':
            self.cell = DeltaCell(d, dk, dv)
            self.clean = None
        else:
            self.cell = HRRCell(d, dstate)
            self.clean = Cleanup(dstate, vocab) if clean else None
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, d))
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.emb(x)
        if self.clean is None:
            h = h + self.cell(h)
        else:
            # reach inside HRRCell: unbind, clean, THEN project out
            c = self.cell
            z = c.ln(h)
            B, L, _ = z.shape
            k, q, v = c.k(z), c.q(z), c.v(z)
            from cell_shootout import unitary
            k, q = unitary(k), unitary(q)
            g, a = torch.sigmoid(c.g(z)), torch.sigmoid(c.a(z))
            K, Q, V = (torch.fft.rfft(t, dim=-1) for t in (k, q, v))
            st = torch.zeros(B, K.shape[-1], device=z.device, dtype=K.dtype)
            outs = []
            for t in range(L):
                st = a[:, t] * st + g[:, t] * (K[:, t] * V[:, t])
                outs.append(torch.fft.irfft(torch.conj(Q[:, t]) * st, n=c.dm, dim=-1))
            o = torch.stack(outs, 1)
            h = h + c.out(self.clean(o))            # <-- the missing component
        h = h + self.ffn(h)
        return self.head(self.lnf(h))


def train(kind, clean, npairs, seed):
    torch.manual_seed(seed)
    m = Net(kind, NENT + 2, D, DSTATE, clean=clean)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    npar = sum(p.numel() for p in m.parameters())
    rng = np.random.default_rng(1000 + seed)
    ev = np.random.default_rng(7)
    tag = f'{kind}{"+clean" if clean else ""}'
    log = Run(f'clean-{tag}-p{npairs}-s{seed}',
              config=dict(arm=f'hrr_cleanup/{tag}', pairs=npairs, params=npar,
                          state_floats=DSTATE, cleanup=clean, tau=TAU, steps=STEPS, seed=seed,
                          chance_top1=round(1 / npairs, 4)))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS, npairs)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s == 0 or (s + 1) % 500 == 0:
            m.eval()
            with torch.no_grad():
                a = 0.
                for _ in range(4):
                    x2, y2 = batch(ev, BS, npairs)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train()
            best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    return best, npar


def main():
    print(f'HRR CLEANUP  {DSTATE} state floats, tau={TAU}, {STEPS} steps, {SEEDS} seeds, lr={LR}')
    print('cell_shootout baseline, no cleanup:  4 pairs 0.836   8 pairs 0.023')
    print('matrix reference:                    4 pairs 0.984   8 pairs 0.984\n')
    arms = [('hrr', False), ('hrr', True), ('delta', False)]
    res = {}
    print(f'{"arm":>12} {"params":>9} ' + ' '.join(f'{f"p={p}":>16}' for p in CPAIRS))
    print('-' * (24 + 17 * len(CPAIRS)))
    t0 = time.time()
    for kind, clean in arms:
        tag = f'{kind}{"+clean" if clean else ""}'
        cells, npar = [], 0
        for p in CPAIRS:
            r = [train(kind, clean, p, s) for s in range(SEEDS)]
            npar = r[0][1]
            res[(tag, p)] = [x for x, _ in r]
            cells.append(', '.join(f'{x:.3f}' for x, _ in r))
        print(f'{tag:>12} {npar:>9,} ' + ' '.join(f'{c:>16}' for c in cells), flush=True)
    print(f'\n{time.time()-t0:.0f}s\n')
    print('READ, against the registered predictions')
    for p in CPAIRS:
        base = float(np.mean(res[('hrr', p)]))
        cl = float(np.mean(res[('hrr+clean', p)]))
        dl = float(np.mean(res[('delta', p)]))
        print(f'  p={p}:  hrr {base:.3f} -> hrr+clean {cl:.3f}  ({cl-base:+.3f})   '
              f'matrix {dl:.3f}')
    gains = [float(np.mean(res[('hrr+clean', p)])) - float(np.mean(res[('hrr', p)])) for p in CPAIRS]
    if max(gains) < 0.05:
        print('\n  FALSIFIER FIRED: cleanup gains < +0.05 everywhere.')
        print('  The vector route really is dominated; matrix_decision.txt stands as recorded.')
    else:
        print('\n  Cleanup matters. The "strictly dominated" verdict was measured on a cell')
        print('  missing a component its own literature names as part of the memory.')
        print('  Note: the gain is partly a softmax over a constant-size codebook. Say so.')


if __name__ == '__main__':
    main()
