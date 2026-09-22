"""Can a GRU solve the S_3 word problem too? The control our one capability win never had.

WHY THIS MATTERS NOW. conjunctive.txt records the cleanest capability result in this project:
on the S_3 word problem -- pure non-commutative state tracking, three slots with fixed contents,
six generators shuffling them, answer what is in slot j -- Saryu scores 1.000/1.000 where a matched
transformer scores 0.562/0.859. Ablating the transport at the query position collapses it to 0.348
against a chance of 0.333, while ablating the memory costs nothing. That is a clean double
dissociation and it has been quoted as the architecture's strongest evidence.

IT WAS NEVER RUN AGAINST A GRU. That omission matters more after the efficiency experiment, which
found a properly-configured GRU TIES saryu-plain on language modelling. The theory says a
transformer cannot do this: fixed-depth log-precision attention is in TC^0, the word problem for a
non-solvable group is NC^1-complete, and regular languages are not all inside TC^0. The theory says
nothing of the kind about a GRU. A GRU is a NONLINEAR recurrence -- a finite automaton with
real-valued state -- and finite automata are exactly what recognises regular languages.

So the honest prior is that the GRU SOLVES IT, and that the transformer is the only arm the theory
predicts should fail. If that is what happens, the framing of our headline result has to change:
it is evidence that a recurrence beats attention at state tracking, not evidence that THIS
recurrence is special. The Householder/delta-rule literature is about recovering this ability in
LINEAR RNNs, which can be trained in parallel -- a GRU has the ability and cannot.

REGISTERED PREDICTIONS, before running:
  P1  the GRU solves S_3 (> 0.9). This is what the theory predicts and what I expect.
  P2  Saryu matches it at 1.000, reproducing conjunctive.txt.
  P3  the transformer does NOT reliably solve it, reproducing 0.562/0.859. That is the arm the
      complexity argument is actually about, and without it the experiment has no reference point.

WHAT EACH OUTCOME MEANS, written before the numbers so it cannot be reinterpreted afterwards:
  GRU solves it        -> our state-tracking result is "recurrence beats attention", NOT "Saryu
                          beats other recurrences". Every place the repo implies the latter needs
                          rewording, and the architecture's distinct claim rests entirely on
                          parallel trainability (evidence/parallel_advantage.py measured 3.7x over
                          a GRU at ctx 2048, so that claim is live).
  GRU fails            -> Saryu has a capability a classical nonlinear RNN lacks at this budget,
                          which would be a genuinely stronger result than anything recorded here
                          and would need explaining, because the theory does not predict it.

    python evidence/state_tracking_gru.py
Environment: STEPS, SEEDS, LR, THREADS.
"""
from __future__ import annotations

import os
import sys
import time

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                    # noqa: E402
from saryu.metrics import Run                                        # noqa: E402
from conjunctive import batch, VOCAB, NGEN                           # noqa: E402

STEPS = int(os.environ.get('STEPS', 3000))
SEEDS = int(os.environ.get('SEEDS', 2))
LR = float(os.environ.get('LR', 1e-3))
D, H, NL, BS = 128, 8, 2, 32
torch.set_num_threads(int(os.environ.get('THREADS', 4)))
# nn.TransformerEncoder emits a nested-tensor warning carrying an ABSOLUTE PATH, which is
# how absolute paths have now reached evidence/results/ three times and blocked a push.
warnings.filterwarnings('ignore', category=UserWarning)


class GRULM(nn.Module):
    """Pre-norm and residuals, matching the arm that tied us on enwik8. A bare stacked nn.GRU
    would make this control meaningless in the same way it nearly did there."""

    kind = 'gru'

    def __init__(self, vocab, d, nl):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(nl)])
        self.rnns = nn.ModuleList([nn.GRU(d, d, batch_first=True) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.emb(idx)
        for ln, rnn in zip(self.lns, self.rnns):
            h, _ = rnn(ln(x))
            x = x + h
        return self.head(self.lnf(x))


class TFLM(nn.Module):
    kind = 'transformer'

    def __init__(self, vocab, d, nl, nhead=4, maxlen=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(maxlen, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, 2 * d, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl)
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        L = x.shape[1]
        h = self.emb(x) + self.pos[:L]
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        return self.head(self.lnf(self.enc(h, mask=mask)))


def build(kind, seed):
    torch.manual_seed(seed)
    if kind == 'saryu':
        return SaryuV3LM(VOCAB, D, NL, H=H)
    if kind == 'gru':
        return GRULM(VOCAB, 184, NL)             # ~matched params to saryu at d=128
    return TFLM(VOCAB, D, NL)


def train(kind, seed):
    m = build(kind, seed)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(1000 + seed)
    log = Run(f'stg-{kind}-s{seed}',
              config=dict(arm=f'state_tracking_gru/{kind}', ngen=NGEN, nl=NL, d=D, params=npar,
                          steps=STEPS, seed=seed, lr=LR, chance_top1=0.3333))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS, NGEN, True)                    # fixed values -> pure S_3 tracking
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 250 == 0:
            ev = np.random.default_rng(7); a = 0.
            m.eval()
            with torch.no_grad():
                for _ in range(4):
                    x2, y2 = batch(ev, BS, NGEN, True)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train(); best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    return best, npar


def main():
    print(f'STATE TRACKING, WITH THE GRU CONTROL  S_3 word problem, {STEPS} steps, {SEEDS} seeds, '
          f'chance 0.333')
    print('conjunctive.txt: saryu 1.000/1.000, transformer 0.562/0.859. No GRU was ever run.')
    print('theory says a TRANSFORMER cannot do this (TC^0 vs NC^1). It says nothing against a')
    print('GRU, which is a nonlinear recurrence -- i.e. a finite automaton with real state.\n')
    t0 = time.time()
    res = {}
    print(f'{"arm":>14} {"params":>10}  best per seed')
    print('-' * 50)
    for kind in ('saryu', 'gru', 'transformer'):
        out = [train(kind, s) for s in range(SEEDS)]
        res[kind] = [a for a, _ in out]
        print(f'{kind:>14} {out[0][1]:>10,}  ' + ', '.join(f'{a:.3f}' for a in res[kind]),
              flush=True)
    print(f'\n{time.time()-t0:.0f}s\n')
    print('READ, against the registered predictions')
    g, s_, t = res['gru'], res['saryu'], res['transformer']
    gs = sum(a > 0.9 for a in g)
    print(f'  P1 GRU solves it: {gs}/{SEEDS} seeds above 0.9 -> '
          f'{"HOLDS" if gs else "FALSIFIED"}')
    print(f'  P2 saryu reproduces 1.000: {sum(a > 0.9 for a in s_)}/{SEEDS}')
    print(f'  P3 transformer does not reliably solve it: '
          f'{sum(a > 0.9 for a in t)}/{SEEDS} above 0.9')
    if gs:
        print('\n  THE GRU SOLVES IT. Our state-tracking result is "a recurrence beats attention",')
        print('  NOT "this recurrence beats other recurrences". Every place the repo implies the')
        print('  latter needs rewording. The architecture\'s distinct claim is then parallel')
        print('  trainability alone -- measured at 3.7x over a GRU at ctx 2048 in')
        print('  results/parallel_advantage.txt -- and not expressivity over classical RNNs.')
    else:
        print('\n  The GRU FAILS where Saryu succeeds. That is stronger than anything recorded')
        print('  here and the theory does not predict it, so it needs explaining before it is')
        print('  claimed: check budget, width, and learning rate before concluding anything.')


if __name__ == '__main__':
    main()
