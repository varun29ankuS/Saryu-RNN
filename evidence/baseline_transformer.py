"""The control this project never ran: does a comparable TRANSFORMER solve our MQAR task?

170+ runs in this repository measure the Saryu recurrence against itself. None establishes what the
task is worth. If a matched transformer also sits at 0.32, the task is harder than assumed at this
size and budget, and every architectural conclusion drawn from it is suspect. If the transformer
solves it, the wall is ours and the comparison finally has a scale.

Matched as closely as the architectures allow: same vocabulary, same sequences, same optimiser
family, same steps, same seeds, comparable parameter count. Two pair counts, because n=2 is the
regime Saryu solves and n=4 the one it never does -- so the transformer's n=2 result calibrates the
harness and its n=4 result is the actual question.

REGISTERED PREDICTIONS, before running:
  P1  transformer solves n=2 (> 0.9). If not, the harness is broken, not the architectures.
  P2  transformer solves n=4. The literature (Arora et al., Zoology) reports small transformers
      handling MQAR at this scale, so failure here would mean our task or budget is atypical.
  P3  Saryu's 0.32 at n=4 is therefore an architecture-specific wall rather than a task property.

FALSIFIER FOR THE WHOLE PROJECT'S FRAMING: the transformer also lands near 0.32 at n=4. Then the
wall is the task or the training budget, twenty-one interventions were measuring noise around an
unreachable target, and the conclusions in CLAIMS.md need revisiting.

    python evidence/baseline_transformer.py
Environment: NPAIRS, GAP, STEPS, LR, D, NL, NHEAD, SEEDS, THREADS.
"""
from __future__ import annotations
import os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                     # noqa: E402
from saryu.metrics import Run                                        # noqa: E402
from binding_long import Muon                                        # noqa: E402

PAIRS = [int(x) for x in os.environ.get('NPAIRS', '2,4').split(',')]
GAP = int(os.environ.get('GAP', 4)); STEPS = int(os.environ.get('STEPS', 5000)); BS = 32
LR = float(os.environ.get('LR', 3e-4)); D = int(os.environ.get('D', 128))
NL = int(os.environ.get('NL', 2)); NHEAD = int(os.environ.get('NHEAD', 8))
SEEDS = int(os.environ.get('SEEDS', 3)); NENT = 64; SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 4)))

class Tiny(nn.Module):
    """A standard pre-norm decoder-only transformer. Learned positions, causal mask."""
    def __init__(self, vocab, d, nl, nhead, maxlen=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d); self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=2 * d, batch_first=True,
                                           norm_first=True, dropout=0.0, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, nl)
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)
    def forward(self, x):
        L = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(L, device=x.device))[None]
        m = torch.triu(torch.full((L, L), float('-inf'), device=x.device), 1)
        return self.head(self.lnf(self.enc(h, mask=m)))

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

def train(kind, n, seed):
    torch.manual_seed(seed)
    if kind == 'transformer':
        m = Tiny(NENT + 2, D, NL, NHEAD)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    else:
        m = SaryuV3LM(NENT + 2, D, NL, nh=2, H=8)
        opt = Muon(list(m.parameters()), lr=LR)
    npar = sum(p.numel() for p in m.parameters())
    rng = np.random.default_rng(1000 + seed); ev = np.random.default_rng(7)
    log = Run(f'base-{kind}-p{n}-s{seed}',
              config=dict(arm=f'baseline/{kind}', pairs=n, params=npar, steps=STEPS, seed=seed,
                          chance_top1=round(1 / n, 4)))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS, n)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s == 0 or (s + 1) % 250 == 0:
            m.eval()
            with torch.no_grad():
                a = 0.
                for _ in range(4):
                    x2, y2 = batch(ev, BS, n)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train(); best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done(); return best, npar

def main():
    print(f'BASELINE  the control 170 runs never had. d={D}, {NL} layers, {STEPS} steps, '
          f'{SEEDS} seeds\n')
    print(f'{"pairs":>6} {"model":>12} {"params":>9} {"chance":>7}  best per seed')
    print('-' * 62)
    t0 = time.time()
    for n in PAIRS:
        for kind in ('transformer', 'saryu'):
            r = [train(kind, n, s) for s in range(SEEDS)]
            b = [x for x, _ in r]
            print(f'{n:>6} {kind:>12} {r[0][1]:>9,} {1/n:>7.3f}  '
                  + ', '.join(f'{x:.3f}' for x in b), flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ')
    print('  transformer solves n=4, saryu does not -> the wall is OURS, and now it has a scale')
    print('  both stuck near 0.32                   -> the wall is the TASK or the budget, and')
    print('                                            21 interventions were noise around an')
    print('                                            unreachable target')

if __name__ == '__main__':
    main()
