"""Is the conjunctive task solvable at all? A matched transformer says.

conjunctive.py found each half solved at 1.000 and the conjunction at 0.39-0.44 against chance
0.333, with the matrix memory making no difference. Two readings are indistinguishable without a
baseline: the architecture cannot compose two things it can each do, OR the task is too hard at
this size and step budget for anything.

That is exactly the hole that produced the retracted matrix decision -- 170 runs at 0.32 that meant
nothing until a transformer scored 1.000 on the same task and showed the wall was ours.

REGISTERED PREDICTION: the transformer solves 'both' (> 0.8). Then the conjunction is learnable and
the failure is Saryu's.
FALSIFIER: the transformer also lands near chance. Then the task is not learnable at this budget,
conjunctive.py measures nothing about composition, and the task needs redesigning before any
conclusion is drawn from it.

    python evidence/conjunctive_baseline.py
"""
import os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.metrics import Run                                       # noqa: E402
from conjunctive import batch, VOCAB, NGEN, BS                      # noqa: E402

STEPS = int(os.environ.get('STEPS', 3000))
SEEDS = int(os.environ.get('SEEDS', 2))
D, NL, NHEAD = 128, 2, 4
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


class TF(nn.Module):
    def __init__(self, vocab, d, nl, nhead, maxlen=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(maxlen, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, 2 * d, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl)
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)

    def forward(self, x):
        L = x.shape[1]
        h = self.emb(x) + self.pos[:L]
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        return self.head(self.lnf(self.enc(h, mask=mask)))


def train(task, seed):
    ngen, fixed = task
    torch.manual_seed(seed)
    m = TF(VOCAB, D, NL, NHEAD)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    rng = np.random.default_rng(1000 + seed)
    name = {(NGEN, False): 'both', (0, False): 'recall', (NGEN, True): 'track'}[task]
    log = Run(f'cbase-{name}-s{seed}',
              config=dict(arm=f'conjunctive_baseline/{name}', ngen=ngen, fixed_vals=fixed,
                          params=npar, steps=STEPS, seed=seed, chance_top1=0.3333))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS, ngen, fixed)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 250 == 0:
            ev = np.random.default_rng(7); a = 0.
            m.eval()
            with torch.no_grad():
                for _ in range(4):
                    x2, y2 = batch(ev, BS, ngen, fixed)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train(); best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    return best, npar


def main():
    print(f'CONJUNCTIVE BASELINE  matched transformer, d={D} L={NL} heads={NHEAD}, '
          f'{STEPS} steps, {SEEDS} seeds, chance 0.333')
    print('Saryu on these tasks: both 0.422/0.438 (mem off) 0.391/0.398 (mem on),')
    print('                      recall 1.000, track 1.000\n')
    t0 = time.time()
    for task, label in [((NGEN, False), 'both'), ((0, False), 'recall'), ((NGEN, True), 'track')]:
        r = [train(task, s) for s in range(SEEDS)]
        print(f'  {label:<8} {r[0][1]:>8,} params   ' + ', '.join(f'{b:.3f}' for b, _ in r),
              flush=True)
    print(f'\n{time.time()-t0:.0f}s')
    print('READ: transformer solves "both" -> the conjunction is learnable and the failure is ours.')
    print('      transformer also near chance -> the task is not learnable at this budget and')
    print('      conjunctive.py measures nothing about composition.')


if __name__ == '__main__':
    main()
