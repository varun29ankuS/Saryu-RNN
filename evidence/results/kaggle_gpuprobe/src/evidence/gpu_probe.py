"""What can this architecture actually train on one T4 in a week? The number nothing else can be planned without.

THIS MODEL HAS NEVER RUN ON A GPU. Every result in this repository is CPU, 5M parameters, context
128. The kernel is unfused PyTorch eager -- a Python loop issuing many small tensor operations per
chunk -- and on CPU that cost 2.7x against a transformer at equal parameters
(results/parallel_advantage.txt and the profile behind it: ~10,000 aten calls per two steps,
aten::mm 23.6% of time across 380 calls, aten::copy_ 4,280 calls).

A GPU punishes exactly that pattern harder than a CPU does, because kernel-launch overhead is a
fixed cost per operation and the operations here are small. It is entirely possible that this
model gets WORSE relative to a transformer on a T4 than it is on a laptop. Nobody knows, because
nobody has measured it, and every plan for the rest of the week depends on the answer.

WHAT IS MEASURED. Tokens per second for the model at several widths and context lengths, against
a parameter-matched transformer on the same device, plus the memory each needs. Then the only
number that matters for planning: at this throughput, how many tokens fit in a 9-hour Kaggle
session, and what does that make feasible.

REGISTERED PREDICTIONS, before running:
  P1  the transformer/Saryu throughput ratio on a T4 is WORSE than the 2.0-2.9x measured on CPU,
      because launch overhead dominates for small operations and the T4 has far more parallelism
      to leave idle. This is the prediction I expect to hold and it is bad news.
  P2  the ratio improves with batch size, since larger batches amortise launch overhead across
      more work per call. If it does, the plan is large batches; if it does not, the kernel needs
      rewriting before anything else is worth doing.
  P3  a 25-50M model trains on >= 500M tokens within two 9-hour sessions. That is the smallest
      scale at which a character-level model says anything, and if it does not fit, the week's
      target has to change rather than the schedule tighten.

FALSIFIER, and the decision it forces: if throughput implies under ~100M tokens per session at
any usable size, then no amount of scheduling produces a model this week, and the honest move is
to say so on day one rather than on Friday.

    python evidence/gpu_probe.py
Environment: STEPS, WIDTHS, LENS, BATCHES, THREADS.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from saryu.model import SaryuV3LM                                    # noqa: E402

V = 512
STEPS = int(os.environ.get('STEPS', 20))
WIDTHS = [int(x) for x in os.environ.get('WIDTHS', '328,512,768').split(',')]
LENS = [int(x) for x in os.environ.get('LENS', '512,1024').split(',')]
BATCHES = [int(x) for x in os.environ.get('BATCHES', '8,32').split(',')]
CHUNK = int(os.environ.get('CHUNK', 32))
NL = int(os.environ.get('NL', 8))
SESSION_HOURS = float(os.environ.get('SESSION_HOURS', 9))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class TF(nn.Module):
    def __init__(self, vocab, d, nl, nh=8, maxlen=4096):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(d, nh, 4 * d, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl)
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))[None]
        m = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        return self.head(self.lnf(self.enc(h, mask=m)))


def bench(build, d, B, T, steps=STEPS):
    """Tokens/sec and peak memory. Returns None if the configuration does not fit."""
    try:
        torch.manual_seed(0)
        m = build(d).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x = torch.randint(0, V, (B, T), device=DEV)
        if DEV == 'cuda':
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        for _ in range(3):                                   # warm up cudnn/allocator
            F.cross_entropy(m(x).reshape(-1, V), x.reshape(-1)).backward()
            opt.step(); opt.zero_grad()
        if DEV == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            F.cross_entropy(m(x).reshape(-1, V), x.reshape(-1)).backward()
            opt.step(); opt.zero_grad()
        if DEV == 'cuda':
            torch.cuda.synchronize()
        dt = (time.time() - t0) / steps
        peak = torch.cuda.max_memory_allocated() / 2**30 if DEV == 'cuda' else 0.0
        npar = sum(p.numel() for p in m.parameters())
        del m, opt, x
        if DEV == 'cuda':
            torch.cuda.empty_cache()
        return B * T / dt, dt, peak, npar
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            if DEV == 'cuda':
                torch.cuda.empty_cache()
            return None
        raise


def main():
    print(f'GPU PROBE  device={DEV}, {NL} layers, chunk {CHUNK}, {STEPS} timed steps')
    print('this architecture has never run on a GPU. The kernel is unfused eager, which a GPU')
    print('punishes harder than a CPU does. Everything this week depends on the number below.\n')
    print(f'{"model":>8} {"d":>5} {"B":>4} {"T":>5} {"params":>11} {"tok/s":>10} '
          f'{"s/step":>8} {"peakGB":>7}')
    print('-' * 68)
    rows = []
    for d in WIDTHS:
        for T in LENS:
            for B in BATCHES:
                for tag, build in (('saryu', lambda w: SaryuV3LM(V, w, NL, chunk=CHUNK,
                                                                 mem_chunk=CHUNK)),
                                   ('tf', lambda w: TF(V, w, NL))):
                    r = bench(build, d, B, T)
                    if r is None:
                        print(f'{tag:>8} {d:>5} {B:>4} {T:>5} {"":>11} {"OOM":>10}', flush=True)
                        continue
                    tps, dt, peak, npar = r
                    rows.append((tag, d, B, T, tps, dt, peak, npar))
                    print(f'{tag:>8} {d:>5} {B:>4} {T:>5} {npar:>11,} {tps:>10,.0f} '
                          f'{dt:>8.3f} {peak:>7.2f}', flush=True)

    print('\nREAD, against the registered predictions')
    sar = {(d, B, T): tps for t, d, B, T, tps, *_ in rows if t == 'saryu'}
    tfr = {(d, B, T): tps for t, d, B, T, tps, *_ in rows if t == 'tf'}
    both = sorted(set(sar) & set(tfr))
    if both:
        print('  transformer / saryu throughput ratio (>1 means the transformer is faster):')
        for k in both:
            print(f'    d={k[0]:<5} B={k[1]:<4} T={k[2]:<5}  {tfr[k]/sar[k]:.2f}x')
        worst = max(tfr[k] / sar[k] for k in both)
        best = min(tfr[k] / sar[k] for k in both)
        print(f'  P1 worse than CPU\'s 2.0-2.9x: {"HOLDS" if best > 2.9 else "not shown"} '
              f'(range {best:.2f}-{worst:.2f}x)')
        by_b = sorted({k[1] for k in both})
        if len(by_b) > 1:
            lo = min(tfr[k] / sar[k] for k in both if k[1] == by_b[0])
            hi = min(tfr[k] / sar[k] for k in both if k[1] == by_b[-1])
            print(f'  P2 ratio improves with batch: B={by_b[0]} {lo:.2f}x -> B={by_b[-1]} '
                  f'{hi:.2f}x -> {"HOLDS" if hi < lo else "FALSIFIED -- the kernel needs a rewrite"}')
    if sar:
        print(f'\n  TOKENS PER {SESSION_HOURS:g}-HOUR SESSION, the planning number:')
        for (d, B, T), tps in sorted(sar.items()):
            tok = tps * SESSION_HOURS * 3600
            print(f'    d={d:<5} B={B:<4} T={T:<5}  {tok/1e6:>8.0f}M tokens/session')
        best_tok = max(tps for tps in sar.values()) * SESSION_HOURS * 3600
        print(f'  P3 >= 500M tokens in two sessions: '
              f'{"HOLDS" if 2 * best_tok >= 5e8 else "FALSIFIED"} '
              f'({2*best_tok/1e6:.0f}M at the best configuration)')
        if 2 * best_tok < 2e8:
            print('\n  FALSIFIER FIRED. Under 100M tokens per session at any usable size means no')
            print('  amount of scheduling produces a model this week. Say so now, not on Friday.')


if __name__ == '__main__':
    main()
