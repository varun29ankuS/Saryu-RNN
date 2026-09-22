"""Does the chunk-parallel kernel actually beat a sequential RNN, and does the gap grow with length?

WHY THIS IS NOW THE DECIDING MEASUREMENT. The efficiency experiment (results/efficiency_hybrid.txt)
found a GRU with pre-norm and residuals TIES saryu-plain on final loss: 2.190/2.193 against
2.189/2.176. That falsified P2 as stated. But it does not falsify the architecture's actual claim,
because the claim was never "better than a GRU at language modelling".

A GRU is NONLINEAR, so it was never inside the TC^0 class the state-tracking results are about --
those results say DIAGONAL and LINEAR recurrences cannot state-track, and a GRU is neither. What
the delta-rule / Householder family offers over a classical RNN is not expressivity. It is
expressivity WITH PARALLEL TRAINING: a GRU must be walked one token at a time, and the chunk
kernel need not be.

So the honest claim after the efficiency run is "GRU-quality, parallel-trainable". The second half
of that sentence has never been measured in this repository, and it is the half doing all the work.

WHAT IS MEASURED. Training step time against sequence length at CONSTANT TOKENS PER STEP, so this
is throughput and not batch size. If the parallel kernel is real, Saryu's advantage over the GRU
should WIDEN with length: the GRU's sequential dependency grows linearly in T while the chunk
kernel's serial depth grows as T/C. At ctx 128 the measured gap is only 1.4x (0.69 against 0.98
s/step), which is not yet evidence of anything.

REGISTERED PREDICTIONS, before running:
  P1  Saryu/GRU step-time ratio improves monotonically with sequence length. This is the
      parallel-training claim, stated so it can fail.
  P2  the ratio exceeds 2x by ctx 2048. Below that, "parallel-trainable" is an architectural
      property we cannot cash at any length we can actually train at.

FALSIFIER, and it is a live possibility rather than a formality: nn.GRU dispatches to a FUSED
oneDNN kernel written in C++, while our chunk path is a Python loop issuing many small tensor ops.
A fused sequential kernel can beat an unfused parallel one for a long way. If the ratio is flat or
shrinks, the parallel advantage is theoretical at this scale, the honest position becomes "a GRU
with extra steps", and the case has to rest on GPU kernels we have not written.

Prior evidence that this can go against us: the same measurement against a transformer found
Saryu's DISADVANTAGE widening with length (3.9x at 128 to 5.8x at 2048), because the scan loses
batch parallelism as context grows while attention stays one fused call.

    python evidence/parallel_advantage.py
Environment: TOKENS, LENS, REPS, THREADS.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                    # noqa: E402
from efficiency import GRUArm, TransformerArm, CHUNKSZ               # noqa: E402

V = 481
TOKENS = int(os.environ.get('TOKENS', 2048))          # constant per step: throughput, not batch
LENS = [int(x) for x in os.environ.get('LENS', '128,256,512,1024,2048').split(',')]
REPS = int(os.environ.get('REPS', 3))
torch.set_num_threads(int(os.environ.get('THREADS', max(1, (os.cpu_count() or 4) - 2))))


def step_time(m, B, T, reps):
    x = torch.randint(0, V, (B, T))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(1):                                 # warm up allocator and any lazy init
        F.cross_entropy(m(x).reshape(-1, V), x.reshape(-1)).backward()
        opt.step(); opt.zero_grad()
    t = time.time()
    for _ in range(reps):
        F.cross_entropy(m(x).reshape(-1, V), x.reshape(-1)).backward()
        opt.step(); opt.zero_grad()
    return (time.time() - t) / reps


def main():
    print(f'PARALLEL ADVANTAGE  {TOKENS} tokens/step held constant, chunk {CHUNKSZ}, '
          f'{REPS} reps, ~5M params')
    print('claim under test: "GRU-quality, PARALLEL-TRAINABLE". Only the second half is new,')
    print('and it has never been measured here.\n')
    print(f'{"ctx":>6} {"batch":>6} {"saryu":>9} {"gru":>9} {"tf":>9} {"gru/saryu":>11} '
          f'{"tf/saryu":>10}')
    print('-' * 66)
    rows = []
    for T in LENS:
        B = max(1, TOKENS // T)
        torch.manual_seed(0)
        s = step_time(SaryuV3LM(V, 328, 4, chunk=CHUNKSZ, mem_chunk=CHUNKSZ), B, T, REPS)
        torch.manual_seed(0)
        g = step_time(GRUArm(V, 440, 4), B, T, REPS)
        torch.manual_seed(0)
        f = step_time(TransformerArm(V, 312, 4, True, maxlen=max(LENS)), B, T, REPS)
        rows.append((T, B, s, g, f))
        print(f'{T:>6} {B:>6} {s:>9.2f} {g:>9.2f} {f:>9.2f} {g/s:>10.2f}x {f/s:>9.2f}x',
              flush=True)

    print('\nREAD, against the registered predictions')
    ratios = [g / s for _, _, s, g, _ in rows]
    mono = all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:]))
    print(f'  gru/saryu by length: ' + ' -> '.join(f'{r:.2f}x' for r in ratios))
    print(f'  P1 (ratio grows with length): {"HOLDS" if mono else "FALSIFIED"}')
    last = ratios[-1]
    print(f'  P2 (ratio > 2x at ctx {rows[-1][0]}): '
          f'{"HOLDS" if last > 2 else "FALSIFIED"} -- measured {last:.2f}x')
    if not mono or last <= 2:
        print('  The parallel advantage is not cashable at any length trainable here.')
        print('  nn.GRU is a FUSED C++ kernel and our chunk path is a Python loop over small ops;')
        print('  a fused sequential kernel beating an unfused parallel one is the likely cause,')
        print('  and the fix is a written kernel, not an architectural argument.')
    else:
        print('  The parallel kernel is real and it widens with context, which is the half of')
        print('  "GRU-quality, parallel-trainable" that the efficiency run could not show.')


if __name__ == '__main__':
    main()
