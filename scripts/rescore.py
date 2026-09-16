"""Score a trained checkpoint with the corrected evaluation protocol.

Every context length is scored at the SAME positions: a window of length L ends at one of 64
positions drawn once (seed 99), and only the final 128 characters are scored. Until 2026-09-16 the
window starts were drawn per length, so the columns of the results table scored different text and
differences between them mixed in sampling noise.

    python scripts/rescore.py checkpoints/saryu_v4b_last.pt
    EVAL_LENS=128,512 CHARS=2000000 python scripts/rescore.py checkpoints/saryu_25m.pt

Environment: CORPUS (corpus/enwik8), EVAL_LENS, SCORE_LAST (128), EVAL_WINDOWS (64), EVAL_ANCHOR
(8192, the earliest scoring position), THREADS.
The vocabulary and the 95/5 split are built exactly as in scripts/train.py.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from saryu.model import load_checkpoint  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_LENS = [int(x) for x in os.environ.get('EVAL_LENS', '128,512,2048,8192').split(',') if x.strip()]
SCORE_LAST = int(os.environ.get('SCORE_LAST', 128))
EVAL_WINDOWS = int(os.environ.get('EVAL_WINDOWS', 64))
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


def load(vocab_min=100):
    """enwik8 as character ids, exactly as scripts/train.py encodes it."""
    path = os.environ.get('CORPUS', os.path.join(ROOT, 'corpus', 'enwik8'))
    raw = open(path, 'rb').read().decode('utf-8', errors='ignore')
    if os.environ.get('CHARS'):
        raw = raw[:int(os.environ['CHARS'])]
    cps = np.frombuffer(raw.encode('utf-32-le'), dtype=np.uint32)
    uniq, counts = np.unique(cps, return_counts=True)
    keep = np.sort(uniq[counts >= vocab_min])
    pos = np.clip(np.searchsorted(keep, cps), 0, len(keep) - 1)
    return torch.tensor(np.where(keep[pos] == cps, pos + 1, 0).astype(np.int64)), len(keep) + 1


def eval_ends(va, seed=99):
    """Same positions for every length, anchored at EVAL_ANCHOR (not at the lengths requested), so
    runs that evaluate at different subsets of lengths stay comparable."""
    lo = int(os.environ.get('EVAL_ANCHOR', 8192)) + 1
    if lo >= len(va) - 1:
        lo = max(1, len(va) // 2)
    ge = torch.Generator().manual_seed(seed)
    return torch.randint(lo, len(va) - 1, (EVAL_WINDOWS,), generator=ge).tolist()


@torch.no_grad()
def bpc_at(m, va, L, ends):
    if L > min(ends):
        return float('nan')
    per = max(1, min(32, 8192 // L))
    tot, ntok = 0.0, 0
    for i in range(0, len(ends), per):
        ch = ends[i:i + per]
        x = torch.stack([va[e - L:e] for e in ch]).to(DEV)
        y = torch.stack([va[e - L + 1:e + 1] for e in ch]).to(DEV)
        lg = m(x)[:, -SCORE_LAST:]
        yy = y[:, -SCORE_LAST:]
        tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), yy.reshape(-1), reduction='sum'))
        ntok += yy.numel()
    return tot / ntok / math.log(2)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'checkpoints', 'saryu_v4b_last.pt')
    data, V = load()
    va = data[int(0.95 * len(data)):]
    model, ck = load_checkpoint(path, map_location=DEV)
    model.eval().to(DEV)
    n = sum(p.numel() for p in model.parameters())
    print('{}  arm {!r}  d={}  {:,} params  vocab {}  device {}'.format(
        os.path.basename(path), ck.get('arm'), ck['d'], n, V, DEV), flush=True)
    print('  {:,} evaluation characters, {} windows x last {} characters, same positions at every L'
          .format(len(va), EVAL_WINDOWS, SCORE_LAST), flush=True)
    ends = eval_ends(va)
    for L in EVAL_LENS:
        t0 = time.time()
        print('  bpc@{:<5} {:.4f}   ({:.0f}s)'.format(L, bpc_at(model, va, L, ends), time.time() - t0),
              flush=True)


if __name__ == '__main__':
    main()
