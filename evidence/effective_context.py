"""How much context does a trained Saryu checkpoint actually use?

THE QUESTION. The published results table reports bits per character at context 128, 512, 2048 and
8192 and reads the decrease as "the model uses context it was never trained on". Those numbers were
measured with window STARTS drawn separately per length, so each column scored different text and
part of the difference was sampling. Rescoring at identical positions (2026-09-16) made two columns
of the 5M checkpoint land on the same value to four decimals, which is what prompted this probe.

WHAT THIS MEASURES, on one fixed set of scoring positions:
  1. bits per character at many context lengths, printed to six decimals;
  2. how much the model's PREDICTIONS change when context is added: the mean KL divergence between
     the next-character distribution given L characters of context and given the longest context,
     over the same scored positions;
  3. the state agreement: the cosine similarity between the recurrent state entering the scored
     region under context L and under the longest context.

PREDICTION, stated before running. If the model truly uses long context, both the KL and the state
disagreement should fall steadily as L grows. If its memory saturates at some length, they will
collapse to ~0 beyond that length and the bpc column will stop moving -- and then the claim in the
README and the paper has to be narrowed to that length.

    python evidence/effective_context.py checkpoints/saryu_v4b_last.pt
Environment: CORPUS, LENS, SCORE_LAST (128), EVAL_WINDOWS (32), EVAL_ANCHOR (16384), THREADS.

REFERENCES
  [Grazzi et al. 2024] Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues, arXiv 2411.12537
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

LENS = [int(x) for x in os.environ.get('LENS', '128,256,512,1024,2048,4096,8192').split(',')]
SCORE_LAST = int(os.environ.get('SCORE_LAST', 128))
WINDOWS = int(os.environ.get('EVAL_WINDOWS', 32))
ANCHOR = int(os.environ.get('EVAL_ANCHOR', 16384))
torch.set_num_threads(int(os.environ.get('THREADS', 8)))


def load(vocab_min=100):
    path = os.environ.get('CORPUS', os.path.join(ROOT, 'corpus', 'enwik8'))
    raw = open(path, 'rb').read().decode('utf-8', errors='ignore')
    cps = np.frombuffer(raw.encode('utf-32-le'), dtype=np.uint32)
    uniq, counts = np.unique(cps, return_counts=True)
    keep = np.sort(uniq[counts >= vocab_min])
    pos = np.clip(np.searchsorted(keep, cps), 0, len(keep) - 1)
    return torch.tensor(np.where(keep[pos] == cps, pos + 1, 0).astype(np.int64))


@torch.no_grad()
def states_before_scored(model, x):
    """The per-layer recurrent state at the last position before the scored region.

    Runs the model's own blocks, keeping each block's post-recurrence state for the final token, so
    two context lengths can be compared at the SAME text position."""
    out = []
    h = model.emb(x)
    ve = model.vemb(x) if model.vemb is not None else None
    for blk, ffn in zip(model.mix, model.ffn):
        z = blk.ln(h)
        z = blk.conv(z.transpose(1, 2))[:, :, :h.shape[1]].transpose(1, 2)
        u, beta, a, b = blk.mix(z, ve)
        B, L, H, nh, dh = u.shape
        uf = u.permute(0, 2, 1, 3, 4).reshape(B * H, L, nh, dh)
        bf = beta.permute(0, 2, 1, 3).reshape(B * H, L, nh)
        af = a.permute(0, 2, 1).reshape(B * H, L)
        bbf = b.permute(0, 2, 1, 3).reshape(B * H, L, dh)
        h0 = blk.h0[None].expand(B, H, dh).reshape(B * H, dh)
        from saryu.model import chunkwise
        s = chunkwise(h0, uf, bf, af, bbf)
        out.append(s.view(B, H, L, dh)[:, :, -1].reshape(B, -1).clone())   # state at the last token
        s = blk.hnorm(s.view(B, H, L, dh).permute(0, 2, 1, 3)).reshape(B, L, blk.d)
        h = h + blk.out(s * F.silu(blk.og(z)))
        h = h + ffn(h)
    return out


@torch.no_grad()
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'checkpoints', 'saryu_v4b_last.pt')
    data = load()
    va = data[int(0.95 * len(data)):]
    model, ck = load_checkpoint(path)
    model.eval()
    ge = torch.Generator().manual_seed(99)
    ends = torch.randint(ANCHOR + 1, len(va) - 1, (WINDOWS,), generator=ge).tolist()
    ref = max(LENS)
    print('{}  arm {!r}  d={}  {:,} params'.format(os.path.basename(path), ck.get('arm'), ck['d'],
                                                   sum(p.numel() for p in model.parameters())))
    print('{} windows, scoring the last {} characters, longest context {} as the reference'
          .format(WINDOWS, SCORE_LAST, ref))
    print('{:>7} {:>12} {:>12} {:>14}   {}'.format('ctx', 'bpc', 'KL to ref', 'state cos', 'per layer'))
    print('-' * 78)
    cache, rows = {}, {}
    for L in [ref] + [x for x in sorted(LENS) if x != ref]:      # reference first, it is the baseline
        t0 = time.time()
        tot, ntok, kls, coss = 0.0, 0, [], []
        for e in ends:
            x = va[e - L:e][None]
            y = va[e - L + 1:e + 1][None]
            lg = model(x)[:, -SCORE_LAST:]
            tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y[:, -SCORE_LAST:].reshape(-1),
                                         reduction='sum'))
            ntok += SCORE_LAST
            # the state entering the scored region only exists if some context precedes it
            st = states_before_scored(model, va[e - L:e - SCORE_LAST][None]) if L > SCORE_LAST else None
            if L == ref:
                cache[e] = (lg.clone(), st)
            else:
                rlg, rst = cache[e]
                kls.append(float(F.kl_div(F.log_softmax(lg, -1), F.log_softmax(rlg, -1),
                                          log_target=True, reduction='batchmean')))
                if st is not None and rst is not None:
                    coss.append([float(F.cosine_similarity(s, r, dim=-1).mean()) for s, r in zip(st, rst)])
        bpc = tot / ntok / math.log(2)
        if L == ref:
            rows[L] = '{:>7} {:>12.6f} {:>12} {:>14}   (reference)  ({:.0f}s)'.format(
                L, bpc, '-', '-', time.time() - t0)
        elif coss:
            per = np.mean(coss, axis=0)
            rows[L] = '{:>7} {:>12.6f} {:>12.6f} {:>14.6f}   {}  ({:.0f}s)'.format(
                L, bpc, float(np.mean(kls)), float(per.mean()),
                ' '.join('{:.3f}'.format(c) for c in per), time.time() - t0)
        else:                                   # L == SCORE_LAST: nothing precedes the scored region
            rows[L] = '{:>7} {:>12.6f} {:>12.6f} {:>14}   {}  ({:.0f}s)'.format(
                L, bpc, float(np.mean(kls)), '-', '(no state probe)', time.time() - t0)
    for L in sorted(rows):
        print(rows[L], flush=True)
    print('\nRead: if KL and state disagreement collapse to ~0 beyond some length, the model does not'
          '\nuse context past it, and the results table should say so.')


if __name__ == '__main__':
    main()
