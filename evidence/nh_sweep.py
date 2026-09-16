"""Does the trace law hold in a TRAINED language model, or only by construction?

THE GAP THIS CLOSES. Both papers rest on one unifying claim: n_h, the number of reflections per
token, is a single knob with state tracking at one end and associative binding at the other, because
the expected overlap of a vector with its image is the normalised trace and a product of k
reflections gives overlap ~ exp(-2k/d_h). That claim has been checked

    by construction     evidence/recall_by_construction.py, evidence/trace_law.py -- weights set by
                        HAND, no training. This measures representability, not learnability.
    in a lookup table   evidence/word_problem.py -- one learned transport per group element.

and never in the trained model itself. Until now SaryuV3LM could not even vary n_h: its constructor
did not forward the argument, so every model ever trained in this project used n_h = 2. A law that
has never been tested where it is applied is a conjecture, so this tests it.

THE TWO QUESTIONS, deliberately separated:
  1. Does raising n_h actually lower the trace of the learned transport? The law is derived for
     RANDOM reflections at beta = 2. A trained model chooses beta in [0,2] and may choose correlated
     directions, either of which keeps the transport nearer the identity than the law assumes.
  2. Does a lower trace buy recall? That is the consequence the papers actually claim.
     Question 1 can succeed while 2 fails; then the law is right about geometry and wrong about
     what the model does with it.

THE EXACT LAW. Each reflection multiplies the expected trace by (1 - 2/d_h), so the trace after k of
them is (1 - 2/d_h)^k exactly; exp(-2k/d_h) is its large-d_h approximation. trace_law.py measured at
d = 64, where the two agree to three decimals. Here d_h = 16 and they visibly differ (0.118 against
0.135 at k = 16), so both are reported and the exact form is the one predicted.

A UNIT TEST THAT COMES FIRST. At initialisation b_proj.bias = pi, so beta = 1 - cos(pi) = 2 exactly
and the directions are random: precisely the law's regime. Measured at init, the trace is 0.8750,
0.7665, 0.5850, 0.3437, 0.1165 for n_h = 1, 2, 4, 8, 16 against an exact (1-2/16)^k of 0.8750,
0.7656, 0.5862, 0.3436, 0.1181. The measurement is therefore sound, and anything the trained models
do differently is the model's doing, not the instrument's.

REGISTERED PREDICTIONS, written before the sweep was run. At D = 128 with H = 8 heads, d_h = 16:

        n_h        2        4        8       16
    (1-2/d_h)^k    0.766    0.586    0.344    0.118

  P1  measured trace falls monotonically with n_h and is within 0.15 of the exact value.
  P2  4-pair recall at n_h = 16 exceeds recall at n_h = 2 by at least 0.20.

FALSIFIERS. If P1 holds and P2 fails, the geometry is right but n_h is not the recall knob, and the
unifying claim in both papers must be softened to "by construction". If P1 fails, the trained model
does not use the capacity n_h gives it, and the trace -- not n_h -- is the thing to report.

    python evidence/nh_sweep.py
Environment: NHS, D, NL, NPAIRS, GAP, STEPS, LR, BS, SEEDS, THREADS.
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
from saryu.model import SaryuV3LM                                     # noqa: E402

NHS = [int(x) for x in os.environ.get('NHS', '2,4,8,16').split(',')]
D = int(os.environ.get('D', 128))
NL = int(os.environ.get('NL', 2))
NPAIRS = int(os.environ.get('NPAIRS', 4))
GAP = int(os.environ.get('GAP', 64))
STEPS = int(os.environ.get('STEPS', 1500))
BS = int(os.environ.get('BS', 32))
# recall_distance.py: the 2026-09-15 triage measured 4-pair MQAR at lr 3e-4 -> 1.000 on 3/3 seeds,
# jumping at step 1000-1500, and at lr 1e-3 -> below 0.21. Use the rate that works.
LR = float(os.environ.get('LR', 3e-4))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0').split(',')]
NENT = 64
torch.set_num_threads(int(os.environ.get('THREADS', 8)))


def make_recall(rng, npairs, gap, seqlen, nent=NENT):
    """k1 v1 ... kn vn, `gap` filler tokens, SEP k_i -> v_i. Filler is drawn from entities that are
    neither keys nor values, so it can never be mistaken for an answer."""
    sep, pad = nent, nent + 1
    ent = rng.choice(nent, size=2 * npairs, replace=False)
    ks, vs = ent[:npairs], ent[npairs:]
    rest = np.setdiff1d(np.arange(nent), ent)
    seq = []
    for k, v in zip(ks, vs):
        seq += [int(k), int(v)]
    seq += [int(x) for x in rng.choice(rest, size=gap, replace=True)]
    i = int(rng.integers(0, npairs))
    seq += [sep, int(ks[i])]
    if len(seq) > seqlen:
        return None
    return [pad] * (seqlen - len(seq)) + seq, int(vs[i])


def batch(rng, npairs, gap, seqlen, bs):
    xs, ys = [], []
    while len(xs) < bs:
        s = make_recall(rng, npairs, gap, seqlen)
        if s is None:
            continue
        xs.append(s[0]); ys.append(s[1])
    return torch.tensor(xs), torch.tensor(ys)


def measured_trace(m, x):
    """The quantity the law predicts: E[v . g v] = tr(g)/d_h for g the token's transport.

    Built exactly as SaryuV3Block.forward builds it -- the same projections, the same beta -- then
    assembled into an explicit d_h x d_h matrix per (token, head) so the trace is exact rather than
    sampled. Averaged over tokens and heads."""
    blk = m.mix[0]
    with torch.no_grad():
        z = blk.ln(m.emb(x))
        z = z + blk.conv(z.transpose(1, 2))[..., :z.shape[1]].transpose(1, 2)
        B, L = x.shape
        u = F.normalize(blk.v_proj(z).view(B, L, blk.H, blk.nh, blk.dh), dim=-1)
        beta = 1.0 - torch.cos(blk.b_proj(z).view(B, L, blk.H, blk.nh))
        # g = prod_i (I - beta_i u_i u_i^T); accumulate over the nh reflections.
        g = torch.eye(blk.dh).expand(B, L, blk.H, blk.dh, blk.dh).clone()
        for i in range(blk.nh):
            ui = u[:, :, :, i]                                    # (B,L,H,dh)
            bi = beta[:, :, :, i]
            hh = torch.einsum('blhij,blhj->blhi', g, ui)          # g u_i
            g = g - bi[..., None, None] * hh[..., :, None] * ui[..., None, :]
        tr = torch.diagonal(g, dim1=-2, dim2=-1).sum(-1) / blk.dh
        return float(tr.mean()), float(beta.mean())


def run(nh, seed):
    seqlen = 2 * NPAIRS + GAP + 2
    V = NENT + 2
    torch.manual_seed(seed)
    m = SaryuV3LM(V, D, NL, nh=nh)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
    rng = np.random.default_rng(1000 + seed)
    t0 = time.time()
    every = max(1, STEPS // 4)
    for s in range(STEPS):
        x, y = batch(rng, NPAIRS, GAP, seqlen, BS)
        loss = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if (s + 1) % every == 0:
            el = time.time() - t0
            print(f'      nh={nh} sd{seed} {s+1:>5}/{STEPS} loss {float(loss):.3f} '
                  f'{el:.0f}s elapsed, {el/(s+1)*(STEPS-s-1):.0f}s left', flush=True)
    m.eval()
    ev = np.random.default_rng(99)
    hit = n = 0
    with torch.no_grad():
        for _ in range(8):
            x, y = batch(ev, NPAIRS, GAP, seqlen, BS)
            hit += int((m(x)[:, -1].argmax(-1) == y).sum()); n += len(y)
        xt, _ = batch(ev, NPAIRS, GAP, seqlen, 8)
        tr, bmean = measured_trace(m, xt)
    return hit / n, tr, bmean, time.time() - t0


def main():
    dh = D // 8
    print('DOES THE TRACE LAW HOLD IN A TRAINED MODEL?')
    print(f'MQAR {NPAIRS} pairs, gap {GAP}, d={D}, {NL} layers, d_h={dh}, lr={LR:g}, '
          f'{STEPS} steps, seeds {SEEDS}')
    print(f'chance = 1/{NENT} = {1/NENT:.3f}\n')
    print(f'{"n_h":>4} {"sd":>3} {"recall":>8} {"trace":>8} {"exact":>8} {"exp form":>9} '
          f'{"dev":>7} {"mean beta":>10}')
    print('-' * 64)
    res = {}
    for nh in NHS:
        for sd in SEEDS:
            acc, tr, bmean, el = run(nh, sd)
            pred = (1 - 2 / dh) ** nh                 # exact; exp(-2nh/dh) is the limit form
            res.setdefault(nh, []).append((acc, tr))
            print(f'{nh:>4} {sd:>3} {acc:>8.3f} {tr:>8.3f} {pred:>8.3f} '
                  f'{math.exp(-2*nh/dh):>9.3f} {tr-pred:>+7.3f} {bmean:>10.3f}'
                  f'   ({el:.0f}s)', flush=True)

    print('\nREGISTERED PREDICTIONS')
    trs = [sum(t for _, t in res[nh]) / len(res[nh]) for nh in NHS]
    accs = [sum(a for a, _ in res[nh]) / len(res[nh]) for nh in NHS]
    mono = all(trs[i] > trs[i + 1] for i in range(len(trs) - 1))
    worst = max(abs(trs[i] - (1 - 2 / dh) ** nh) for i, nh in enumerate(NHS))
    p1 = mono and worst < 0.15
    print(f'  P1 trace falls monotonically with n_h and within 0.15 of the law:')
    print(f'     monotone {mono}, worst deviation {worst:.3f}  -> {"HOLDS" if p1 else "FAILS"}')
    gain = accs[-1] - accs[0]
    p2 = gain >= 0.20
    print(f'  P2 recall at n_h={NHS[-1]} exceeds n_h={NHS[0]} by >= 0.20:')
    print(f'     {accs[0]:.3f} -> {accs[-1]:.3f}, gain {gain:+.3f}  -> {"HOLDS" if p2 else "FAILS"}')
    print()
    if p1 and p2:
        print('  Both hold: n_h is the knob in a trained model, not only by construction.')
    elif p1 and not p2:
        print('  P1 without P2: the geometry is right, but n_h alone does not buy recall here.')
        print('  The unifying claim in both papers must be softened to "by construction".')
    elif not p1:
        print('  P1 fails: the trained model does not drive the trace where n_h allows.')
        print('  Report the measured trace, not n_h, as the variable that matters.')


if __name__ == '__main__':
    main()
