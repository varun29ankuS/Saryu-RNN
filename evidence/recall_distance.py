"""Can the model USE long-range information at all, or does it merely fail to need it?

WHY THIS EXISTS. Four gate experiments ended with this (evidence/results/gate_timescales.txt): with
a hard per-head ceiling the state provably RETAINS long memory -- retentions of 877, 3166, 10078
tokens, pinned on 68-100% of tokens -- and the model's predictions past ~512 characters are
unchanged, at a cost of 0.04-0.05 bpc. Retained but unused. Two explanations survive and they point
opposite ways:

    "need not"   character-level enwik8 at 1M parameters does not reward memory past a few hundred
                 characters, so the architecture is fine and the gate work was measuring the task.
    "cannot"     the convex write dilutes what is stored, or the readout cannot address it, so the
                 write rule is the constraint and that is where the work belongs.

A language-model loss cannot separate those. A task whose answer REQUIRES a fact from a controlled
distance can.

THE TWO TASKS, and why both are needed. Saryu's vector state is built to track WHERE a sequence is,
not to store many facts, so recall alone would be an unfair test whose failure is easy to over-read.
  recall     make_mqar with a gap: k1 v1 ... kn vn, GAP filler tokens, SEP k_i -> v_i. The gap is
             swept independently of the number of pairs, so DISTANCE is separated from LOAD.
  transport  make_transport (ordered): the relations arrive in path order, so a transport model
             composes as it reads and needs no storage. This is the task the vector state IS for,
             and it is the control: if recall fails while transport holds at the same distance, the
             failure is about storage, not about carrying information forward.

PREDICTION, registered before running:
  1. If the ceiling arm reaches high recall accuracy at gaps of 500+ while the baseline collapses,
     the architecture CAN use long-range state and enwik8 simply does not ask it to.
  2. If every arm succeeds at short gaps and fails at long ones while TRANSPORT stays flat, the
     write/readout is the constraint for stored facts specifically, which is the case for stacking
     a matrix memory on the state rather than fixing the gate.
  3. If transport also degrades with length, the problem is carrying information at all, and the
     recurrence itself is implicated.

    python evidence/recall_distance.py                       # baseline
    GATE_TIMESCALES=1 GATE_CEILING=1 python evidence/recall_distance.py
Environment: ARM (label), STEPS, D, NL, BS, NPAIRS, GAPS, CHAIN_LENS, SEED, THREADS,
GATE_TIMESCALES, GATE_CEILING, WRITE_SCALE, GATE_W_SCALE.

REFERENCES
  [Arora et al. 2023] Zoology: Measuring and Improving Recall in Efficient Language Models, arXiv 2312.04927
  [Haller et al. 2025] What Matters in Linearizing Language Models?, arXiv 2504.14366
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
sys.path.insert(0, os.path.join(ROOT, 'experimental', 'memory'))
from saryu.model import SaryuV3LM                                     # noqa: E402
from recall_tasks import make_transport, vocab                        # noqa: E402

ARM = os.environ.get('ARM', 'base')
STEPS = int(os.environ.get('STEPS', 1500))
D = int(os.environ.get('D', 128))
NL = int(os.environ.get('NL', 2))
BS = int(os.environ.get('BS', 32))
NPAIRS = int(os.environ.get('NPAIRS', 4))
GAPS = [int(x) for x in os.environ.get('GAPS', '16,64,256,512').split(',')]
CHAIN_LENS = [int(x) for x in os.environ.get('CHAIN_LENS', '4,16,48').split(',')]
SEED = int(os.environ.get('SEED', 0))
NENT, NREL = 64, 4
torch.set_num_threads(int(os.environ.get('THREADS', 8)))

MODEL_KW = dict(timescales=os.environ.get('GATE_TIMESCALES', '0') == '1',
                write_scale=os.environ.get('WRITE_SCALE', '0') == '1',
                gate_ceiling=os.environ.get('GATE_CEILING', '0') == '1',
                gate_w_scale=float(os.environ.get('GATE_W_SCALE', 0.01)))


def make_recall(rng, npairs, gap, seqlen, nent=NENT):
    """k1 v1 ... kn vn, `gap` filler tokens, SEP k_i -> v_i.

    The filler is drawn from entities that are neither keys nor values, so it cannot be mistaken for
    an answer, and the gap moves DISTANCE without moving the number of facts to store. Left-padded,
    so the queried key is the last token, as recall_tasks requires."""
    sep, qry, pad, _ = vocab(nent)
    ent = rng.choice(nent, size=2 * npairs, replace=False)
    ks, vs = ent[:npairs], ent[npairs:]
    rest = np.setdiff1d(np.arange(nent), ent)
    seq = []
    for k, v in zip(ks, vs):
        seq += [int(k), int(v)]
    seq += [int(x) for x in rng.choice(rest, size=gap, replace=True)]
    i = int(rng.integers(0, npairs))
    dist = gap + 2 * (npairs - i)                 # tokens between the queried value and the query
    seq += [sep, int(ks[i])]
    if len(seq) > seqlen:
        return None
    return [pad] * (seqlen - len(seq)) + seq, int(vs[i]), dist


def batch(rng, kind, arg, seqlen):
    xs, ys, ds = [], [], []
    while len(xs) < BS:
        s = (make_recall(rng, NPAIRS, arg, seqlen) if kind == 'recall'
             else make_transport(rng, arg, NREL, seqlen, NENT, ordered=True))
        if s is None:
            continue
        x, y, d = s
        xs.append(x); ys.append(y); ds.append(d)
    return torch.tensor(xs), torch.tensor(ys), ds


def run(kind, args, seqlen, label):
    _, _, _, V = vocab(NENT, NREL)
    torch.manual_seed(SEED)
    m = SaryuV3LM(V, D, NL, **MODEL_KW)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=STEPS, pct_start=0.1)
    rng = np.random.default_rng(1000 + SEED)
    t0 = time.time()
    every = max(1, STEPS // 8)
    for s in range(STEPS):
        arg = args[s % len(args)]                 # interleave, so one model sees every distance
        x, y, _ = batch(rng, kind, arg, seqlen)
        loss = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if (s + 1) % every == 0:                  # without this the run is silent until an arm ends,
            el = time.time() - t0                 # which made a slow run look like a hung one
            print(f'    {label} {s+1:>5}/{STEPS}  loss {float(loss):.3f}  {el:.0f}s elapsed, '
                  f'{el/(s+1)*(STEPS-s-1):.0f}s left', flush=True)
    m.eval()
    out = []
    ev = np.random.default_rng(99)
    with torch.no_grad():
        for arg in args:
            hit = n = 0
            for _ in range(8):
                x, y, ds = batch(ev, kind, arg, seqlen)
                hit += int((m(x)[:, -1].argmax(-1) == y).sum()); n += len(y)
            out.append((arg, hit / n, int(np.mean(ds))))
    print(f'  {label:<10} ' + '  '.join(f'{a}:{acc:.3f}(d~{d})' for a, acc, d in out)
          + f'   [{time.time()-t0:.0f}s, {STEPS} steps]', flush=True)
    return out


if __name__ == '__main__':
    seqlen = 2 * NPAIRS + max(GAPS) + 2
    chain_seqlen = 3 * max(CHAIN_LENS) + max(CHAIN_LENS) + 4
    print(f'arm {ARM!r}  d={D} nl={NL} steps={STEPS} seed={SEED}  options={MODEL_KW}')
    print(f'recall: {NPAIRS} pairs, gaps {GAPS}, sequence {seqlen}  |  chance = 1/{NENT}')
    run('recall', GAPS, seqlen, 'recall')
    print(f'transport (ordered, the control): lengths {CHAIN_LENS}, sequence {chain_seqlen}')
    run('transport', CHAIN_LENS, chain_seqlen, 'transport')
