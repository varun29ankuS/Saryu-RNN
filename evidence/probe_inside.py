"""Open the model. Is the answer PRESENT in the state at 4 pairs, or genuinely absent?

WHY THIS EXISTS. Every diagnostic in this project is input-output: accuracy, in_set, the assignment
matrix, the binding score, sigma1. All of them describe what comes out of the readout. None of them
says what is INSIDE. We now have a model that binds 2 pairs perfectly (muon, sigma1 = 1.000) and one
that fails 4 pairs after 15,000 steps (sigma1 = 0.003), and we have never looked at either.

Four geometric hypotheses about what the STATE can represent have failed this session -- holonomy
and carving, path signatures and the level-2 term, the Birkhoff barycentre, and the trace law's
n_h prescription. The one that worked was about how PARAMETERS MOVE. That pattern says we have been
doing the geometry of assumed mechanisms rather than of the learned one.

THE QUESTION THIS SETTLES. At 4 pairs, is the correct value

    PRESENT BUT UNREADABLE   the state at the query position contains the answer, and the readout
                             cannot extract it. Then the bottleneck is the readout, not memory, and
                             the whole capacity story is aimed at the wrong component.
    GENUINELY ABSENT         the state does not carry it. Then it IS capacity or binding, and the
                             trace law's prescription is the right family of fix.

These need opposite repairs, and no measurement so far distinguishes them.

HOW. Train, then freeze, then fit a LINEAR PROBE from the hidden state at the query position to the
correct value, on held-out sequences. A linear probe can only read what is linearly present, so:

    probe accuracy >> model accuracy   the information is there and the model is not using it
    probe accuracy ~ model accuracy    the model is already extracting what there is
    probe accuracy ~ 1/n_pairs         the state genuinely does not carry the association

Probes are fitted at every layer, and at the residual stream before and after each block, so the
answer also says WHERE the information appears -- or where it is lost.

CONTROL. The same probe is run on the 2-pair model, which binds perfectly. If the probe reads ~1.0
there and ~1/n at 4 pairs, the probe itself is sound and the difference is real.

    python evidence/probe_inside.py
Environment: PAIRS, GAP, STEPS, LR, OPT, NH, SEED, PROBE_N, THREADS.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saryu.model import SaryuV3LM                                     # noqa: E402
from binding_long import Muon                                         # noqa: E402
from saryu.metrics import Run                                         # noqa: E402

PAIRS = int(os.environ.get('PAIRS', 2))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 6000))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 3e-4))
OPT = os.environ.get('OPT', 'muon')
NH = int(os.environ.get('NH', 2))
SEED = int(os.environ.get('SEED', 0))
PROBE_N = int(os.environ.get('PROBE_N', 4000))
D, NL, NENT = 128, 2, 64
SEP = NENT
torch.set_num_threads(int(os.environ.get('THREADS', 8)))


def make(rng):
    ent = rng.choice(NENT, size=2 * PAIRS, replace=False)
    ks, vs = ent[:PAIRS], ent[PAIRS:]
    rest = np.setdiff1d(np.arange(NENT), ent)
    seq = []
    for k, v in zip(ks, vs):
        seq += [int(k), int(v)]
    seq += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, PAIRS))
    seq += [SEP, int(ks[i])]
    return seq, [int(v) for v in vs], i


def batch(rng, bs):
    xs, vs, ii = zip(*(make(rng) for _ in range(bs)))
    return torch.tensor(xs), torch.tensor(vs), torch.tensor(ii)


def train():
    torch.manual_seed(SEED)
    m = SaryuV3LM(NENT + 2, D, NL, nh=NH)
    p = list(m.parameters())
    opt = (torch.optim.AdamW(p, lr=LR, weight_decay=0.01) if OPT == 'adamw' else Muon(p, lr=LR))
    rng = np.random.default_rng(1000 + SEED)
    log = Run(os.environ.get('RUN', f'probe-{OPT}-p{PAIRS}-nh{NH}'),
              config=dict(arm=f'probe/{OPT}', pairs=PAIRS, gap=GAP, lr=LR, nh=NH, steps=STEPS,
                          chance_top1=round(1 / NENT, 4),
                          chance_loss=round(float(np.log(NENT)), 3)))
    ev = np.random.default_rng(5)
    t0 = time.time()
    every = max(1, STEPS // 30)
    for s in range(STEPS):
        x, vs, ii = batch(rng, BS)
        ce = F.cross_entropy(m(x)[:, -1], vs[torch.arange(len(ii)), ii])
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(p, 1.0); opt.step()
        if s == 0 or (s + 1) % every == 0:
            m.eval()
            with torch.no_grad():
                x2, v2, i2 = batch(ev, 128)
                t2 = v2[torch.arange(len(i2)), i2]
                acc = float((m(x2)[:, -1].argmax(-1) == t2).float().mean())
            m.train()
            log.log(s + 1, loss=float(ce), **{'eval/top1': acc})
        if (s + 1) % max(1, STEPS // 3) == 0:
            print(f'    step {s+1}/{STEPS}  ce {float(ce):.3f}  {time.time()-t0:.0f}s', flush=True)
    log.done()
    return m


@torch.no_grad()
def collect(m, n):
    """Hidden states at the QUERY position (the last token), at every depth."""
    rng = np.random.default_rng(31337)
    feats = {}
    ys, accs = [], []
    done = 0
    while done < n:
        x, vs, ii = batch(rng, min(BS, n - done))
        tgt = vs[torch.arange(len(ii)), ii]
        # replicate the forward pass, keeping the residual stream at each stage
        h = m.emb(x)
        ve = m.vemb(x) if m.vemb is not None else None
        taps = {'emb': h}
        for li, (mix, ffn) in enumerate(zip(m.mix, m.ffn)):
            h = h + mix(h, ve)
            taps[f'L{li}_mix'] = h
            h = h + ffn(h)
            taps[f'L{li}_ffn'] = h
        taps['final'] = m.lnf(h)
        for k, v in taps.items():
            feats.setdefault(k, []).append(v[:, -1].clone())     # query position only
        ys.append(tgt)
        accs.append((m(x)[:, -1].argmax(-1) == tgt).float())
        done += len(tgt)
    return ({k: torch.cat(v) for k, v in feats.items()}, torch.cat(ys),
            float(torch.cat(accs).mean()))


def probe(X, y, n_class, epochs=260, lr=0.05):
    """Multinomial logistic regression, held-out split. Reads only what is LINEARLY present."""
    n = len(y); ntr = int(n * 0.8)
    Xtr, Xte, ytr, yte = X[:ntr], X[ntr:], y[:ntr], y[ntr:]
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    W = torch.zeros(X.shape[1], n_class, requires_grad=True)
    b = torch.zeros(n_class, requires_grad=True)
    o = torch.optim.Adam([W, b], lr=lr)
    for _ in range(epochs):
        loss = F.cross_entropy(Xtr @ W + b, ytr)
        o.zero_grad(); loss.backward(); o.step()
    with torch.no_grad():
        return float((( Xte @ W + b).argmax(-1) == yte).float().mean())


def main():
    print(f'PROBE INSIDE  {PAIRS} pairs, gap {GAP}, {OPT} lr {LR:g}, n_h {NH}, {STEPS} steps')
    print(f'chance = 1/{NENT} = {1/NENT:.3f}   rank-one readout would give {1/PAIRS:.3f}\n')
    m = train(); m.eval()
    feats, y, acc = collect(m, PROBE_N)
    print(f'\n  model accuracy on the same data: {acc:.3f}')
    print(f'\n  {"tap":<10} {"probe acc":>10}   reads the answer out of the residual stream')
    print('  ' + '-' * 56)
    for k in ['emb'] + [f'L{i}_{w}' for i in range(NL) for w in ('mix', 'ffn')] + ['final']:
        if k in feats:
            p = probe(feats[k], y, NENT + 2)
            flag = ('  <- answer is present here' if p > max(acc + 0.15, 1.5 / PAIRS) else '')
            print(f'  {k:<10} {p:>10.3f}{flag}')
    print('\nREAD')
    print('  probe >> model  : the state carries the answer and the readout cannot extract it')
    print('                    -- the bottleneck is the readout, not memory.')
    print('  probe ~ model   : the model already uses what is there.')
    print(f'  probe ~ {1/PAIRS:.3f}   : the association is genuinely not in the state.')


if __name__ == '__main__':
    main()
