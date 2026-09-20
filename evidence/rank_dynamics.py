"""What IS the escape, geometrically? Track the rank of key-dependence through the transition.

WHAT WE KNOW. At 2 pairs the model sits at a rank-one, key-independent readout for ~8000 steps --
binding score B oscillating around zero, accuracy pinned at 1/2 -- and then breaks: B 0.00 -> 0.39
-> 0.59 -> 0.89, accuracy 0.52 -> 0.95 (binding_long.txt). The gradient pointed at binding the whole
time (cos = 0.75, binding_gradient.txt), so this was never a symmetry barrier. It is a plateau
followed by a sharp escape.

THE RIGHT COORDINATE. Let M be the n x n assignment, M[i,j] = P(value of pair j | key of pair i).
A key-independent readout has every row equal to the mean row, so

    R = M - 1 (mean row)^T        is EXACTLY zero when the readout ignores the key

R is therefore the key-dependence itself, and its SINGULAR VALUES say how much of it there is and of
what rank. That turns a vague question into a measurable one:

    sigma_1(R) = 0            rank-one plateau: no key-dependence at all
    sigma_1(R) > 0            one direction of key-dependence has appeared
    sigma_2(R) > 0            a second, and so on up to n-1 for a full permutation

THE HYPOTHESIS THIS TESTS. Long plateau then sharp jump, with the gradient aligned throughout, is
the signature of SADDLE-TO-SADDLE dynamics: gradient descent on a low-rank problem visits a sequence
of saddles of increasing rank, spending a long time at each and moving between them quickly. It is
well documented for deep linear networks and matrix factorisation (Saxe et al.; Gidel et al.;
Jacot et al.), where learning proceeds one singular direction at a time and the plateau length is
set by the initialisation scale and the singular-value gaps.

  PREDICTION P1  sigma_1(R) stays ~0 through the plateau and jumps at the transition, rather than
                 creeping up smoothly. The escape is a RANK INCREASE, not a gradual sharpening.
  PREDICTION P2  at 4 pairs, where key-dependence needs rank up to 3, the singular values switch on
                 IN SEQUENCE with separate plateaus -- one saddle per rank -- rather than together.
                 This is why more pairs cost so much more than proportionally.
  PREDICTION P3  accuracy tracks sigma_1 and not the loss: the loss falls during the plateau (it is
                 buying the value set and the positional prior) while accuracy does not move.

FALSIFIER: sigma_1(R) rising smoothly from step 0, with accuracy following it. Then there is no
saddle and no rank structure -- just slow continuous learning, and the plateau language is wrong.

    python evidence/rank_dynamics.py
Environment: PAIRS, GAP, STEPS, LR, EVERY, SEED, OPT, THREADS.
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
from saryu.metrics import Run                                         # noqa: E402
from binding_long import Muon                                         # noqa: E402

PAIRS = int(os.environ.get('PAIRS', 2))
GAP = int(os.environ.get('GAP', 4))
STEPS = int(os.environ.get('STEPS', 20000))
EVERY = int(os.environ.get('EVERY', 250))
BS = int(os.environ.get('BS', 32))
LR = float(os.environ.get('LR', 1e-3))
SEED = int(os.environ.get('SEED', 0))
OPT = os.environ.get('OPT', 'adamw')
D, NENT = 128, 64
NL = int(os.environ.get('NL', 2))   # depth: sequential selections available at read
GCAP = float(os.environ.get('GATE_CAP', 0.90))   # 1.0 = a true flip-flop (constant map)
NH = int(os.environ.get('NH', 2))   # reflections per token: the trace-law capacity knob
SEP = NENT
SEQ = 2 * PAIRS + GAP + 2
torch.set_num_threads(int(os.environ.get('THREADS', 8)))


def make(rng, qi=None):
    ent = rng.choice(NENT, size=2 * PAIRS, replace=False)
    ks, vs = ent[:PAIRS], ent[PAIRS:]
    rest = np.setdiff1d(np.arange(NENT), ent)
    seq = []
    for k, v in zip(ks, vs):
        seq += [int(k), int(v)]
    seq += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, PAIRS)) if qi is None else qi
    seq += [SEP, int(ks[i])]
    return seq, [int(v) for v in vs], i


def batch(rng, bs):
    xs, vs, ii = zip(*(make(rng) for _ in range(bs)))
    return torch.tensor(xs), torch.tensor(vs), torch.tensor(ii)


@torch.no_grad()
def assignment(m, n_seq=64):
    """M[i,j], built by asking every key of the SAME sequence so the rows share a value set."""
    ev = np.random.default_rng(99)
    M = np.zeros((PAIRS, PAIRS))
    for _ in range(n_seq):
        seq, vs, _ = make(ev, qi=0)
        rows = []
        for i in range(PAIRS):
            s = list(seq); s[-1] = s[2 * i]
            rows.append(s)
        p = F.softmax(m(torch.tensor(rows))[:, -1], -1)          # [PAIRS, V]
        mass = p[:, torch.tensor(vs)]                            # [PAIRS, PAIRS]
        M += (mass / mass.sum(1, keepdim=True).clamp_min(1e-9)).numpy()
    return M / n_seq


def main():
    print(f'RANK DYNAMICS  {PAIRS} pairs, gap {GAP}, {STEPS} steps, {OPT}, lr {LR:g}, seed {SEED}')
    print('R = M - 1(mean row)^T is EXACTLY 0 for a key-independent readout; its singular values')
    print(f'measure key-dependence and its rank (max rank {PAIRS-1}).\n')
    torch.manual_seed(SEED)
    m = SaryuV3LM(NENT + 2, D, NL, nh=NH, gate_cap=GCAP)
    params = list(m.parameters())
    # adamw is a DIAGONAL preconditioner: the step in each coordinate is divided by the running
    # root-second-moment, so a small but consistent gradient gets a step comparable to a large one.
    # sgd is the control with no preconditioner at all; muon orthogonalises momentum instead.
    opt = (torch.optim.AdamW(params, lr=LR, weight_decay=0.01) if OPT == 'adamw'
           else torch.optim.SGD(params, lr=LR, momentum=0.9) if OPT == 'sgd'
           else Muon(params, lr=LR))
    rng = np.random.default_rng(1000 + SEED)
    # The run name MUST include every swept knob. Sweeping lr with a fixed name made each
    # learning rate truncate the previous one, and two concurrent sweeps interleaved their points
    # into ONE file -- which is what made the dashboard look like a single run swinging wildly.
    tag = os.environ.get('RUN', f'rank-{OPT}-p{PAIRS}-lr{LR:g}-nh{NH}-L{NL}-g{GCAP:g}')
    log = Run(tag, config=dict(arm=OPT, pairs=PAIRS, gap=GAP, lr=LR, nh=NH, nl=NL, gate_cap=GCAP,
                                                  steps=STEPS, chance_top1=round(1 / NENT, 4),
                                                  chance_loss=round(float(np.log(NENT)), 3)))
    ns = min(PAIRS - 1, 3)
    hdr = (f'{"step":>7} {"ce":>7} {"acc":>7} {"diag":>7} ' +
           ' '.join(f'{"sig"+str(i+1):>8}' for i in range(ns)) + f' {"|R|":>8}')
    print(hdr); print('-' * len(hdr))
    ev = np.random.default_rng(7)
    t0 = time.time()
    for s in range(STEPS):
        x, vs, ii = batch(rng, BS)
        logits = m(x)[:, -1]
        ce = F.cross_entropy(logits, vs[torch.arange(len(ii)), ii])
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if s == 0 or (s + 1) % EVERY == 0:
            m.eval()
            M = assignment(m)
            R = M - M.mean(0, keepdims=True)          # exactly 0 if the readout ignores the key
            sig = np.linalg.svd(R, compute_uv=False)
            with torch.no_grad():
                acc = 0.
                for _ in range(6):
                    x2, v2, i2 = batch(ev, BS)
                    lg = m(x2)[:, -1]
                    acc += float((lg.argmax(-1) == v2[torch.arange(len(i2)), i2])
                                 .float().mean()) / 6
            m.train()
            diag = float(np.mean(np.diag(M)))
            log.log(s + 1, loss=float(ce), **{'eval/top1': acc, 'rank/diag_mass': diag,
                    'rank/R_norm': float(np.linalg.norm(R)),
                    **{f'rank/sigma{i+1}': float(sig[i]) for i in range(ns)}})
            print(f'{s+1:>7} {float(ce):>7.3f} {acc:>7.3f} {diag:>7.3f} '
                  + ' '.join(f'{sig[i]:>8.4f}' for i in range(ns))
                  + f' {np.linalg.norm(R):>8.4f}', flush=True)
    log.done()
    print(f'\n{time.time()-t0:.0f}s')
    print('READ: sigma_1 flat at ~0 then jumping = the escape is a RANK INCREASE (saddle-to-saddle).')
    print('      sigma_i switching on in sequence = one saddle per rank, which is why more pairs')
    print('      cost far more than proportionally.')


if __name__ == '__main__':
    main()
