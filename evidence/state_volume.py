"""The state as a space-time volume: are there persistent tubes, and do the two paths compose?

THE FRAMING, which came from Video Summagator (Nguyen, Niu & Liu, CHI 2012). Render a video as a
space-time cube and static content becomes a straight PRISM extruded along t, while moving content
becomes a slanted tube. Persistence and motion are shapes, not statistics.

Our state is the same kind of object -- [position x feature] per head -- and the two paths should
have visibly different geometry in it:

    transport   rotates the WHOLE state every token, so nothing is a straight prism by construction
    memory      erases rank-one and targeted, so an item whose key is orthogonal to k is UNTOUCHED
                -- it is a straight tube through the volume

So "tracking versus storage" is not only an accuracy claim. It predicts a measurable difference in
how long structure survives along the time axis, and that is what this measures.

WHY IT MATTERS MORE THAN THE ACCURACY NUMBER. memory_on.py showed 0.875/0.977 against 0.234/0.281,
which says the model got better. It does not say the memory is what held the facts -- a bigger
model with more parameters also gets better. This tests the MECHANISM: if the matrix is storing,
its read must persist across time where the transport's state does not.

AND IT TESTS COMPLEMENTARITY, which is assumed everywhere in this project and measured nowhere.
Every task so far isolates one axis: group word problems need order and no facts, MQAR needs facts
and no order. If the two paths compose, giving the model a memory should change what the transport
does -- it no longer has to try to store. If they interfere, it should not, or should get worse.
The one experiment that supposedly tested this was among the retracted numbers (CLAIMS.md #25).

MEASURED, on trained models, on real task batches:
  1. accuracy, as a sanity check against the logged memory_on result
  2. gate retention 1/g per head -- the transport's own tube length, and the quantity
     trained_geometry.txt item 5 found spanning 11 to 132 tokens in the shipped 25M model
  3. autocorrelation vs lag of the transport state, cos(h_t, h_{t+D}) averaged over t and heads
  4. the same for the memory read

P1 AS FIRST WRITTEN WAS A BAD TEST, and the smoke run showed it before the real one was spent.
It predicted the memory read would be more persistent than the transport state, and it is --
0.813 against 0.261 at lag 24 after only 250 steps. But the memory is an ACCUMULATOR: S += ...
with no decay term, so its read is autocorrelated by construction. Smoothness would pass that
test whether or not a single fact was retrievable, which makes it nearly unfalsifiable. The
autocorrelation is still reported below as a description; it is no longer a prediction.

REGISTERED PREDICTIONS, before running:
  P1  CAUSAL. The answer depends on the memory path. Zeroing the memory read AT THE QUERY
      POSITION ONLY collapses accuracy; zeroing the transport there does not, or does so less.
      Smoothness cannot pass this -- an accumulator that holds nothing retrievable loses nothing
      when it is ablated.
  P2  COMPLEMENTARITY. Turning the memory on CHANGES the transport's gate retention. Direction is
      deliberately not predicted; that it moves at all is the test, because it would mean the
      transport reallocates when it no longer has to store.

FALSIFIER for P1, and it is the one that matters: ablating the memory at the query position leaves
accuracy roughly intact. Then the matrix is not where the fact is, the gain came from somewhere
else (more parameters, a deeper effective path), and the mechanism claimed in model.py's comments
is wrong however good the score looks.

FALSIFIER for P2: gate retention is unchanged within seed spread. Then the paths are merely
additive, not complementary, and no claim of a division of labour is supported.

    python evidence/state_volume.py
Environment: STEPS, SEEDS, LR, THREADS.
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
from saryu.model import SaryuV3LM                                    # noqa: E402
from saryu.metrics import Run                                       # noqa: E402

STEPS = int(os.environ.get('STEPS', 2000))
SEEDS = int(os.environ.get('SEEDS', 2))
LR = float(os.environ.get('LR', 1e-3))
D, H, NH, NL, BS, NENT = 128, 8, 2, 2, 32, 64
SEP = NENT
GAP = int(os.environ.get('GAP', 24))      # long enough that persistence has work to do
PAIRS = 4
torch.set_num_threads(int(os.environ.get('THREADS', 4)))


def make(rng, n):
    e = rng.choice(NENT, size=2 * n, replace=False)
    ks, vs = e[:n], e[n:]
    rest = np.setdiff1d(np.arange(NENT), e)
    s = []
    for a, b in zip(ks, vs):
        s += [int(a), int(b)]
    s += [int(x) for x in rng.choice(rest, size=GAP, replace=True)]
    i = int(rng.integers(0, n))
    s += [SEP, int(ks[i])]
    return s, int(vs[i])


def batch(rng, bs, n):
    o = [make(rng, n) for _ in range(bs)]
    return torch.tensor([a for a, _ in o]), torch.tensor([b for _, b in o])


def autocorr(x, lags):
    """mean cos(x_t, x_{t+D}) over batch, heads and t, for each lag D.   x: [B, L, H, Dh]"""
    xn = F.normalize(x, dim=-1)
    out = []
    for d in lags:
        if d >= x.shape[1]:
            out.append(float('nan')); continue
        out.append(float((xn[:, :-d] * xn[:, d:]).sum(-1).mean()))
    return out


@torch.no_grad()
def probe(m, x):
    """Gate retention per head, and the two paths' state volumes, on a real batch."""
    caught = {}
    hooks = []

    def gate_hook(i, blk):
        def f(_m, _i, out):
            g = ((1.0 - torch.cos(out)) / 2.0).clamp(max=blk.gate_cap)
            if blk.g_max is not None:
                g = torch.minimum(g, blk.g_max)
            caught[f'g{i}'] = (1.0 / g.clamp(min=1e-6)).mean(dim=(0, 1))     # [H] tokens
        return f

    def hnorm_hook(i):
        def f(_m, _i, out):
            caught[f'h{i}'] = out.detach()                                   # [B,L,H,dh]
        return f

    for i, blk in enumerate(m.mix):
        hooks.append(blk.g_proj.register_forward_hook(gate_hook(i, blk)))
        hooks.append(blk.hnorm.register_forward_hook(hnorm_hook(i)))
        if blk.memory:
            hooks.append(blk.mnorm.register_forward_hook(
                (lambda i: (lambda _m, _i, out: caught.__setitem__(f'm{i}', out.detach())))(i)))
    m.eval(); m(x)
    for h in hooks:
        h.remove()
    return caught


@torch.no_grad()
def ablate(m, rng, memory):
    """Which path is the answer actually coming from?

    Zeroes one path's contribution AT THE FINAL POSITION ONLY -- where the query sits and the
    answer is read out -- and re-scores. A path that carries the fact loses accuracy when removed;
    a path that is merely smooth loses nothing."""
    m.eval()
    hooks, mode = [], {'kill': None}

    def mk_mem(blk):
        def f(_mod, _in, out):
            if mode['kill'] == 'memory':
                out = out.clone(); out[:, -1] = 0
            return out
        return f

    def mk_trans(blk):
        def f(_mod, _in, out):
            if mode['kill'] == 'transport':
                out = out.clone(); out[:, -1] = 0
            return out
        return f

    for blk in m.mix:
        hooks.append(blk.out.register_forward_hook(mk_trans(blk)))
        if blk.memory:
            hooks.append(blk.m_out.register_forward_hook(mk_mem(blk)))
    scores = {}
    for kill in (None, 'memory', 'transport'):
        if kill == 'memory' and not memory:
            continue
        mode['kill'] = kill
        ev = np.random.default_rng(7); a = 0.
        for _ in range(8):
            x2, y2 = batch(ev, BS, PAIRS)
            a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 8
        scores['full' if kill is None else kill] = a
    for h in hooks:
        h.remove()
    mode['kill'] = None
    return scores


def train(memory, seed):
    torch.manual_seed(seed)
    m = SaryuV3LM(NENT + 2, D, NL, nh=NH, H=H, memory=memory)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(1000 + seed)
    log = Run(f'vol-{"on" if memory else "off"}-p{PAIRS}-g{GAP}-s{seed}',
              config=dict(arm=f'state_volume/{"on" if memory else "off"}', pairs=PAIRS, gap=GAP,
                          memory=memory, steps=STEPS, seed=seed, lr=LR,
                          params=sum(p.numel() for p in m.parameters()), chance_top1=1 / PAIRS))
    best = 0.
    for s in range(STEPS):
        x, y = batch(rng, BS, PAIRS)
        ce = F.cross_entropy(m(x)[:, -1], y)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 250 == 0:
            ev = np.random.default_rng(7); a = 0.
            m.eval()
            with torch.no_grad():
                for _ in range(4):
                    x2, y2 = batch(ev, BS, PAIRS)
                    a += float((m(x2)[:, -1].argmax(-1) == y2).float().mean()) / 4
            m.train(); best = max(best, a)
            log.log(s + 1, loss=float(ce), **{'eval/top1': a, 'eval/best': best})
    log.done()
    xp, _ = batch(np.random.default_rng(7), BS, PAIRS)
    return best, probe(m, xp), ablate(m, np.random.default_rng(7), memory)


def main():
    L = 2 * PAIRS + GAP + 2
    lags = [1, 2, 4, 8, 16, 24]
    print(f'STATE VOLUME  {PAIRS} pairs, gap {GAP}, sequence length {L}, d={D} H={H} L={NL}, '
          f'{STEPS} steps, {SEEDS} seeds')
    print('memory_on.txt (gap 4, 3000 steps): off 0.234/0.281, on 0.875/0.977\n')
    t0 = time.time()
    res = {}
    for memory in (False, True):
        accs, probes, abls = [], [], []
        for sd in range(SEEDS):
            a, pr, ab = train(memory, sd)
            accs.append(a); probes.append(pr); abls.append(ab)
        res[memory] = (accs, probes, abls)
        print(f'  memory={str(memory):<5} recall {", ".join(f"{a:.3f}" for a in accs)}', flush=True)

    print('\n1. CAUSAL ABLATION at the query position -- where is the answer actually stored?')
    for memory in (False, True):
        for sd, ab in enumerate(res[memory][2]):
            cells = '  '.join(f'{k}={v:.3f}' for k, v in ab.items())
            drop = (f"   ablating memory costs {ab['full'] - ab['memory']:+.3f}"
                    if 'memory' in ab else '')
            print(f'   memory={str(memory):<5} seed {sd}: {cells}{drop}')

    print(f'\n2. GATE RETENTION 1/g per layer, mean over heads  (the transport\'s tube length)')
    for memory in (False, True):
        for i in range(NL):
            v = [float(p[f'g{i}'].mean()) for p in res[memory][1]]
            mx = [float(p[f'g{i}'].max()) for p in res[memory][1]]
            print(f'   memory={str(memory):<5} layer {i}: mean '
                  + ', '.join(f'{x:.1f}' for x in v) + '   max ' + ', '.join(f'{x:.1f}' for x in mx))

    print(f'\n3/4. AUTOCORRELATION vs lag   cos(x_t, x_t+D), mean over t and heads')
    print(f'{"path":>28}  ' + '  '.join(f'D={d:<4}' for d in lags))
    for memory in (False, True):
        for i in range(NL):
            ac = np.mean([autocorr(p[f'h{i}'], lags) for p in res[memory][1]], axis=0)
            print(f'{f"transport L{i}, memory={memory}":>28}  '
                  + '  '.join(f'{x:+.3f}' for x in ac))
    for i in range(NL):
        ps = [p for p in res[True][1] if f'm{i}' in p]
        if ps:
            ac = np.mean([autocorr(p[f'm{i}'], lags) for p in ps], axis=0)
            print(f'{f"MEMORY read L{i}":>28}  ' + '  '.join(f'{x:+.3f}' for x in ac))

    print(f'\n{time.time()-t0:.0f}s')
    print('READ')
    print('  P1 holds if ablating the memory at the query position collapses accuracy while')
    print('     ablating the transport there costs less. Smoothness cannot pass that.')
    print('  P1 FALSIFIED if accuracy survives the memory ablation -- then the fact is not in the')
    print('     matrix and the mechanism in model.py is wrong however good the score looks.')
    print('  P2 holds if gate retention moves when the memory is switched on.')
    print('  The autocorrelation table is DESCRIPTION, not a test: the memory is an accumulator')
    print('  and is autocorrelated by construction, so it would pass holding nothing.')


if __name__ == '__main__':
    main()
