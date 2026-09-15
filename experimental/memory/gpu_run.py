"""THE GPU RUN, rebuilt around the claim that survived scrutiny. Not launched by being written.

WHAT CHANGED AND WHY. The previous version tested cells at NL=2 where composition is NOT
REQUIRED: mqar-4 is one hop and chain-d2 is two, and a 2-layer model resolves a 2-hop chain by
LAYERING - layer 1 retrieves the first edge, layer 2 the second - with no S.S at all. Layered
retrieval is a shallower credit-assignment path, so gradient descent takes it whenever it is
available. We would have spent the last account on cells where the mechanism cannot appear, and
read the null as refutation. See PREDICTIONS.md P1.

Following that through: at SOTA depths of 48-92 layers, composition is essentially never
NECESSARY. So "multi-hop from one layer" is not a capability claim against a deep model. What
survives is better:

        HOP BUDGET MULTIPLIES ACROSS LAYERS.    ours n_layers x n_read,  theirs n_layers x 1

Layer 1 with n_read=n resolves n hops and writes the result into the residual stream; layer 2
resolves n more from there. To reach 48 hops they need 48 layers; we need 12 at n_read=4. THE
CLAIM IS PARAMETER EFFICIENCY, and the experiment is a LAYERS-vs-READS TRADE at equal hop budget.

THE DESIGN. Three arms all have hop budget 8 and wildly different parameter counts:

        NL=8 n_read=1     budget 8    full params      <- the incumbent's shape
        NL=4 n_read=2     budget 8    ~half
        NL=2 n_read=4     budget 8    ~quarter

If the quarter-parameter arm tracks the full one across chain depth, the claim is demonstrated.

PARAMS ARE NOT MATCHED BY WIDENING the shallow arms, deliberately: that would change head_dim and
so the state size, confounding the very thing under test. They are reported instead, and the
asymmetry is ONE-DIRECTIONAL - a win with fewer parameters is unimpeachable, a loss could be the
budget. The same logic already governs the tied/untied reading.

DEPTH IS THE PRIMARY AXIS, not a cell. The prediction is a gap that OPENS at depth 3 and WIDENS -
a trend across four depths is far harder to fake than one cell, and it is what distinguishes
"composition works" from "this cell happened to favour us".

EVERY OTHER CONSTRAINT IS INHERITED AND TRACED:
  ParityBlock, not SaryuBlock   the claim is tied-vs-untied x n_read, and ParityBlock has both
                                verified (cpu_precheck 6, trace_logic 14/14). SaryuBlock is always
                                tied and would add binding as a confound; its stack is a later run.
  tap_init='uniform'            smoke_corrected bug 2 - 'shift' starts the tap AT the answer
  channel_gates=False           MIRAS Eq 17 has no gates; with them the residual is not Sk-v
  robust_eps=1e-4               cpu_precheck 3: the sole non-finite cell of 18 is fp16, ||u||=0,
                                eps=1e-12. A T4 is Turing, no bf16, so fp16 is what runs.
  tied invariant AT RUNTIME     cpu_precheck 6 - two bugs once had the tied arm silently untied
                                while every other check passed green
  lag_cos, not tap_max          conv_vs_tap: the short conv supplies the lag by itself (a flat tap
                                composes 0.565 with conv, 0.011 without), so tap_max reads 0.25
                                while the mechanism works. P3 predicts exactly that.
  per-head decay logged         P4 predicts the distribution goes BIMODAL - slow heads for
                                composition, fast for local work. That split is the signature.
  conv=False arm                P3's corollary: removing the conv should FORCE the tap and RAISE
                                composition (0.832 vs 0.667 at a sharp tap).
  ceiling gate per cell         ceiling_sweep: a cell whose ceiling is under 0.90 ranks nothing.

RUNTIME, MEASURED RATHER THAN HOPED. The fused path is live (ParityBlock.kernel_eligible) and
verified on a T4 against this file's own loop: rel 1.3e-06 to 1.3e-05 across tied/untied x
n_read 1/2/4, with gradients reaching the tap (kaggle/assert).

    d=256, head_dim=128, L=128, B=64     fused 0.0208 s/fwd    loop 0.0819 s/fwd    3.9x

3.9x, NOT the 76x kaggle/kernel reported - that was measured at B=2, where the loop is almost
pure launch latency. At B=64 each launch does 32x more work and the loop amortises far better.

So, at ~0.0052 s per kernel call, a budget-8 arm is ~0.042 s forward and ~0.12 s per step; six
fused arms x 6000 steps is ~65 min PER DEPTH. And TWO ARMS DO NOT FUSE AT ALL:
  gdn ref   GDNRef is a separate class with its own python loop - ~60 min per depth on its own
  robust    robust_push modifies u inside the recurrence; the kernel never exposes u
gdn ref is therefore run at the FIRST DEPTH ONLY. It answers "is our untied baseline a strawman",
which does not need re-answering at every depth, and running it everywhere would be ~80% of the
bill. Total lands near 5.5 hours against a 9-hour session cap.

TEST P1 CHEAPLY FIRST - ~15 MINUTES INSTEAD OF 5.5 HOURS, and no code change:

    STEPS=2500 SEEDS=0 D=128 HEAD_DIM=64 NENT=32 DEPTHS=2,6 ARMS=ceiling,untied,tied python gpu_run.py

Four independent multipliers: drop gdn ref (does not fuse, ~80% of the bill), halve d to 128
(~4x less work), two depths instead of four, 2500 steps instead of 6000. Depths 2 and 6 straddle
NL2's layer count, which is precisely where P1 says the arms must separate.

WHY A SHORT RUN CAN STILL ANSWER IT: P1 predicts an ORDERING, not a converged accuracy - NL2 r1
(budget 2) collapsing at depth 6 while NL2 r4 and NL8 r1 (both budget 8) hold. Orderings appear
long before convergence and survive shrinking the model.

THE CATCH, NAMED SO IT DOES NOT BITE: at d=128 and 2500 steps the transformer ceiling will likely
NOT clear 0.90, so raw accuracy ranks nothing - that is what ceiling_sweep established. Read
POSITION RELATIVE TO THE SINK FLOOR instead (0.333 at d2, 0.143 at d6). NL2 r4 clearly above the
floor at depth 6 while NL2 r1 sits on it IS P1. Both on the floor probably kills P1 and saves
five hours.

The one outcome that is NOT a result: everything at the floor. That says the cell is too hard at
this scale, which is indistinguishable from the mechanism failing. If it happens, make the TASK
easier rather than the model bigger - NENT=16 with ratio=0.5 - so the ceiling clears sooner.

usage: python gpu_run.py        # env: STEPS SEEDS D HEAD_DIM NENT DEPTHS ARMS DMODE RATIO SEQPAD CEIL_MULT

AUDIT 2026-09-15 - what changed below, and why the v1-v4 logs must be read with it in mind:
  task      DMODE=chain by default; the v1-v4 'edge' task has a query-free shortcut at 1.000
  floor     FLOOR = max(sink guess, that shortcut), printed per depth
  seqlen    from the body length + SEQPAD, not 32+16*depth (~80% PAD)
  seeds     per-seed acc printed; every instrument pooled over seeds (lag_cos/decay were seed-last)
  lag       probe(): task batch, every tied layer, lag1 AND lag2, at init and end
  decay     probe(): the decay actually applied; head_decay dropped a_proj(z)
  swap      accuracy after swapping the query start node - does the model read the query?
  ceiling   gate reads the BEST transformer, not whichever was listed last
  wd        _no_weight_decay honoured, 1-D params exempt (tap, norms, A_log, dt_bias were decayed)
  kernel    fused==loop checked for every arm config, gates 1e-4 fp32 / 2e-2 autocast

SECOND REVIEW 2026-09-15 (Fable), also applied:
  kernel    checked at the run's own seqlens + L=100, not a fixed 64 (partial vs full chunk)
  ceiling   NaN-safe best_ceiling(); MODE=ceiling_triage tests the transformer on MQAR-4 first
  swap      adds `flip`, the paired statistic (new/old alone is weak at RATIO=1)
  probe     prints the tap and a no-lag background `bg`
  arms      tied NL2 r2, tied NL2 r1 3x (capacity vs speed), untied NL2 r1
bundle: parity_block.py recall_tasks.py shift_arms.py cpu_precheck.py

REFERENCES
  [Yang et al. 2024a] Parallelizing Linear Transformers with the Delta Rule over Sequence Length, arXiv 2406.06484
  [Gu & Dao 2023] Mamba: Linear-Time Sequence Modeling with Selective State Spaces, arXiv 2312.00752
  [Behrouz et al. 2025] It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization, arXiv 2504.13173
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from parity_block import ParityBlock, _FLA_CHUNK
from recall_tasks import vocab, make_chain, swap_query, heuristic_floors
from shift_arms import batch, Net as RefNet
from cpu_precheck import Net as ParityNet          # verified by cpu_precheck check 2b

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = DEV == 'cuda'


def assert_tied(net):
    """cpu_precheck check 6 against the ACTUAL model about to train. The one failure mode that
    leaves every other signal looking healthy. RANDOM ids, never zeros: with identical tokens
    every position shares an embedding, so a lag-1 shift of a constant sequence equals itself and
    this would pass for a map that is not tied at all."""
    #   EVERY tied block, not the first: a bug confined to layers >= 2 would pass a layer-0 check.
    worst = 0.0
    for li, blk in enumerate(net.mix):
        if not (isinstance(blk, ParityBlock) and blk.tied):
            continue
        tap = blk.tap.detach().clone()
        with torch.no_grad():
            blk.tap.copy_(torch.tensor([20., 0., 0., 0.], device=blk.tap.device, dtype=blk.tap.dtype))
        blk.debug = True
        with torch.no_grad():
            net(torch.randint(0, net.emb.num_embeddings, (2, 10), device=DEV))
        st = blk.trace['steps']
        err = max(float((st[t][0] - st[t - 1][1]).abs().max()) for t in range(1, len(st)))
        blk.debug, blk.trace = False, None
        with torch.no_grad():
            blk.tap.copy_(tap)
        assert err < 1e-4, 'TIED INVARIANT BROKEN in layer %d: %.3e' % (li, err)
        worst = max(worst, err)
    return worst


#   The configurations the arms ACTUALLY run. Only tied n_read=2 was checked before, while every
#   P1 run trained n_read 1 and 4; conv=False and untied had never been compared at all.
KERNEL_CHECKS = (dict(tied=True, n_read=1), dict(tied=True, n_read=2), dict(tied=True, n_read=4),
                 dict(tied=False, n_read=1), dict(tied=True, n_read=4, conv=False))


def assert_kernel_matches_loop(d, nent, lengths=(64,), **blk):
    """THE BLOCK'S FUSED PATH MUST EQUAL THE BLOCK'S LOOP PATH.

    kaggle/kernel verified fla's kernel against a hand-written reference at rel 3.773e-07. That is
    NOT the same as verifying that ParityBlock wires it in correctly - wrong g, a missed scale, a
    transposed head axis, or n_read applied once instead of n times would all still produce a
    plausible falling loss. That gap is exactly what let the tied map ship silently untied while
    every tensor-level check passed green, so it gets an assert rather than an assumption.

    Same weights, same input, kernel on and off. Returns the relative disagreement, or None when
    there is no kernel to compare against."""
    if DEV != 'cuda' or _FLA_CHUNK is None:
        return None
    _, _, _, V = vocab(nent)
    #   RUN IT UNDER AUTOCAST TOO, because that is what training uses. The first P1 run died on
    #   a Triton CompilationError - "Both operands must be same dtype. Got fp32 and fp16" - and
    #   no check caught it, because every kernel verification to that point ran in plain fp32
    #   under no_grad. Verifying the component outside the context it runs in is how the untied
    #   map and the unbound query both got through.
    #   AT THE LENGTHS TRAINING USES (second review, 2026-09-15). This was a fixed L=64 while v5
    #   trains at 15-23 - a single partial chunk in fla's chunked kernel, a different code path
    #   from a full one. `lengths` should be the run's seqlens plus one multi-chunk length.
    worst, fails = 0.0, []
    for cfg, L, amp_on in [(c, n, a) for c in KERNEL_CHECKS for n in lengths for a in (False, True)]:
        torch.manual_seed(0)
        net = ParityNet(V, d, 1, **dict(blk, **cfg)).to(DEV)
        x = torch.randint(0, V, (2, L), device=DEV)
        outs = {}
        for uk in (True, False):
            for m in net.mix:
                m.use_kernel = uk
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp_on,
                                                     dtype=torch.float16):
                outs[uk] = net(x).float()
        den = float(outs[False].abs().max())
        rel = float((outs[True] - outs[False]).abs().max()) / max(den, 1e-6)
        #   fp16 rounding alone moves the output, so autocast gets a looser gate than fp32. These
        #   were 2e-2 / 1e-1 - four orders above the measured fp32 agreement (1.3e-6 to 1.3e-5 on
        #   a T4) and ~100x above the measured autocast one (7.97e-4, v4) - loose enough to pass a
        #   missed scale factor. A failure here costs a minute at startup; a pass that should not
        #   have passed costs the run.
        gate = 1e-4 if not amp_on else 2e-2
        #   PRINT EVERY CONFIG, assert after. untied (raw v) and conv=False have never been
        #   measured on a T4, so a first failure should hand back all ten numbers, not one.
        print('  kernel vs loop  %-42s L=%-4d autocast=%-5s rel %.3e  gate %.0e  %s'
              % (cfg, L, amp_on, rel, gate, 'ok' if rel < gate else '*** FAIL ***'), flush=True)
        if rel >= gate:
            fails.append((cfg, L, amp_on, rel))
        worst = max(worst, rel)
    assert not fails, 'FUSED PATH DISAGREES WITH THE LOOP: %s' % fails
    return worst


def probe(net, x, pad):
    """Every per-head instrument, read in ONE traced forward pass over a TASK batch.

    lag1, lag2   cos(k_t, v_(t-1)) and cos(k_t, v_(t-2)), over EVERY tied layer, at positions whose
                 whole tap window is non-PAD. Replaces lag_cos, which an audit (2026-09-15) found
                 was mostly not measuring a transition operator: an UNTRAINED block with conv read
                 0.782, and one with its tap forced onto the WRONG lag still read 0.456 - SiLU
                 after the conv makes percepts mostly positive, so any two share ~0.465 cosine.
                 t=1 always added exactly 1.0 (only lag 1 exists there), and it read random ids
                 and the first layer only. What "S maps p_(t-1) -> p_t" asserts is the CONTRAST
                 lag1 - lag2, read against the same arm at init.
    dec          per-head decay AS APPLIED, averaged over non-PAD positions, taken from the
                 recurrence's own trace. The old head_decay computed exp(-exp(A_log) *
                 softplus(dt_bias)) and dropped a_proj(z), trained bias included, so every v1-v4
                 "dec slow/fast" figure was the bias-only decay (probe: reported 0.991/0.984
                 against an applied 0.942/0.882).

    The trace forces the python loop; the batch is small, so that is cheap."""
    blks = [m for m in net.mix if isinstance(m, ParityBlock)]
    if not blks:
        return dict(lag1=None, lag2=None, bg=None, tap=None, dec=np.array([]))
    x = x.to(DEV)
    live = x != pad
    L = x.shape[1]
    for b in blks:
        b.debug = True
    try:
        with torch.no_grad():
            net(x)
        n1 = n2 = nb = den = dbg = 0.0
        dec, taps = [], []
        for b in blks:
            st = b.trace['steps']
            D_ = torch.stack([s[3] for s in st], 1).float()                  # (B, L, H)
            lv = live.float().unsqueeze(-1)
            dec.append(((D_ * lv).sum((0, 1)) / lv.sum((0, 1)).clamp_min(1)).cpu().numpy())
            if not b.tied:
                continue
            K = torch.stack([s[0] for s in st], 1).float()                   # (B, L, H, d)
            P = torch.stack([s[1] for s in st], 1).float()                   # v = p when tied
            n = b.ntap
            c1 = F.cosine_similarity(K[:, n:], P[:, n - 1:L - 1], dim=-1)
            c2 = F.cosine_similarity(K[:, n:], P[:, n - 2:L - 2], dim=-1)
            m = live[:, n:].clone()
            for j in range(1, n + 1):
                m &= live[:, n - j:L - j]
            #   BACKGROUND: the same lag-1 cosine against ANOTHER sequence's percept at the same
            #   position (batch rolled by one). No temporal relation at all, so it measures pure
            #   percept similarity - the ~0.47 that SiLU puts under every raw lag reading with conv
            #   on. lag1 - bg is what the tap and S actually add.
            cb = F.cosine_similarity(K[:, n:], P.roll(1, dims=0)[:, n - 1:L - 1], dim=-1)
            mb = m & m.roll(1, dims=0)
            m = m.unsqueeze(-1).expand_as(c1).float()
            mb = mb.unsqueeze(-1).expand_as(cb).float()
            n1 += float((c1 * m).sum())
            n2 += float((c2 * m).sum())
            den += float(m.sum())
            nb += float((cb * mb).sum())
            dbg += float(mb.sum())
            #   the tap itself - the DIRECT instrument, which no table printed.
            taps.append(b.tap_weights().detach().float().cpu().numpy())
    finally:
        for b in blks:
            b.debug, b.trace = False, None
    tied = any(b.tied for b in blks)
    return dict(lag1=(n1 / den if den else float('nan')) if tied else None,
                lag2=(n2 / den if den else float('nan')) if tied else None,
                bg=(nb / dbg if dbg else float('nan')) if tied else None,
                tap=np.mean(taps, axis=0) if taps else None,
                dec=np.concatenate(dec))


def lag_cos(net):
    """LEGACY - the v1-v4 metric, kept so those logs (and eps_ablation) stay reproducible. Random
    ids, first tied layer only, t=1 included. Do not read it as evidence; see probe()."""
    blk = next((m for m in net.mix if isinstance(m, ParityBlock) and m.tied), None)
    if blk is None:
        return None
    blk.debug = True
    with torch.no_grad():
        net(torch.randint(0, net.emb.num_embeddings, (4, 16), device=DEV))
    st = blk.trace['steps']
    blk.debug, blk.trace = False, None
    return float(np.mean([float(F.cosine_similarity(st[i][0], st[i - 1][1], dim=-1).mean())
                          for i in range(1, len(st))]))


def head_decay(net):
    """LEGACY and WRONG: drops a_proj(z). Kept only to reproduce v1-v4 logs; use probe()['dec']."""
    out = []
    for m in net.mix:
        if isinstance(m, ParityBlock):
            g = -torch.exp(m.A_log) * F.softplus(m.dt_bias)
            out.append(torch.exp(g).detach().float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def bimodality(d):
    """P4's signature. Fraction slow, fraction fast, and the gap between the clusters - a decay
    distribution that has SPLIT is the mechanism specialising, and is more informative than acc."""
    if len(d) == 0:
        return '-'
    slow, fast = float((d >= 0.95).mean()), float((d <= 0.5).mean())
    return '%.0f%%/%.0f%% gap %.2f' % (100 * slow, 100 * fast, float(d.max() - d.min()))


def decay_str(d0, d1):
    """slow/fast percentages at init and at the end. Init MUST be shown: the Mamba-style init
    already spreads decays across ~0.2-1.0, so an end-of-run split means nothing without it."""
    if len(d0) == 0 or len(d1) == 0:
        return '-'
    f = lambda d: '%d/%d' % (round(100 * float((d >= 0.95).mean())), round(100 * float((d <= 0.5).mean())))
    return '%s>%s n=%d' % (f(d0), f(d1), len(d1))


def floors(nent, trials=2000, **cell):
    """(sink, shortcut) - expected accuracy of the two QUERY-FREE heuristics
    (recall_tasks.heuristic_floors). A score is evidence of composition only above the higher.

    sink: guess among nodes never seen as a source. Not 1/nent, and it MOVES WITH DEPTH under
    dmode='edge' (0.333/0.250/0.200/0.143 at depths 2/3/4/6), so a flat accuracy curve can be a
    model riding a moving floor. shortcut: the sink whose source is also a target - 1.000 at every
    depth under dmode='edge', which is why that form can no longer support a result."""
    sep, _, pad, _ = vocab(nent)
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(trials):
        r = make_chain(rng, nent=nent, **cell)
        if r is not None:
            hits.append(heuristic_floors(r[0], r[1], pad, sep))
    if not hits:
        return float('nan'), float('nan')
    sink, short = np.mean(hits, axis=0)
    return float(sink), float(short)


def _predict(net, seqs):
    X = torch.tensor(seqs, device=DEV)
    pred = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=AMP, dtype=torch.float16):
        for i in range(0, len(seqs), 128):
            pred += net(X[i:i + 128]).argmax(-1).tolist()
    return np.array(pred)


def swap_acc(net, nent, n=512, **cell):
    """(new, old, flip) after swapping the query's start node for another chain's root.

    new/old: accuracy on the swapped sample against its new target and against the original one.
    flip:    fraction of samples whose PREDICTION CHANGES when only the query changes - the paired
             statistic, added after a second review showed new/old alone is weak: at RATIO=1 the
             swapped sample is the exact mirror instance, so `new ~ acc` holds for ANY model,
             including one that ignores the query. A query-free model has flip = 0 by construction.
    None under dmode='edge', where no other root has a depth-hop walk."""
    sep, _, pad, _ = vocab(nent)
    rng = np.random.default_rng(4242)
    orig, xs, new, old = [], [], [], []
    for _ in range(20 * n):
        if len(xs) >= n:
            break
        r = make_chain(rng, nent=nent, **cell)
        sw = None if r is None else swap_query(rng, r[0], cell['depth'], pad, sep)
        if sw is not None:
            orig.append(r[0]); xs.append(sw[0]); new.append(sw[1]); old.append(r[1])
    if not xs:
        return None
    p0, p1 = _predict(net, orig), _predict(net, xs)
    return (float((p1 == np.array(new)).mean()), float((p1 == np.array(old)).mean()),
            float((p1 != p0).mean()))


def best_ceiling(ceilings):
    """(label, acc) of the best FINITE transformer, or None. max() over a dict holding NaN is
    order-dependent (NaN compares False both ways), so a diverged NL2 listed first could be
    'best' and silently suppress the below-0.90 warning."""
    fin = {k: v for k, v in ceilings.items() if np.isfinite(v)}
    if not fin:
        return None
    k = max(fin, key=fin.get)
    return k, fin[k]


def param_groups(net, wd):
    """AdamW groups that honour `_no_weight_decay`. parity_block set that flag on A_log and dt_bias
    and nothing read it, so both were decayed - as were the tap (pulled toward UNIFORM, the very
    quantity P3 measures), every norm weight and every bias. 1-D tensors are exempt too."""
    decay, keep = [], []
    for p in net.parameters():
        (keep if getattr(p, '_no_weight_decay', False) or p.ndim < 2 else decay).append(p)
    return [dict(params=decay, weight_decay=wd), dict(params=keep, weight_decay=0.0)]


def eval_acc(net, task, nent, n_batches, seed, **cell):
    """Accuracy over n_batches x 128 samples from a FIXED rng, so every arm is scored on the same
    sequences. Leaves the net in eval mode; callers that keep training must call net.train()."""
    net.eval()
    rng = np.random.default_rng(seed)
    corr = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=AMP, dtype=torch.float16):
        for _ in range(n_batches):
            x, y = batch(task, 128, rng, nent, **cell)
            corr += (net(x.to(DEV)).argmax(-1) == y.to(DEV)).tolist()
    return float(np.mean(corr))


def train(arm, nl, n_read, blk_kw, task, steps, d, nent, seed, lr=None, bs=64, wd=0.01,
          ceil_mult=None, curve_every=0, **cell):
    #   THE CEILING NEEDS ITS OWN LEARNING RATE, or it is not a ceiling.
    #   In v3 the transformer scored 0.150 and 0.107 - BELOW the sink floor - while our arms
    #   reached 1.000 on the same cells. The gate then announced "nothing at this depth ranks",
    #   which was wrong: our arms solving the task proves it is solvable and the harness sound,
    #   so the failure was the ceiling arm's, not the cell's. lr=3e-3 across 8 transformer layers
    #   was the suspected cause (our blocks tolerate it). CORRECTION, audit 2026-09-15: this said
    #   "with no warmup", which was false - OneCycleLR starts at lr/25 and warms up over the
    #   first 10% of steps. v4 ran WITH this lr fix and the transformer still sat below the floor,
    #   so lr was not the cause and the cause is still unknown: step budget (CEIL_MULT below),
    #   depth (the NL2 ceiling), ~80% PAD (now trimmed), or ~2 layers needed per hop.
    #   A baseline that was never given a fair chance cannot bound anything, and every "we beat
    #   the ceiling" reading taken against it would be worthless.
    if lr is None:
        lr = 1e-3 if arm == 'transformer' else 3e-3

    #   AND ITS OWN STEP BUDGET, for the same reason. ceiling_sweep measured a transformer needing
    #   6000 steps to solve mqar-2 - ONE hop, two pairs - and managing only 0.589 at 2000. We were
    #   giving it 3000 on a strictly harder task, so the ceiling's 0.229 / 0.163 / 0.087 is very
    #   likely a budget artefact. The asymmetry runs AGAINST us, which is the right direction: a
    #   ceiling exists to bound readability, not to be beaten, and a bound built from an
    #   undertrained model is worthless. It is also nearly free - the ceiling took ~90s for three
    #   seeds against 143-211s for our arms.
    if arm == 'transformer':
        steps = steps * (int(os.environ.get('CEIL_MULT', 3)) if ceil_mult is None else ceil_mult)
    torch.manual_seed(seed)
    _, _, _, V = vocab(nent)
    if arm in ('transformer', 'gdn'):
        net = RefNet(arm, V, d, nl, n_read=1).to(DEV)
    else:
        net = ParityNet(V, d, nl, tied=(arm == 'tied'), n_read=n_read, **blk_kw).to(DEV)
    npar = sum(p.numel() for p in net.parameters())
    tie_err = assert_tied(net) if arm == 'tied' else None
    _, _, pad, _ = vocab(nent)
    #   ONE fixed probe batch, read at init and at the end, so the two are on identical inputs.
    #   Its own rng, so the training stream is the same with or without instrumentation.
    xp, _ = batch(task, 16, np.random.default_rng(12345), nent, **cell)
    pr0 = probe(net, xp, pad)

    opt = torch.optim.AdamW(param_groups(net, wd), lr)
    #   pct_start floored at 2/steps: OneCycleLR divides by (warmup steps - 1), which is zero below
    #   20 total steps and raised ZeroDivisionError in a 10-step bundle smoke test. Identical
    #   schedule (0.1) for every run of 20+ steps.
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps,
                                              pct_start=max(0.1, 2.0 / steps))
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    rng = np.random.default_rng(seed)
    t0, last_loss, nonfin, skipped = time.time(), None, 0, 0
    curve = []                                   # (step, acc) every curve_every steps; 0 = off
    for i in range(steps):
        x, y = batch(task, bs, rng, nent, **cell)
        with torch.amp.autocast('cuda', enabled=AMP, dtype=torch.float16):
            loss = F.cross_entropy(net(x.to(DEV)), y.to(DEV))
        if not torch.isfinite(loss):
            nonfin += 1
            opt.zero_grad(set_to_none=True)
            if nonfin > 10:
                return dict(acc=float('nan'), params=npar, note='DIVERGED at %d' % i)
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        #   STEP THE SCHEDULER ONLY IF THE OPTIMIZER ACTUALLY STEPPED. GradScaler SKIPS the
        #   update when it finds inf/nan, and calling sch.step() anyway desynchronises the LR
        #   schedule from the weights - which is what produced the "lr_scheduler.step() before
        #   optimizer.step()" warning in the P1 run. Counting the skips is the more useful half:
        #   a nonzero `skipped` means fp16 is overflowing, and that is a signal about the run,
        #   not a cosmetic warning.
        _s = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() >= _s:
            sch.step()
        else:
            skipped += 1
        last_loss = loss.detach()                # a tensor: float() every step was a sync
        if curve_every and (i + 1) % curve_every == 0:
            curve.append((i + 1, eval_acc(net, task, nent, 2, 777, **cell)))
            net.train()

    acc = eval_acc(net, task, nent, 8, 9999, **cell)
    pr1 = probe(net, xp, pad)
    return dict(acc=acc, params=npar, secs=time.time() - t0, steps=steps, lr=lr, wd=wd,
                loss1=float(last_loss) if last_loss is not None else float('nan'),
                lag1_0=pr0['lag1'], lag2_0=pr0['lag2'], lag1_1=pr1['lag1'], lag2_1=pr1['lag2'],
                bg_0=pr0['bg'], bg_1=pr1['bg'], tap0=pr0['tap'], tap1=pr1['tap'], curve=curve,
                dec0=pr0['dec'], dec1=pr1['dec'],
                swap=swap_acc(net, nent, **cell) if task == 'chain' else None,
                tie_err=tie_err, nonfin=nonfin, skipped=skipped,
                #   surfaced through `note` so it actually reaches the table. A counter nothing
                #   prints is dead instrumentation - the same failure as an assert nothing calls.
                #   Nonzero means GradScaler found inf/nan and skipped optimizer steps, i.e. fp16
                #   is overflowing: a fact about the run, not a warning to ignore.
                note=('fp16 SKIPPED %d/%d steps' % (skipped, steps)) if skipped else '')


#   label, arm, NL, n_read, extra block kwargs.  BUDGET = NL * n_read.
ARMS = [
    #   TWO CEILINGS, to separate depth from budget. v4's NL8 transformer sat below the sink
    #   floor at every depth even after the lr fix, and v3 had already shown 8 layers do not train
    #   here in 3000 steps - which is why tied NL8 r1 was dropped. Leaving an NL8 ceiling in place
    #   meant comparing our 2-layer arms against a baseline disqualified for exactly that reason.
    #   NL2 matches the arms under test; if it clears while NL8 does not, the ceiling's failure
    #   was depth, and if neither clears with 3x steps it is something else again.
    ('ceiling  NL2',        'transformer', 2, 1, {}),
    ('ceiling  NL8',        'transformer', 8, 1, {}),
    #   A REAL GatedDeltaNet, not our untied block wearing its clothes. "Tied beats untied" is
    #   worth nothing if our own untied arm is accidentally a strawman, and the two share every
    #   line of ParityBlock - so a bug that cripples one cripples both and cancels out of the
    #   comparison invisibly. This is the outside check on the baseline. NOTE it is NOT parameter-
    #   matched: RefNet uses H=4 (head_dim 64 at d=256) where our blocks use head_dim 128. It is a
    #   REFERENCE for "is our baseline sane", not an arm in the layers-vs-reads trade.
    ('gdn ref  NL8',        'gdn',         8, 1, {}),
    ('untied   NL8 r1',     'untied',      8, 1, {}),            # budget 8, incumbent shape
    ('tied     NL8 r1',     'tied',        8, 1, {}),            # budget 8
    ('tied     NL4 r2',     'tied',        4, 2, {}),            # budget 8, ~half params
    ('tied     NL2 r4',     'tied',        2, 4, {}),            # budget 8, ~quarter params
    ('tied     NL2 r1',     'tied',        2, 1, {}),            # budget 2 - control
    ('tied     NL2 r4 nc',  'tied',        2, 4, dict(conv=False)),   # P3 corollary
    #   Added after the second review (2026-09-15). r2 fills the n_read axis between 1 and 4.
    #   `step_mult` is popped by the runner, never passed to the block: r1 at 3x steps separates
    #   CAPACITY from OPTIMISATION SPEED - if it catches r4, reads only speed learning up and
    #   "reads substitute for layers" does not follow. untied NL2 r1 is the incumbent at this size.
    ('tied     NL2 r2',     'tied',        2, 2, {}),            # budget 4
    ('tied     NL2 r1 3x',  'tied',        2, 1, dict(step_mult=3)),  # budget 2, 3x steps
    ('untied   NL2 r1',     'untied',      2, 1, {}),            # budget 2, no tying
]


def ceiling_triage():
    """MODE=ceiling_triage - can the transformer baseline do SINGLE-HOP recall at all?

    A second review (2026-09-15, CPU, d=64) ran this exact train(): transformer NL2 on MQAR with
    4 pairs sat at 0.166 after 3000 steps - below the 0.25 chance among the values - while tied
    NL2 r1 reached 0.811 in 1000 steps on the same cell, and neither d=128 nor NL4 rescued it
    (MQAR-2 was solved, 0.997). Every v3/v4 'ceiling' row was therefore a baseline that cannot do
    one hop, and nothing should be spent on NL8 until NL2 clears this cell.

    Grid: NL x lr x wd, a learning curve per config, plus the tied block on the SAME cell as a
    HARNESS CONTROL - if the control fails as well, the harness is broken, not the transformer."""
    STEPS = int(os.environ.get('STEPS', 30000))
    SEEDS = [int(s) for s in os.environ.get('SEEDS', '0').split(',')]
    D = int(os.environ.get('D', 128))
    HEAD_DIM = int(os.environ.get('HEAD_DIM', 64))
    NENT = int(os.environ.get('NENT', 32))
    NPAIRS = int(os.environ.get('NPAIRS', 4))
    SEQPAD = int(os.environ.get('SEQPAD', 4))
    LRS = [float(v) for v in os.environ.get('TRIAGE_LRS', '3e-4,1e-3,3e-3').split(',')]
    WDS = [float(v) for v in os.environ.get('TRIAGE_WDS', '0,0.01').split(',')]
    NLS = [int(v) for v in os.environ.get('TRIAGE_NL', '2').split(',')]
    CTRL = int(os.environ.get('CTRL_STEPS', 1000))
    every = int(os.environ.get('CURVE_EVERY', 0)) or max(1, STEPS // 10)
    cell = dict(npairs=NPAIRS, seqlen=2 * NPAIRS + 2 + SEQPAD)
    print('CEILING TRIAGE  dev=%s d=%d steps=%d seeds=%s  mqar npairs=%d seqlen=%d  nl=%s lrs=%s wds=%s'
          % (DEV, D, STEPS, SEEDS, NPAIRS, cell['seqlen'], NLS, LRS, WDS), flush=True)
    print('  chance among the %d values %.3f; a usable ceiling must clear 0.90' % (NPAIRS, 1.0 / NPAIRS))

    base = dict(head_dim=HEAD_DIM, channel_gates=False, tap_init='uniform', robust_eps=1e-4,
                Delta=0.3, bias='l2')
    r = train('tied', 2, 1, base, 'mqar', CTRL, D, NENT, SEEDS[0], **cell)
    print('  HARNESS CONTROL  tied NL2 r1, %d steps: acc %.3f  (%.0fs)  %s'
          % (CTRL, r['acc'], r.get('secs', 0), 'the harness trains this cell' if r['acc'] >= 0.5
             else '*** CONTROL FAILED TOO - suspect the harness, not the transformer ***'), flush=True)

    print('  %-3s %-7s %-6s %-26s %-6s %s' % ('NL', 'lr', 'wd', 'final acc [per seed]', 'secs',
                                              'curve, seed %d (step:acc)' % SEEDS[0]))
    best = None
    for nl in NLS:
        for lr in LRS:
            for wd in WDS:
                runs = [train('transformer', nl, 1, {}, 'mqar', STEPS, D, NENT, sd, lr=lr, wd=wd,
                              ceil_mult=1, curve_every=every, **cell) for sd in SEEDS]
                accs = [x['acc'] for x in runs]
                fin = [a for a in accs if np.isfinite(a)]
                m = float(np.mean(fin)) if fin else float('nan')
                crv = runs[0].get('curve', [])
                print('  %-3d %-7.0e %-6g %-26s %-6.0f %s%s'
                      % (nl, lr, wd, '%.3f [%s]' % (m, ' '.join('%.2f' % a for a in accs)),
                         sum(x.get('secs', 0) for x in runs),
                         ' '.join('%d:%.2f' % sa for sa in crv),
                         ''.join('  ' + x['note'] for x in runs if x.get('note'))), flush=True)
                if np.isfinite(m) and (best is None or m > best[0]):
                    hit = next((s for s, a in crv if a >= 0.9), None)
                    best = (m, nl, lr, wd, hit)
    print()
    if r['acc'] < 0.5:
        print('VERDICT: INCONCLUSIVE - the harness control (tied NL2 r1, %d steps) scored %.3f, so a'
              % (CTRL, r['acc']))
        print('  transformer failure here cannot be attributed to the transformer. Raise CTRL_STEPS.')
    elif best is None:
        print('VERDICT: every config diverged.')
    elif best[0] >= 0.9:
        print('VERDICT: NL%d CLEARS mqar-%d at lr=%g wd=%g (%.3f; curve first >= 0.90 at step %s).'
              % (best[1], NPAIRS, best[2], best[3], best[0], best[4]))
        print('  Use that lr/wd and at least that many steps for every ceiling arm in v5.')
    else:
        print('VERDICT: NO config clears mqar-%d (best %.3f: NL%d lr=%g wd=%g). The transformer arm is'
              % (NPAIRS, best[0], best[1], best[2], best[3]))
        print('  impaired beyond lr / wd / step budget. Do not use it as a ceiling; inspect Attn before v5.')

if __name__ == '__main__':
    if os.environ.get('MODE') == 'ceiling_triage':
        ceiling_triage()
        raise SystemExit(0)
    STEPS = int(os.environ.get('STEPS', 6000))
    SEEDS = [int(s) for s in os.environ.get('SEEDS', '0').split(',')]
    D = int(os.environ.get('D', 256))
    HEAD_DIM = int(os.environ.get('HEAD_DIM', 128))
    NENT = int(os.environ.get('NENT', 64))
    DEPTHS = [int(x) for x in os.environ.get('DEPTHS', '2,3,4,6').split(',')]
    WANT = os.environ.get('ARMS', '')
    #   dmode='chain' is the default because 'edge' has a query-free shortcut scoring 1.000 at
    #   every depth (recall_tasks.make_chain). DMODE=edge reproduces v1-v4's task.
    DMODE = os.environ.get('DMODE', 'chain')
    RATIO = float(os.environ.get('RATIO', 1.0))
    SEQPAD = int(os.environ.get('SEQPAD', 4))

    base = dict(head_dim=HEAD_DIM, channel_gates=False, tap_init='uniform',
                robust_eps=1e-4, Delta=0.3, bias='l2')
    #   Match EITHER the first word ('tied' selects every tied arm) OR the whole label with
    #   whitespace collapsed ('tied_NL2_r4' selects exactly one). The first form alone could not
    #   express "these three tied arms and no others", which is the difference between a 26-minute
    #   job and a 50-minute one when seeds are what the budget is being spent on.
    def _key(lab):
        return '_'.join(lab.split())

    want = [w.strip() for w in WANT.split(',') if w.strip()]
    arms = [a for a in ARMS
            if not want or a[0].split()[0] in want or _key(a[0]) in want]
    if want:
        missing = [w for w in want
                   if w not in [x[0].split()[0] for x in ARMS] + [_key(x[0]) for x in ARMS]]
        assert not missing, 'ARMS names nothing: %s (labels: %s)' % (
            missing, [_key(x[0]) for x in ARMS])

    def cell_for(depth):
        #   SEQUENCE LENGTH FROM THE BODY. The body is 2*(depth + distractor edges) + 3 tokens -
        #   11/15/19 at depths 2/3/4, ratio 1 - and the old 32+16*depth padded that to 64/80/96,
        #   ~80% PAD that every arm processed and the transformer attended over. SEQPAD is slack.
        ndis = int(round(depth * RATIO))
        return dict(depth=depth, ratio=RATIO, dmode=DMODE, seqlen=2 * (depth + ndis) + 3 + SEQPAD)

    #   RUN THE FUSED-vs-LOOP ASSERT BEFORE ANY TRAINING. An assert that is never called is worse
    #   than no assert, because it reads as coverage in the file and provides none. At every
    #   seqlen this run trains at, plus 100 so a multi-chunk path is covered too.
    krel = assert_kernel_matches_loop(D, NENT, sorted({cell_for(d)['seqlen'] for d in DEPTHS} | {100}),
                                      head_dim=HEAD_DIM, channel_gates=False,
                                      tap_init='uniform', bias='l2')
    print('GPU RUN  dev=%s d=%d head_dim=%d steps=%d seeds=%s depths=%s dmode=%s ratio=%s '
          'seqpad=%d ceil_mult=%s' % (DEV, D, HEAD_DIM, STEPS, SEEDS, DEPTHS, DMODE, RATIO, SEQPAD,
                                      os.environ.get('CEIL_MULT', 3)))
    try:
        import fla
        print('fla %s' % getattr(fla, '__version__', '(no __version__)'))
    except Exception:       # noqa: BLE001
        pass
    if krel is None:
        print('fused path: UNAVAILABLE (no fla or no cuda) - every arm runs the python loop')
    else:
        print('fused path: ACTIVE, matches the loop across %d configs (tied r1/r2/r4, untied, '
              'no-conv), worst rel %.2e. NOTE bias=robust CANNOT be fused' % (len(KERNEL_CHECKS), krel))
        print('  (robust_push modifies u INSIDE the recurrence and the kernel never exposes u),')
        print('  so any robust arm silently falls back to the loop and runs ~76x slower.')
    print('LAYERS-vs-READS at equal hop budget. Depth is the AXIS; the signal is the TREND.')
    print('Params are NOT matched - a win with fewer is unimpeachable, a loss could be budget.\n')

    def _mean(runs, key):
        v = [r[key] for r in runs if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(v)) if v else None

    def _pair(a, b, fmt='%.2f>%.2f'):
        return '-' if a is None or b is None else fmt % (a, b)

    def _tap(runs):
        t0 = [r['tap0'] for r in runs if r.get('tap0') is not None]
        t1 = [r['tap1'] for r in runs if r.get('tap1') is not None]
        if not t0 or not t1:
            return '-'
        f = lambda t: '[' + ' '.join('%.2f' % w for w in t) + ']'
        return '%s>%s' % (f(np.mean(t0, axis=0)), f(np.mean(t1, axis=0)))

    table, floor_by = {}, {}
    for depth in DEPTHS:
        cell = cell_for(depth)
        sink, short = floors(NENT, **cell)
        floor_by[depth] = max(sink, short)
        print('=== chain depth %d  (dmode=%s ratio=%s seqlen=%d)   FLOOR %.3f  (sink %.3f, '
              'query-free shortcut %.3f)  chance %.3f ==='
              % (depth, DMODE, RATIO, cell['seqlen'], floor_by[depth], sink, short, 1.0 / NENT),
              flush=True)
        if short > sink + 1e-9:
            print('  *** THIS CELL LEAKS: a rule that never reads the query scores %.3f. No score here '
                  'is evidence of composition. ***' % short, flush=True)
        print('  anything at or below FLOOR is a query-free heuristic, not the model', flush=True)
        print('  %-20s %-6s %-30s %-9s %-17s %s'
              % ('arm', 'budget', 'acc mean [per seed]', 'params', 'swap new/old/flip', 'secs'))
        print('      then, for recurrent arms, init>end: lag1 lag2 bg(no-lag background) '
              'tap[lag1..4] dec(slow/fast)')
        ceilings = {}
        for label, arm, nl, nr, extra in arms:
            #   gdn ref does not fuse (its own python loop) and costs ~60 min per depth, which is
            #   most of the bill. It is a sanity check on our baseline, not an arm in the trade,
            #   so one depth answers it.
            if arm == 'gdn' and depth != DEPTHS[0]:
                continue
            ex = dict(extra)
            smult = ex.pop('step_mult', 1)           # a runner setting, not a block kwarg
            runs = [train(arm, nl, nr, dict(base, **ex), 'chain', STEPS * smult, D, NENT, sd, **cell)
                    for sd in SEEDS]
            #   EVERY instrument is pooled over seeds. acc used to be the mean while lag_cos and
            #   decay came from the LAST seed only, so v4's "lag_cos corroborates" was one seed.
            accs = [r['acc'] for r in runs]
            table[(depth, label)] = accs
            acc = float(np.mean(accs))
            if arm == 'transformer':
                ceilings[label] = acc
            sw = [r['swap'] for r in runs if r.get('swap') is not None]
            swap = '%.2f/%.2f/%.2f' % tuple(np.mean(sw, axis=0)) if sw else '-'
            d0 = np.concatenate([r['dec0'] for r in runs if 'dec0' in r] or [np.array([])])
            d1 = np.concatenate([r['dec1'] for r in runs if 'dec1' in r] or [np.array([])])
            skipped = sum(r.get('skipped', 0) for r in runs)
            notes = [r['note'] for r in runs if r.get('note', '').startswith('DIVERGED')]
            if skipped:
                notes.append('fp16 SKIPPED %d/%d steps' % (skipped, sum(r.get('steps', 0) for r in runs)))
            print('  %-20s %-6d %-30s %-9d %-17s %.0f%s'
                  % (label, nl * nr,
                     '%.3f [%s]' % (acc, ' '.join('%.2f' % a for a in accs)),
                     runs[0].get('params', 0), swap, sum(r.get('secs', 0) for r in runs),
                     ('  ' + '; '.join(notes)) if notes else ''), flush=True)
            if len(d1) or _mean(runs, 'lag1_1') is not None:
                print('      lag1 %s  lag2 %s  bg %s  tap %s  dec %s'
                      % (_pair(_mean(runs, 'lag1_0'), _mean(runs, 'lag1_1')),
                         _pair(_mean(runs, 'lag2_0'), _mean(runs, 'lag2_1')),
                         _pair(_mean(runs, 'bg_0'), _mean(runs, 'bg_1')),
                         _tap(runs), decay_str(d0, d1)), flush=True)
        #   THE BEST FINITE transformer, not the last one listed. With two ceiling arms this read
        #   NL8 - the one already known to fail - every time.
        bc = best_ceiling(ceilings)
        if ceilings and bc is None:
            print('  *** every transformer baseline DIVERGED at this depth ***')
        elif bc is not None and bc[1] <= 0.90:
            print('  *** best transformer (%s) %.3f is below 0.90: a baseline that did not solve '
                  'this cell, NOT an upper bound. Read the arms against FLOOR. ***'
                  % (' '.join(bc[0].split()), bc[1]))
        print(flush=True)

    print('THE TREND, which is the actual result  (mean, min-max over seeds)')
    print('  %-20s %s' % ('FLOOR', '  '.join('%-18.3f' % floor_by[d] for d in DEPTHS)))
    print('  %-20s %s' % ('arm', '  '.join('d=%-16d' % d for d in DEPTHS)))
    for label, arm, nl, nr, extra in arms:
        #   'not run' rather than nan: gdn ref is deliberately restricted to the first depth, and
        #   a nan in the results table reads as a DIVERGED arm. This table is what the run gets
        #   read off, so an absent cell must not look like a failed one.
        cells = []
        for d in DEPTHS:
            a = table.get((d, label))
            cells.append('%-18s' % ('not run' if a is None else
                                    '%.3f (%.2f-%.2f)' % (np.mean(a), min(a), max(a))))
        print('  %-20s %s' % (label, '  '.join(cells)), flush=True)
    print()
    print('READING IT')
    print('  FLOOR is the better of two query-free heuristics. Under dmode=edge the shortcut is')
    print('      1.000 at every depth, so nothing on those cells is composition.')
    print('  P1b: NL2 r4 minus NL2 r1 - equal params, equal layers, only n_read differs - widening')
    print('      with depth. Seeds once spanned 0.074-0.496 on this task: a gap inside the per-seed')
    print('      spread is not a gap.')
    print('  CEILINGS are transformer BASELINES, not upper bounds. NL2 is depth-matched to the arms.')
    print('  swap new/old/flip: accuracy after swapping the query start node, against the new and the')
    print('      old target, and the fraction of predictions that changed. A query-blind model has')
    print('      flip = 0. An untrained net flips 30-50% at random, so flip alone proves nothing:')
    print('      query reading is new ~ acc WITH old near 0.')
    print('  bg: lag-1 cosine against another sequence - pure percept similarity. lag1 - bg is what')
    print('      the tap and S add; tap[...] is the learned lag weighting itself.')
    print('  NL2 r1 3x: if it catches NL2 r4, reads buy optimisation speed, not capacity.')
    print('  lag1/lag2: read the CHANGE from init and against bg. With conv on, the raw values carry')
    print('      a ~0.47 background from positive percepts; conv and nc do not compare.')
    print('  P4 needs many heads; at d=128 head_dim=64 NL2 there are 4 in total (n= shows the count).')
