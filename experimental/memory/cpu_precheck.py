"""CPU PRE-CHECK FOR THE GPU RUN. This is a CODE CHECK. It cannot support any claim.

The ceiling sweep already established that no CPU cell discriminates the arms (only mqar-2 clears
0.90, and every arm solves it), so nothing here is evidence about the mechanism. What it IS for:
GPU quota is scarce, and a run that dies at hour three on a NaN, or that
compares two things differing in more than one place, wastes it.

FIVE CHECKS, in order of what they would cost if skipped:

  1  Delta=0 degenerates to l2 exactly.  WIRING ONLY. This check passes even with the sign of the
     robust term inverted, because the term is multiplied by zero. On its own it proves nothing;
     it is here to catch a flag that is not plumbed through at all.
  2  Delta=0.3 against bias_fair.grad in numpy, float64, over several steps. THIS is the check
     that the algorithm implemented is the algorithm intended. It calls parity_block.robust_push
     directly - the shipped function, not a retyped copy of it, so it cannot pass by sharing a
     mistake with the thing it verifies.
  3  Autograd through the normalisation, at and near ||u|| = 0. The robust term divides by the
     update's own norm, and the state is DESIGNED to drive that norm to zero (it is the residual;
     fitting a value exactly is success, not failure). So the degenerate case is reached on
     purpose, mid-training, at the moment the memory works. Reported empirically across dtypes
     rather than asserted - I do not know in advance which of forward or backward breaks.
  4  Parameter counts identical between l2 and robust. Delta is a constant; any difference means
     something unintended was added.
  6  THE TIED MAP IS ACTUALLY TIED. Added after checks 1-5 all passed green on a block whose
     tied arm was silently untied by two separate bugs (a second short conv on the key side, and
     a normalisation applied to k but not v). Every check above verifies the UPDATE RULE and none
     of them looks at where k and v come from, so all five passed regardless. With a lag-1 tap
     the key at t must EQUAL the value at t-1; the untied arm is the control and must fail it.

  5  The 2x2 {l2, robust} x {untied, tied} trains: finite, loss falling, ||S|| not exploding.
     A 2x2 rather than "ours vs theirs" because ours-vs-theirs differs in TWO places at once and
     the resulting number cannot be attributed to either.

     AND THE TIED AXIS HERE IS ITSELF NOT CLEAN - the same mistake one level down. It carries
     n_read=2 while untied carries n_read=1, so that column moves the map AND the read count
     together. On mqar npairs=2, a one-hop lookup, a second read is pure handicap, which is the
     likely reason tied's loss barely fell (3.64->2.98) next to untied's (3.86->1.83). Nothing
     in check 5 is a claim so this does not invalidate it, but THE GPU SCRIPT MUST NOT INHERIT
     IT: hold n_read constant across all four cells, or make it a third axis.
     Two further asymmetries to state rather than hide:
       - tied carries fewer params than untied (one shared map replaces two, and with the tied
         fix there is no key-side conv either). Inherent to the mechanism, not a bug, but
         one-directional for inference: a tied LOSS could be the smaller budget, a tied WIN
         cannot be. check 5 now PRINTS the gap rather than asserting a number that goes stale.
       - tap_init='shift' starts the tap at the answer, which is fine for a code check and
         disqualifying for a result (smoke_corrected.py's second bug). The GPU script must use
         'uniform' anywhere the lag is meant to be discovered.

CHANNEL GATES ARE OFF throughout. MIRAS Eq 17 has no gates, and with them on the residual is no
longer a clean Sk-v, so `robust` would not be MIRAS's objective and check 2 could not be written.

usage: cd experiments && python cpu_precheck.py

REFERENCES
  [Behrouz et al. 2025] It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization, arXiv 2504.13173
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bias_fair import grad as np_grad
from parity_block import ParityBlock, robust_push
from shift_arms import SwiGLU, batch
from recall_tasks import vocab

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

DELTA = 0.3
OK, BAD = 'OK', '*** FAIL ***'


def line(n, title):
    print('\n%d. %s' % (n, title))


# ------------------------------------------------------------------ 1  wiring
def check1():
    line(1, 'Delta=0 -> plain delta rule   [WIRING ONLY - passes even with the sign wrong]')
    torch.manual_seed(0)
    B, H, dv = 2, 2, 8
    u = torch.randn(B, H, dv, dtype=torch.float64)
    eta = torch.rand(B, H, 1, dtype=torch.float64)
    err = float((robust_push(u, eta, 0.0, 1e-12) - u).abs().max())
    print('   max err %.3e   %s' % (err, OK if err < 1e-15 else BAD))
    return err < 1e-15


# ------------------------------------------------------------------ 2  the real one
def check2():
    line(2, 'Delta=%.1f vs bias_fair.grad (numpy, float64, 8 steps)   [ALGORITHM CHECK]' % DELTA)
    d, n = 16, 8
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(n + 1, d))
    phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    etas = rng.uniform(0.2, 0.9, size=n)

    S_np = np.zeros((d, d))
    for t in range(n):
        e = S_np @ phi[t] - phi[t + 1]
        S_np = S_np - etas[t] * np.outer(np_grad(e, 'robust', Delta=DELTA), phi[t])

    def torch_side(eps):
        S_t = torch.zeros(1, 1, d, d, dtype=torch.float64)
        for t in range(n):
            k = torch.tensor(phi[t], dtype=torch.float64).view(1, 1, d)
            v = torch.tensor(phi[t + 1], dtype=torch.float64).view(1, 1, d)
            eta = torch.full((1, 1, 1), float(etas[t]), dtype=torch.float64)
            u = eta * v - torch.einsum('bhvk,bhk->bhv', S_t, eta * k)   # = -eta*e, gates off
            u = robust_push(u, eta, DELTA, eps)
            S_t = S_t + u.unsqueeze(-1) * k.unsqueeze(-2)
        return np.abs(S_t.numpy().reshape(d, d) - S_np).max()

    #   AT eps=0 THE TWO ARE THE SAME ALGORITHM and must agree to round-off. The residual is
    #   bias_fair.grad's own hardcoded 1e-12, which is why the gate is 1e-10 and not 1e-15.
    err = float(torch_side(0.0))
    print('   eps=0  max |S_torch - S_numpy| = %.3e   %s' % (err, OK if err < 1e-10 else BAD))
    if err >= 1e-10:
        print('   the robust term is NOT MIRAS Eq 17 as implemented - do not launch')
    #   At the shipped eps the denominators differ by O(eps) BY DESIGN (||u||=eta*||e||, and eps
    #   is deliberately not scaled by eta - see robust_push). Reported, not gated.
    print('   eps=1e-4 deviation = %.3e   (expected O(eps): a numerical floor, not the objective)'
          % float(torch_side(1e-4)))
    return err < 1e-10


# ------------------------------------------------------------------ 2b  end-to-end
def check2b():
    """Checks 1 and 2 test robust_push IN ISOLATION. Neither shows that ParityBlock.forward calls
    it, calls it with the right eta, or uses what it returns. This replays the block's OWN
    per-step tensors (captured via debug=True) through an independent numpy implementation of the
    recurrence and compares the final state.

    Writing that numpy implementation by re-reading forward() caught a detail a retyped test
    would plausibly get wrong: the rank-1 commit is u x kt, the RAW key, while the erase uses
    ke = bg*kt*bt. With gates off those differ by the factor bt, so getting it wrong would still
    produce a plausible falling loss and never be noticed.

    SCOPE, and it was a coverage gap until an audit caught it: this replays the RECURRENCE given
    whatever k and v arrive - it takes them from the trace and does not re-derive them. It
    therefore says NOTHING about whether the tied construction builds k and v correctly; that is
    check6's job and only check6's. The tied arm is nonetheless covered HERE too, because the
    recurrence could in principle interact with the tied stream (the robust push divides by an
    update norm, and tied k/v are unit-norm where untied v is raw). Reading "the block was
    replayed through independent numpy" as covering tying would be wrong, so both are run and
    both are labelled."""
    print('\n2b. ParityBlock.forward replayed through independent numpy   [END-TO-END]')
    ok = True
    for tied in (False, True):
        for bias in ('l2', 'robust'):
            torch.manual_seed(0)
            blk = ParityBlock(64, head_dim=32, channel_gates=False, tied=tied, bias=bias,
                              Delta=DELTA, robust_eps=1e-12).double()
            blk.debug = True
            torch.manual_seed(1)
            x = torch.randn(2, 12, 64, dtype=torch.float64)
            with torch.no_grad():
                blk(x)
            steps, S_ref = blk.trace['steps'], blk.trace['S'].numpy()

            B, H, dv, dk = S_ref.shape
            S = np.zeros((B, H, dv, dk))
            for kt, vt, bt, dc in steps:
                kt, vt, bt, dc = kt.numpy(), vt.numpy(), bt.numpy(), dc.numpy()
                S = dc[:, :, None, None] * S
                ke, vw = kt * bt, vt * bt
                u = vw - np.einsum('bhvk,bhk->bhv', S, ke)
                if bias == 'robust':
                    un = np.linalg.norm(u, axis=-1, keepdims=True)
                    u = u + bt * DELTA * u / (un + 1e-12)
                S = S + u[:, :, :, None] * kt[:, :, None, :]      # RAW key, not the gated one
            err = float(np.abs(S - S_ref).max())
            ok &= err < 1e-11
            print('   %-6s bias=%-7s %d steps  max |S_block - S_numpy| = %.3e   %s'
                  % ('tied' if tied else 'untied', bias, len(steps), err,
                     OK if err < 1e-11 else BAD))

    #   the correction must be LIVE end-to-end: computed AND used.
    outs = {}
    for D_ in (0.0, DELTA):
        for bias in ('l2', 'robust'):
            torch.manual_seed(0)
            blk = ParityBlock(64, head_dim=32, channel_gates=False, tied=False,
                              bias=bias, Delta=D_, robust_eps=1e-12).double()
            torch.manual_seed(1)
            with torch.no_grad():
                outs[(D_, bias)] = blk(torch.randn(2, 12, 64, dtype=torch.float64))
    d0 = float((outs[(0.0, 'robust')] - outs[(0.0, 'l2')]).abs().max())
    dD = float((outs[(DELTA, 'robust')] - outs[(DELTA, 'l2')]).abs().max())
    print('   Delta=0   block output vs l2: %.3e   %s (wiring)' % (d0, OK if d0 == 0.0 else BAD))
    print('   Delta=%.1f block output vs l2: %.3e   %s (correction is USED, not discarded)'
          % (DELTA, dD, OK if dD > 1e-6 else BAD))
    return ok and d0 == 0.0 and dD > 1e-6


# ------------------------------------------------------------------ 6  the tying invariant
def check6():
    """THE REGRESSION TEST FOR THE BUG THE OTHER FIVE CHECKS ALL MISSED.

    Checks 1-5 verify the UPDATE RULE. Not one of them looks at whether k and v come from the
    same map, because every one of them passes just as well when they do not: shapes are
    unchanged, nothing raises, params barely move, and the loss still falls. The tied arm can be
    completely untied and every gate stays green.

    The invariant: with a pure lag-1 tap, the key at step t IS the value at step t-1. Not close -
    equal, because it is the same tensor read at an offset. If that fails, S maps one map's
    output into another map's input, S.S is meaningless, and the whole reason for tying is gone.

    Reads k and v from the block's OWN captured trace, so it cannot pass by sharing a derivation
    with forward(). The untied arm is the control: it must FAIL this, or the test is vacuous."""
    print('\n6. tied invariant: with a lag-1 tap, k_t == v_(t-1)   [CATCHES SILENT UNTYING]')
    out = {}
    for tied in (True, False):
        torch.manual_seed(0)
        blk = ParityBlock(64, head_dim=32, channel_gates=False, tied=tied,
                          bias='l2', n_read=1).double()
        if tied:
            with torch.no_grad():                      # force the tap to a pure lag of 1
                blk.tap.copy_(torch.tensor([20.0, 0.0, 0.0, 0.0], dtype=torch.float64))
            w = blk.tap_weights()
            assert float(w[0]) > 0.999, 'tap not sharp: %s' % w
        blk.debug = True
        torch.manual_seed(1)
        with torch.no_grad():
            blk(torch.randn(2, 10, 64, dtype=torch.float64))
        steps = blk.trace['steps']
        err = max(float((steps[t][0] - steps[t - 1][1]).abs().max())
                  for t in range(1, len(steps)))
        out[tied] = err
        expect = 'must be ~0' if tied else 'must be LARGE (control)'
        good = (err < 1e-10) if tied else (err > 1e-3)
        print('   tied=%-6s max |k_t - v_(t-1)| = %.3e   %-24s %s'
              % (tied, err, expect, OK if good else BAD))
    return out[True] < 1e-10 and out[False] > 1e-3


# ------------------------------------------------------------------ 3  the degenerate case
def check3():
    line(3, 'autograd at and near ||u|| = 0   [the state is DESIGNED to reach this]')
    print('   %-10s %-12s %-10s  %-10s  %s'
          % ('dtype', '||u||', 'eps', 'fwd finite', 'bwd finite'))
    rows = []
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        for scale in (1.0, 1e-4, 0.0):
            for eps in (1e-12, 1e-4):
                u0 = torch.randn(1, 1, 8, dtype=torch.float32) * scale
                u = u0.to(dt).requires_grad_(True)
                eta = torch.full((1, 1, 1), 0.5, dtype=dt)
                try:
                    y = robust_push(u, eta, DELTA, eps)
                    f = bool(torch.isfinite(y).all())
                    y.float().sum().backward()
                    b = bool(torch.isfinite(u.grad).all())
                except Exception as ex:                       # noqa: BLE001
                    f, b = False, False
                    print('   raised: %s' % ex)
                rows.append((dt, scale, eps, f, b))
                print('   %-10s %-12.0e %-10.0e  %-10s  %s'
                      % (str(dt).replace('torch.', ''), scale, eps,
                         'yes' if f else 'NO', 'yes' if b else 'NO'), flush=True)
    bad = [r for r in rows if not (r[3] and r[4])]
    print('   -> %d of %d cells non-finite' % (len(bad), len(rows)))
    if bad:
        print('   NOTE: a non-finite BACKWARD at ||u||=0 is the dangerous one. It does not appear')
        print('   at step 0; it appears the first time the memory fits a value exactly, i.e. when')
        print('   the mechanism starts working. d||u||/du = u/||u|| is 0/0 there.')
    return rows


# ------------------------------------------------------------------ 4  parameters
def check4():
    line(4, 'parameter counts identical (Delta is a constant, not a parameter)')
    ok = True
    for tied in (False, True):
        n = [sum(p.numel() for p in ParityBlock(
            64, head_dim=32, channel_gates=False, tied=tied, bias=b,
            n_read=2 if tied else 1).parameters()) for b in ('l2', 'robust')]
        same = n[0] == n[1]
        ok &= same
        print('   tied=%-6s l2 %d  robust %d   %s' % (tied, n[0], n[1], OK if same else BAD))
    return ok


# ------------------------------------------------------------------ 5  does it train
class Net(nn.Module):
    def __init__(self, V, d, nl, **blk):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.mix = nn.ModuleList([ParityBlock(d, **blk) for _ in range(nl)])
        self.ffn = nn.ModuleList([SwiGLU(d) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, idx):
        x = self.emb(idx)
        for m, f in zip(self.mix, self.ffn):
            x = x + m(x)
            x = x + f(x)
        return self.head(self.lnf(x))[:, -1]

    def snorm(self):
        return float(np.mean([m.last_S_norm for m in self.mix]))


def train(steps, d, nl, nent, **blk):
    torch.manual_seed(0)
    _, _, _, V = vocab(nent)
    m = Net(V, d, nl, **blk)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    t0, losses = time.time(), []
    for i in range(steps):
        x, y = batch('mqar', 32, rng, nent, npairs=2, seqlen=32)
        loss = F.cross_entropy(m(x), y)
        if not torch.isfinite(loss):
            return dict(params=npar, note='NON-FINITE LOSS at step %d' % i, loss0=losses[0] if losses else float('nan'),
                        loss1=float('nan'), acc=float('nan'), snorm=float('nan'), secs=time.time() - t0)
        opt.zero_grad()
        loss.backward()
        bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            return dict(params=npar, note='NON-FINITE GRAD at step %d: %s' % (i, bad[0]),
                        loss0=losses[0], loss1=float(loss), acc=float('nan'),
                        snorm=m.snorm(), secs=time.time() - t0)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        losses.append(float(loss))
    m.eval()
    rng2 = np.random.default_rng(9999)
    corr = []
    with torch.no_grad():
        for _ in range(4):
            x, y = batch('mqar', 64, rng2, nent, npairs=2, seqlen=32)
            corr += (m(x).argmax(-1) == y).tolist()
    return dict(params=npar, note='', loss0=losses[0], loss1=losses[-1],
                acc=float(np.mean(corr)), snorm=m.snorm(), secs=time.time() - t0)


def check5(steps):
    line(5, 'the 2x2 trains   d=64 head_dim=32 nl=2 steps=%d  mqar npairs=2, gates OFF' % steps)
    print('   acc is NOT a result: %d steps is far short of the 6000 the ceiling sweep needed' % steps)
    print('   NOR is the tied/untied gap: that column also moves n_read (2 vs 1) and params.')
    print('   Only the l2-vs-robust comparison WITHIN a row is controlled.')
    print('   %-8s %-8s %-9s %-9s %-8s %-9s %-6s %s'
          % ('bias', 'map', 'loss0', 'loss1', 'acc', 'params', 'secs', 'end ||S||'))
    res = {}
    for bias in ('l2', 'robust'):
        for tied in (False, True):
            r = train(steps, 64, 2, 32, head_dim=32, channel_gates=False, tied=tied,
                      bias=bias, Delta=DELTA, n_read=2 if tied else 1)
            res[(bias, tied)] = r
            print('   %-8s %-8s %-9.3f %-9.3f %-8.3f %-9d %-6.0f %-9.2f %s'
                  % (bias, 'tied' if tied else 'untied', r['loss0'], r['loss1'], r['acc'],
                     r['params'], r['secs'], r['snorm'], r['note']), flush=True)
    #   COMPUTED, never typed. An earlier version hardcoded these two numbers into the header and
    #   they went stale the moment the tied arm stopped building a second key conv.
    pt, pu = res[('l2', True)]['params'], res[('l2', False)]['params']
    print('   param gap: tied %d vs untied %d (%+d, one shared map replaces two). One-directional'
          % (pt, pu, pt - pu))
    print('   for inference: a tied LOSS could be the smaller budget, a tied WIN could not.')
    return res


if __name__ == '__main__':
    STEPS = int(os.environ.get('STEPS', 400))
    print('CPU PRE-CHECK - code only. No number here is evidence about the mechanism.')
    c1, c2 = check1(), check2()
    c2b = check2b()
    c6 = check6()
    rows3 = check3()
    c4 = check4()
    res5 = check5(STEPS)

    print('\n' + '=' * 74)
    print('GATE FOR THE GPU RUN')
    grow = [k for k, r in res5.items() if r['note'] or not np.isfinite(r['loss1'])]
    fell = [k for k, r in res5.items() if not r['note'] and r['loss1'] < r['loss0']]
    print('  check 2 (algorithm is MIRAS Eq 17) ........ %s' % (OK if c2 else BAD))
    print('  check 2b (the BLOCK uses it, end-to-end) .. %s' % (OK if c2b else BAD))
    print('  check 6 (tied map is ACTUALLY tied) ....... %s' % (OK if c6 else BAD))
    print('  check 4 (params identical) ................ %s' % (OK if c4 else BAD))
    print('  check 5 (all four cells finite) ........... %s'
          % (OK if not grow else BAD + ' ' + str(grow)))
    print('  check 5 (loss fell in all four) ........... %s'
          % (OK if len(fell) == 4 else '%d of 4' % len(fell)))
    bad3 = [r for r in rows3 if not (r[3] and r[4])]
    print('  check 3 (autograd) ........................ %d of %d dtype/scale/eps cells non-finite'
          % (len(bad3), len(rows3)))
    print('\n  check 1 is deliberately NOT in this gate: it cannot fail informatively.')
    if not (c2 and c2b and c6 and c4) or grow:
        print('\n  DO NOT LAUNCH.')
