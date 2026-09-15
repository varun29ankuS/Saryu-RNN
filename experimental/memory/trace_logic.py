"""STEP BY STEP: is the block doing what we think it is doing? Nothing is launched until this
chain holds link by link.

Every check so far has been about the UPDATE RULE being MIRAS Eq 17, and the tying bug showed
that a rule can be perfectly correct while the thing it is applied to is wrong. So this walks the
whole path in order - percept, key, causality, write, decay order, read, gradient - and asserts
INTENT at each link rather than "it ran".

Run in float64, tied, gates off, conv ON (the shipping config), n_read=2. beta and decay are
pinned where a test needs them, by setting the parameters that produce them, so each link is
isolated instead of being confounded by learned gates.

RESULT: 14 of 14 links hold. The chain, in order, with what each one rules out:

  1a/1b  no k_proj, no kc when tied        there is literally one map, not two
  2      ||v|| = 1 to 2.22e-16             the value IS the percept
  3a/b/c perturb token t -> v_t moves      CAUSALITY. k_t is built only from lags >= 1, so it
         (0.433), k_t does NOT (0.0e+00),  cannot see its own value. Had k_t seen v_t, S could
         k_(t+1) moves (0.433)             store v_t at its own address and read it straight
                                           back, and every task here would need no memory at all.
  4      sharp tap: k_t == v_(t-1)         2.22e-16 - the tying invariant, exactly
  6      beta=1 write is an exact fit      1.39e-16
  7      dec=0.1, fit still 2.78e-16       decay lands on the OLD state, not the fresh write.
                                           Had the order been reversed the value would come back
                                           scaled by 0.1 and this would read ~0.9.
  8a/8b  r == norm(S norm(S q)) at 0.0,    n_read=2 is a real pointer chase, not one read twice
         and differs from one read by 0.86
  9      d loss/d tap = 5.31e-03           the tap is in the gradient path and CAN be learned

TWO BUGS IN THIS FILE, both of which made a check pass or fail for the wrong reason:

  THE PERTURBATION WAS INVISIBLE. The causality test first did x2[:, t] += 3.0 - the same
  constant added to every component. The block's first operation is LayerNorm, which subtracts
  the mean and is exactly invariant to that, so the perturbation never entered the block: v_t
  moved by 1.7e-16 and 3b "leaves k_t untouched" passed while nothing anywhere had changed. A
  vacuous pass is worse than a failure, because it reads as evidence.

  sigmoid(20) IS NOT 1. Forcing beta via a bias of 20 gives 1 - 2.06e-9, and the exact-fit check
  duly failed at 1.05e-09 - reporting MY beta, not the block's arithmetic. Same class as the
  robust_push fp32 downcast: the instrument was wrong, and its error was the right SIZE to look
  like a real finding. Bias is now 40, which rounds to exactly 1.0 in float64.

  (A third: step 7 asked for dec=0.1 and set A_log=log(20), dt_bias=0.1, which gives dec=3.4e-7.
  It printed as 0.000. The check still passed in substance - near-total forgetting and the fit
  was STILL exact, which is the evidence wanted - but by accident rather than by design.)

RAISED HERE, ANSWERED IN conv_vs_tap.py: at a UNIFORM tap, cos(k_t, v_(t-1)) is already 0.852 and
2-hop composition is 0.565 against a sharp tap's 0.667 - far closer than sharpness_sweep's
"collapses by 0.54" implies. The suspect was the causal short conv, which correlates adjacent
positions and so supplies part of the lag by itself. CONFIRMED, and it cuts both ways:

    conv=False, uniform tap  ->  cos 0.587,  2-hop 0.011   (dead, as sharpness_sweep predicts)
    conv=True,  uniform tap  ->  cos 0.852,  2-hop 0.565   (alive - the CONV is doing the lag)
    conv=False, sharp tap    ->  cos 1.000,  2-hop 0.832
    conv=True,  sharp tap    ->  cos 1.000,  2-hop 0.667   (the conv BLURS a working tap)

So tap_max is not a sufficient diagnostic in either direction: it can read 0.25 while the
mechanism works, and 1.00 while the conv degrades it. gpu_run now logs lag_cos = cos(k_t, v_(t-1))
with the LEARNED tap, which is the quantity that decides whether S is a transition operator, and
its old note about tap_max falling below 0.90 has been removed as wrong.

usage: cd experiments && python trace_logic.py

REFERENCES
  [Behrouz et al. 2025] It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization, arXiv 2504.13173
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from parity_block import ParityBlock

OK, BAD = 'OK', '*** FAIL ***'
D, HD, L = 64, 32, 10
RES = []


def report(step, what, detail, good):
    RES.append(good)
    print('  %-4s %-46s %-26s %s' % (step, what, detail, OK if good else BAD), flush=True)


def build(n_read=2, tap='sharp', beta1=True, dec1=True, bias='l2'):
    torch.manual_seed(0)
    blk = ParityBlock(D, head_dim=HD, channel_gates=False, conv=True, tied=True,
                      n_read=n_read, bias=bias, robust_eps=1e-12).double()
    with torch.no_grad():
        if tap == 'sharp':
            blk.tap.copy_(torch.tensor([20., 0., 0., 0.], dtype=torch.float64))
        elif tap == 'uniform':
            blk.tap.zero_()
        if beta1:                        # force beta -> 1 so a write is an EXACT fit
            blk.beta.weight.zero_()
            #   40, not 20. sigmoid(20) = 1 - 2.06e-9, so "beta=1" was really beta = 1 - 2e-9 and
            #   the exact-fit check failed at 1.05e-09 - which WAS that gap, not a block error.
            #   sigmoid(40) rounds to exactly 1.0 in float64.
            blk.beta.bias.fill_(40.0)
        if dec1:                         # force dec -> 1 so nothing fades during the trace
            blk.A_log.fill_(math.log(1e-12))
    blk.debug = True
    return blk


def run(blk, x):
    with torch.no_grad():
        blk(x)
    return blk.trace


if __name__ == '__main__':
    print(__doc__.strip().splitlines()[0])
    print('d=%d head_dim=%d L=%d  float64\n' % (D, HD, L))
    torch.manual_seed(1)
    x = torch.randn(2, L, D, dtype=torch.float64)

    # ---------------------------------------------------------------- 1  one map
    print('THE MAP')
    blk = build()
    report('1a', 'tied block has NO separate key projection',
           'k_proj absent: %s' % (not hasattr(blk, 'k_proj')), not hasattr(blk, 'k_proj'))
    report('1b', 'tied block has NO separate key conv',
           'kc is None: %s' % (blk.kc is None), blk.kc is None)

    tr = run(blk, x)
    steps, reads = tr['steps'], tr['reads']
    vn = max(abs(float(s[1].norm(dim=-1).max()) - 1.0) for s in steps)
    report('2', 'the value IS the percept, unit norm', 'max |1-||v|| | = %.2e' % vn, vn < 1e-12)

    # ---------------------------------------------------------------- 3  causality
    print('\nCAUSALITY - the one that would make the whole task trivial if wrong')
    t = 5
    x2 = x.clone()
    #   A RANDOM perturbation, not a constant. The first version did x2[:, t] += 3.0, adding the
    #   SAME value to every component - and the block's first op is LayerNorm, which subtracts the
    #   mean and is exactly invariant to that. The perturbation never reached the block, v_t moved
    #   by 1.7e-16, and 3b "passed" while testing nothing at all.
    torch.manual_seed(7)
    x2[:, t] = torch.randn(x.shape[0], D, dtype=torch.float64)
    tr2 = run(build(), x2)
    s1, s2 = steps, tr2['steps']
    dv_t = float((s1[t][1] - s2[t][1]).abs().max())        # value AT t must move
    dk_t = float((s1[t][0] - s2[t][0]).abs().max())        # key AT t must NOT
    dk_n = float((s1[t + 1][0] - s2[t + 1][0]).abs().max())  # key at t+1 must move
    report('3a', 'changing token t CHANGES v_t', 'delta = %.3e' % dv_t, dv_t > 1e-6)
    report('3b', 'changing token t leaves k_t UNTOUCHED', 'delta = %.3e' % dk_t, dk_t < 1e-12)
    report('3c', 'changing token t CHANGES k_(t+1)', 'delta = %.3e' % dk_n, dk_n > 1e-6)
    if dk_t >= 1e-12:
        print('       a key that sees its own value means S can store v_t at its own address')
        print('       and read it straight back - the task would need no memory at all.')

    # ---------------------------------------------------------------- 4/5  the tap
    print('\nTHE TAP - sharp is the mechanism, uniform is what training actually starts from')
    e_sharp = max(float((steps[i][0] - steps[i - 1][1]).abs().max()) for i in range(1, L))
    report('4', 'sharp tap: k_t EQUALS v_(t-1)', 'max delta = %.2e' % e_sharp, e_sharp < 1e-12)

    bu = build(tap='uniform')
    tu = run(bu, x)
    su = tu['steps']
    cos_u = sum(float(F.cosine_similarity(su[i][0], su[i - 1][1], dim=-1).mean())
                for i in range(1, L)) / (L - 1)
    wmax = float(bu.tap_weights().max())
    report('5a', 'uniform tap: dominant weight', '%.3f' % wmax, True)
    report('5b', 'uniform tap: cos(k_t, v_(t-1))', '%.3f' % cos_u, True)
    print('       sharpness_sweep: composition survives while the dominant tap is >=0.90 and')
    print('       collapses by 0.54. Uniform starts at %.2f, BELOW the collapse point - so at' % wmax)
    print('       init the tied arm cannot compose and must LEARN the lag first. Whether it does')
    print('       is the open question; tap_max is logged every eval in gpu_run for this reason.')

    # ---------------------------------------------------------------- 6/7  write and decay
    print('\nTHE WRITE, and whether decay lands before or after it')
    S_end = tr['S']
    kT, vT = steps[L - 1][0], steps[L - 1][1]
    fit = float((torch.einsum('bhvk,bhk->bhv', S_end, kT) - vT).abs().max())
    report('6', 'beta=1 write is an EXACT fit: S k_t == v_t', 'max err = %.2e' % fit, fit < 1e-9)

    #   decay must hit the OLD state, not the fresh write. With dec ~ 0.1, a write applied BEFORE
    #   the decay would come back scaled by 0.1; applied after the decay it comes back intact.
    bd = build(dec1=False)
    with torch.no_grad():
        #   aim for dec = 0.1 exactly: g = -exp(A_log)*softplus(dt_bias) must be ln(0.1).
        #   softplus(0.54132) = 1, so exp(A_log) = 2.302585 does it. The first version set
        #   A_log=log(20), dt_bias=0.1 -> g = -14.9 -> dec = 3.4e-7, which printed as 0.000:
        #   I asked for mild forgetting and produced total forgetting.
        bd.A_log.fill_(math.log(2.302585))
        bd.dt_bias.fill_(0.54132)
    td = run(bd, x)
    sd, Sd = td['steps'], td['S']
    dec_val = float(torch.exp(-torch.exp(bd.A_log) * F.softplus(bd.dt_bias)).mean())
    fit_d = float((torch.einsum('bhvk,bhk->bhv', Sd, sd[L - 1][0]) - sd[L - 1][1]).abs().max())
    report('7', 'decay hits the OLD state, not the fresh write',
           'dec=%.3f, fit err %.2e' % (dec_val, fit_d), fit_d < 1e-9)

    # ---------------------------------------------------------------- 8  the read
    print('\nTHE READ - n_read=2 must be a genuine pointer chase, not one read done twice')
    S_t, q_t, r_t = reads[L - 1]
    manual = q_t
    for _ in range(2):
        manual = F.normalize(torch.einsum('bhvk,bhk->bhv', S_t, manual), dim=-1)
    e_read = float((manual - r_t).abs().max())
    report('8a', 'r == normalize(S normalize(S q))', 'max err = %.2e' % e_read, e_read < 1e-12)
    one = F.normalize(torch.einsum('bhvk,bhk->bhv', S_t, q_t), dim=-1)
    sep = float((one - r_t).abs().max())
    report('8b', 'two reads DIFFER from one (the chase is real)', 'delta = %.3e' % sep, sep > 1e-6)

    # ---------------------------------------------------------------- 9  gradient
    print('\nTHE GRADIENT - the tap must be learnable or "uniform init" is a dead end')
    bg = build(tap='uniform')
    bg.debug = False
    y = bg(x.clone().requires_grad_(False))
    y.pow(2).mean().backward()
    g_tap = bg.tap.grad
    gn = float(g_tap.abs().max()) if g_tap is not None else 0.0
    report('9', 'd loss / d tap is non-zero', 'max |grad| = %.3e' % gn, gn > 1e-12)

    # ---------------------------------------------------------------- 10  composition
    print('\nCOMPOSITION THROUGH THE BLOCK\'S OWN STATE (not a numpy mock-up)')
    for name, tap in (('sharp', 'sharp'), ('uniform', 'uniform')):
        b = build(tap=tap)
        t_ = run(b, x)
        S_, st = t_['S'], t_['steps']
        #   walk: start from the value at step i, apply S twice, compare to the value 2 later
        cs = []
        for i in range(2, L - 1):
            z0 = st[i - 2][1]
            z1 = F.normalize(torch.einsum('bhvk,bhk->bhv', S_, z0), dim=-1)
            z2 = F.normalize(torch.einsum('bhvk,bhk->bhv', S_, z1), dim=-1)
            cs.append(float(F.cosine_similarity(z2, st[i][1], dim=-1).mean()))
        print('    %-10s 2-hop cos through S: %.3f' % (name, sum(cs) / len(cs)), flush=True)
    print('    sharp should compose; uniform is the honest picture of step 0 of training.')

    print('\n' + '=' * 74)
    print('%d of %d links hold.' % (sum(RES), len(RES)))
    print('GO' if all(RES) else 'DO NOT RUN - a link in the chain is broken.')
