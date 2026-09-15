"""IS THE SHORT CONV DOING THE TAP'S JOB? Raised by trace_logic, and the answer changes a metric.

trace_logic found something that did not fit: at a UNIFORM tap - a flat average over four lags,
dominant weight 0.25 - cos(k_t, v_(t-1)) was already 0.852 and 2-hop composition ran 0.565
against a sharp tap's 0.667. sharpness_sweep says composition collapses once the dominant tap
falls to 0.54, so a flat tap should have been dead and was not.

The suspect: ShortConv is a CAUSAL depthwise conv of width 4 applied to the percept stream, so
p_t already mixes z_(t-3..t). Adjacent percepts are therefore correlated before the tap ever runs,
and a flat tap over a correlated stream still points roughly one step back.

usage: cd experiments && python conv_vs_tap.py

RESULT

    conv     tap       cos(k_t,v_t-1)     2-hop
    True     sharp     1.000              0.667
    True     uniform   0.852              0.565
    False    sharp     1.000              0.832
    False    uniform   0.587              0.011

BOTH HALVES MATTER, AND THEY POINT OPPOSITE WAYS.

  The conv MASKS a broken tap. Without it, a uniform tap gives cos 0.587 - the honest value for a
  blend of four near-orthogonal percepts - and composition is 0.011, i.e. DEAD. With it, the same
  flat tap composes at 0.565. So a model can compose while its tap never sharpens, because the
  conv supplies the lag. CONSEQUENCE: tap_max is NOT a sufficient diagnostic. gpu_run's note that
  "if tap_max drifts below 0.90 the lag was never learned" was wrong, and is now fixed - the run
  logs cos(k_t, v_(t-1)) with the LEARNED tap, which is the quantity that actually decides whether
  S is a transition operator.

  The conv also DEGRADES a working tap: 0.667 with conv against 0.832 without, at an identical
  sharp tap. The smearing that helps a flat tap hurts a sharp one, because p_t is no longer a
  clean distinct percept but a mixture of four, so S maps blurred to blurred and iterating
  amplifies the blur. Composition is the thing that iterates, so composition pays.

That is the FOURTH time a parity feature has worked against the mechanism, after the key-side
conv, the k-only normalisation, and the Mamba decay spread. The pattern is consistent enough to
state plainly: the incumbent's machinery is tuned for ONE-SHOT recall, and every piece of it that
smooths or forgets buys recall at composition's expense.

NOT A LICENCE TO DROP THE CONV. It is in every DeltaNet/GDN implementation, removing it unilaterally
from the tied arm would hand that arm a change the untied arm did not get, and the 0.832 here is a
hand-set tap on random input, not a trained model. The right move is to make conv an AXIS if there
is budget, and otherwise to log the honest metric and interpret accordingly.

REFERENCES
  [Yang et al. 2024a] Parallelizing Linear Transformers with the Delta Rule over Sequence Length, arXiv 2406.06484
  [Gu & Dao 2023] Mamba: Linear-Time Sequence Modeling with Selective State Spaces, arXiv 2312.00752
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from parity_block import ParityBlock

D, HD, L = 64, 32, 10


def probe(conv, tap, d=D, hd=HD, seqlen=L):
    torch.manual_seed(1)
    x = torch.randn(2, seqlen, d, dtype=torch.float64)
    torch.manual_seed(0)
    b = ParityBlock(d, head_dim=hd, channel_gates=False, conv=conv, tied=True, n_read=2).double()
    with torch.no_grad():
        if tap == 'sharp':
            b.tap.copy_(torch.tensor([20., 0., 0., 0.], dtype=torch.float64))
        else:
            b.tap.zero_()
        b.beta.weight.zero_()
        b.beta.bias.fill_(40.)          # beta -> exactly 1.0 in float64
        b.A_log.fill_(-27.63)           # dec -> 1, nothing fades during the probe
    b.debug = True
    with torch.no_grad():
        b(x)
    st, S = b.trace['steps'], b.trace['S']
    cos = sum(float(F.cosine_similarity(st[i][0], st[i - 1][1], dim=-1).mean())
              for i in range(1, seqlen)) / (seqlen - 1)
    hops = []
    for i in range(2, seqlen - 1):
        z = st[i - 2][1]
        for _ in range(2):
            z = F.normalize(torch.einsum('bhvk,bhk->bhv', S, z), dim=-1)
        hops.append(float(F.cosine_similarity(z, st[i][1], dim=-1).mean()))
    return cos, sum(hops) / len(hops)


if __name__ == '__main__':
    print('IS THE CONV DOING THE TAP\'S JOB?   d=%d head_dim=%d L=%d float64' % (D, HD, L))
    print('  %-8s %-9s %-18s %s' % ('conv', 'tap', 'cos(k_t,v_t-1)', '2-hop'))
    out = {}
    for conv in (True, False):
        for tap in ('sharp', 'uniform'):
            c, h = probe(conv, tap)
            out[(conv, tap)] = (c, h)
            print('  %-8s %-9s %-18.3f %.3f' % (conv, tap, c, h), flush=True)
    print()
    print('  masks a broken tap:  uniform composes %.3f WITH conv vs %.3f without'
          % (out[(True, 'uniform')][1], out[(False, 'uniform')][1]))
    print('  degrades a good one: sharp composes %.3f WITH conv vs %.3f without'
          % (out[(True, 'sharp')][1], out[(False, 'sharp')][1]))
    print('  => tap_max cannot tell you whether the lag is there. Log cos(k_t, v_(t-1)) instead.')
