"""IS THERE ANY CELL ON CPU WHERE THE CEILING CLEARS 0.90?

The 2000-step gate failed: transformer 0.122 on MQAR-4, 0.174 on chain d2 r=1.0. But the same
model overfits a single batch to acc 1.000, so it CAN fit - it cannot generalise at this budget.
reason_battery showed the same shape: a transformer solved load 2 at 4000 steps (1.000) but only
reached 0.310 at load 3. That is the saturate/floor gap again - trivial cells everything solves,
harder cells nothing solves, nothing readable between.

This sweeps width and steps to find a readable cell, or to establish there isn't one on CPU -
which is itself the answer to "verify before sending to GPU".
"""
import os
from shift_arms import run

print('CEILING SWEEP - transformer only. Gate is acc > 0.90.')
print('chance = 1/32 = 0.031')
print()
print('  %-6s %-7s %-9s %8s %9s %6s  %s' % ('d', 'steps', 'cell', 'acc', 'loss1', 'secs', 'verdict'))
print('  ' + '-' * 70)
found = []
for d in (64, 128):
    for steps in (2000, 6000):
        for npairs in (2, 3, 4):
            r = run('transformer', 'mqar', steps, d, 2, 32, npairs=npairs, seqlen=32)
            ok = r['acc'] > 0.90
            if ok:
                found.append((d, steps, npairs, r['acc']))
            print('  %-6d %-7d %-9s %8.3f %9.3f %6.0f  %s'
                  % (d, steps, 'mqar-%d' % npairs, r['acc'], r['loss1'], r.get('secs', 0),
                     'READABLE' if ok else ''), flush=True)
print()
if found:
    print('  READABLE CELLS FOUND:')
    for d, s, n, a in found:
        print('    d=%d steps=%d mqar-%d -> %.3f' % (d, s, n, a))
    print('  The largest npairs that clears 0.90 is the hardest cell we can compare arms on locally.')
else:
    print('  NO READABLE CELL ON CPU at these budgets. The code is verified (all arms train, all')
    print('  overfit a batch to 1.000) but a DISCRIMINATIVE comparison needs GPU scale. That is a')
    print('  legitimate outcome of local verification, not a failure of it.')
