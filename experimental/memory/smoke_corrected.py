"""CORRECTED CPU SMOKE. Same arms, but on cells that actually test the claim.

Two bugs the first smoke exposed, both fixed here:

  1. THE ZERO-DISTRACTOR CHAIN HAS A SHORTCUT. At ratio=0.0 the sequence contains only path
     edges, so the answer is always the unique node that never appears as a SOURCE - correct
     100.0% of the time with no composition at all. The untied control, which compose_probe
     proves cannot compose past one hop, duly scored 0.987 there. Every claim-bearing cell
     now uses ratio >= 0.5, where that heuristic is correct 0.0% of the time.
  2. THE TAP WAS INITIALISED AT THE ANSWER. tap[0]=4.0 starts the model at the hard shift, so
     the reported [1.0, 0, 0, 0] showed only that it had not moved. `tied-learn` below starts
     from UNIFORM and has to discover the lag, which is the honest test.

Also: 400 steps was far too few - the transformer sat at 0.107 on a 4-pair lookup while being
able to overfit a batch to 1.000, i.e. undertrained, not broken. Budget is now a knob.

usage: STEPS=1500 python experiments/smoke_corrected.py
"""
from __future__ import annotations

import os
import numpy as np

from shift_arms import run

STEPS = int(os.environ.get('STEPS', 1500))
D = int(os.environ.get('D', 64))
NL = int(os.environ.get('NL', 2))
NENT = int(os.environ.get('NENT', 32))

ARMS = [
    ('transformer', dict(n_read=1)),                 # THE CEILING - gates every other number
    ('gdn',         dict(n_read=1)),                 # the incumbent
    ('untied',      dict(n_read=2)),                 # control: iterated read, INDEPENDENT k/v
    ('tied',        dict(n_read=1)),                 # tied map, ONE read -> one hop only
    ('tied',        dict(n_read=2)),                 # tied map + pointer chase -> the claim
]

CELLS = [
    ('mqar  npairs=4',        'mqar',  dict(npairs=4, seqlen=32)),
    ('chain d2 ratio=1.0',    'chain', dict(depth=2, ratio=1.0, seqlen=48)),
]


def label(arm, kw):
    return '%s(n_read=%d)' % (arm, kw['n_read'])


if __name__ == '__main__':
    print('CORRECTED SMOKE  d=%d nl=%d steps=%d nent=%d' % (D, NL, STEPS, NENT))
    print('  chain uses ratio=1.0: the unique-sink shortcut is correct 0.0%% there (100%% at ratio 0)')
    print('  fla is GPU-only, so `gdn` is the sequential pure-PyTorch reference')
    print()
    for cell_name, task, kw in CELLS:
        print('=== %s ===' % cell_name, flush=True)
        print('  %-22s %8s %9s %9s %9s %6s  %s'
              % ('arm', 'acc', 'loss0', 'loss1', 'params', 'secs', 'tap'), flush=True)
        ceiling = None
        for arm, akw in ARMS:
            r = run(arm, task, STEPS, D, NL, NENT, **akw, **kw)
            if arm == 'transformer':
                ceiling = r['acc']
            tap = ('max %.3f %s' % (r['tap_max'], r['tap'])) if r.get('tap_max') else '-'
            print('  %-22s %8.3f %9.3f %9.3f %9d %6.0f  %s'
                  % (label(arm, akw), r['acc'], r['loss0'], r['loss1'],
                     r['params'], r.get('secs', 0), tap), flush=True)
        if ceiling is not None:
            print('  CEILING %.3f -> %s' % (
                ceiling, 'cell READABLE' if ceiling > 0.90 else
                '*** UNREADABLE: ceiling below 0.90, no arm number here counts ***'), flush=True)
        print(flush=True)

    print('WHAT THIS CAN AND CANNOT SHOW')
    print('  CAN: that every arm trains, that the ceiling can solve the cell, that the tap stays')
    print('       sharp, and whether a tied map + pointer chase separates from the incumbent at all.')
    print('  CANNOT: support the claim. d=%d, %d layers, %d steps is a CODE CHECK. Any gap here is'
          % (D, NL, STEPS))
    print('       provisional and must be reproduced at the spec scale with 3 seeds before it counts.')
    print('  NOTE on MQAR: keys sit immediately before their values, so a lag-1 tied map has an')
    print('       inductive bias matched to the task surface form. A tied win on MQAR may be that')
    print('       artefact rather than a general recall advantage - the CHAIN cell is the real test.')
