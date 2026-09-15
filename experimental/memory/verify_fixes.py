"""Targeted checks for each audit fix. Each check is built to FAIL on the pre-fix code."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gpu_run as G
from gpu_run import ParityNet, probe, param_groups, swap_acc, floors, assert_tied, head_decay
from shift_arms import batch
from recall_tasks import vocab, make_chain

ok = True


def rep(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print('  %-62s %-12s %s' % (name, 'OK' if cond else '*** FAIL ***', detail), flush=True)


NENT = 32
sep, qry, pad, V = vocab(NENT)
base = dict(head_dim=32, channel_gates=False, tap_init='uniform', robust_eps=1e-4, Delta=0.3, bias='l2')
cell = dict(depth=3, ratio=1.0, dmode='chain', seqlen=2 * (3 + 3) + 3 + 4)


def force_tap(net, lag):
    for m in net.mix:
        with torch.no_grad():
            t = torch.zeros(4)
            t[lag - 1] = 20.0
            m.tap.copy_(t)


print('1. probe(): lag1/lag2 respond to WHERE the tap sits, on a task batch, every layer')
for conv in (True, False):
    torch.manual_seed(0)
    net = ParityNet(V, 64, 2, tied=True, n_read=4, conv=conv, **base)
    xp, _ = batch('chain', 16, np.random.default_rng(1), NENT, **cell)
    force_tap(net, 1)
    a = probe(net, xp, pad)
    force_tap(net, 2)
    b = probe(net, xp, pad)
    rep('conv=%s tap@lag1 -> lag1=1, lag2<1' % conv, abs(a['lag1'] - 1) < 1e-4 and a['lag2'] < 0.99,
        'lag1 %.4f lag2 %.3f' % (a['lag1'], a['lag2']))
    rep('conv=%s tap@lag2 -> lag2=1, lag1<1' % conv, abs(b['lag2'] - 1) < 1e-4 and b['lag1'] < 0.99,
        'lag1 %.3f lag2 %.4f' % (b['lag1'], b['lag2']))

print('\n2. probe(): decay is the one APPLIED (independent route: pre-hook + the forward formula)')
torch.manual_seed(0)
net = ParityNet(V, 64, 2, tied=True, n_read=1, **base)
for m in net.mix:
    with torch.no_grad():
        m.a_proj.bias.fill_(2.0)
xp, _ = batch('chain', 16, np.random.default_rng(1), NENT, **cell)
seen = {}
hooks = [m.register_forward_pre_hook(lambda mod, inp: seen.__setitem__(mod, inp[0])) for m in net.mix]
with torch.no_grad():
    net(xp)
    lv = (xp != pad).float().unsqueeze(-1)
    manual = []
    for m in net.mix:
        dec = torch.exp(-torch.exp(m.A_log) * F.softplus(m.a_proj(m.ln(seen[m])) + m.dt_bias))
        manual.append(((dec * lv).sum((0, 1)) / lv.sum((0, 1))).numpy())
for h in hooks:
    h.remove()
manual = np.concatenate(manual)
got = probe(net, xp, pad)['dec']
legacy = head_decay(net)
rep('probe dec == hook+formula', np.abs(got - manual).max() < 1e-5,
    'max diff %.1e  (legacy head_decay off by %.3f)' % (np.abs(got - manual).max(), np.abs(legacy - manual).max()))
rep('n heads = layers x H', len(got) == 2 * (64 // 32), 'n=%d' % len(got))

print('\n3. param_groups(): no-decay set is exactly the flagged + 1-D tensors')
names = {id(p): n for n, p in net.named_parameters()}
g = param_groups(net, 0.01)
nd = {names[id(p)] for p in g[1]['params']}
dd = {names[id(p)] for p in g[0]['params']}
rep('A_log, dt_bias, tap exempt', all(any(n.endswith(s) for n in nd) for s in ('A_log', 'dt_bias', 'tap')))
rep('no 2-D weight exempt, no 1-D decayed',
    all(dict(net.named_parameters())[n].ndim < 2 for n in nd) and all(dict(net.named_parameters())[n].ndim >= 2 for n in dd))
rep('every parameter in exactly one group', len(nd) + len(dd) == len(names) and not (nd & dd),
    '%d decay / %d exempt' % (len(dd), len(nd)))

print('\n4. swap_acc(): an oracle that follows the chain from the query scores new=1, old=0')


class Oracle(nn.Module):
    def forward(self, x):
        out = torch.full((x.shape[0], V), -1e9)
        for i, row in enumerate(x.tolist()):
            body = [t for t in row if t != pad]
            cut = body.index(sep)
            succ = dict((body[j], body[j + 1]) for j in range(0, cut, 2))
            node = row[-2]
            for _ in range(cell['depth']):
                node = succ[node]
            out[i, node] = 0.0
        return out


new, old, flip = swap_acc(Oracle(), NENT, n=256, **cell)
rep('oracle new/old/flip', new == 1.0 and old == 0.0 and flip == 1.0, '%.3f / %.3f / %.3f' % (new, old, flip))


class QueryBlind(nn.Module):
    """Answers the end of the chain that does NOT start at the query-independent 'first root in
    the body' - i.e. ignores the query entirely. Must have flip == 0 by construction."""
    def forward(self, x):
        out = torch.full((x.shape[0], V), -1e9)
        for i, row in enumerate(x.tolist()):
            body = [t for t in row if t != pad]
            cut = body.index(sep)
            es = [(body[j], body[j + 1]) for j in range(0, cut, 2)]
            sinks = sorted({b for _, b in es} - {a for a, _ in es})
            out[i, sinks[0]] = 0.0
        return out


new, old, flip = swap_acc(QueryBlind(), NENT, n=256, **cell)
rep('query-blind model: flip == 0, new ~ old ~ 0.5', flip == 0.0 and abs(new + old - 1) < 1e-9,
    '%.3f / %.3f / %.3f  (new alone would look like a 0.5 model)' % (new, old, flip))

print('\n4b. best_ceiling(): NaN never wins, all-NaN gives None')
rep('{NL2: nan, NL8: 0.12} -> NL8', G.best_ceiling({'ceiling  NL2': float('nan'), 'ceiling  NL8': 0.12}) == ('ceiling  NL8', 0.12))
rep('{NL8: 0.12, NL2: nan} -> NL8', G.best_ceiling({'ceiling  NL8': 0.12, 'ceiling  NL2': float('nan')}) == ('ceiling  NL8', 0.12))
rep('all NaN -> None', G.best_ceiling({'a': float('nan')}) is None)
_old = max({'ceiling  NL2': float('nan'), 'ceiling  NL8': 0.12}.items(), key=lambda kv: kv[1])[0]
rep('(the bug it fixes: plain max picks the NaN arm)', _old == 'ceiling  NL2', 'plain max -> %s' % _old)

print('\n4c. probe(): tap printed, and bg measures percept similarity with no temporal relation')
for conv in (True, False):
    torch.manual_seed(0)
    net = ParityNet(V, 64, 2, tied=True, n_read=1, conv=conv, **base)
    xp, _ = batch('chain', 16, np.random.default_rng(1), NENT, **cell)
    force_tap(net, 1)
    p = probe(net, xp, pad)
    rep('conv=%s tap@lag1: tap[0]~1, lag1=1 > bg' % conv,
        p['tap'][0] > 0.999 and abs(p['lag1'] - 1) < 1e-4 and p['bg'] < 0.9,
        'tap %s  lag1 %.3f  lag2 %.3f  bg %.3f' % (np.round(p['tap'], 3), p['lag1'], p['lag2'], p['bg']))
rep('dmode=edge gives None (no second depth-hop root)', swap_acc(Oracle(), NENT, n=16, **dict(cell, dmode='edge')) is None)

print('\n5. floors(): the leak is reported under edge and closed under chain')
for depth in (2, 4):
    se, sh = floors(NENT, trials=500, depth=depth, ratio=1.0, seqlen=40, dmode='edge')
    ce, ch = floors(NENT, trials=500, depth=depth, ratio=1.0, seqlen=40, dmode='chain')
    rep('depth %d edge shortcut=1, chain shortcut=sink=0.5' % depth,
        sh == 1.0 and abs(ch - 0.5) < 1e-9 and abs(ce - 0.5) < 1e-9,
        'edge %.3f/%.3f  chain %.3f/%.3f' % (se, sh, ce, ch))

print('\n6. batch() raises on an unbuildable cell instead of hanging')
try:
    batch('chain', 4, np.random.default_rng(0), NENT, depth=20, ratio=1.0, seqlen=200, dmode='chain')
    rep('raised', False)
except ValueError as e:
    rep('raised ValueError', True, str(e)[:60])

print('\n7. assert_tied(): checks EVERY tied layer (sabotage layer 1 only)')
torch.manual_seed(0)
net = ParityNet(V, 64, 2, tied=True, n_read=1, **base)
rep('clean 2-layer net passes', assert_tied(net) < 1e-4)
net.mix[1].tap_weights = lambda: torch.tensor([0.0, 1.0, 0.0, 0.0])
try:
    assert_tied(net)
    rep('layer-1 sabotage caught', False, 'NOT CAUGHT')
except AssertionError as e:
    rep('layer-1 sabotage caught', 'layer 1' in str(e), str(e))

print('\n8. last_S_norm is lazy and still readable')
torch.manual_seed(0)
blk = net.mix[0]
with torch.no_grad():
    net(xp)
rep('finite float after forward', isinstance(blk.last_S_norm, float) and np.isfinite(blk.last_S_norm),
    '%.3f' % blk.last_S_norm)

print('\nALL FIX CHECKS PASSED' if ok else '\n*** SOME FIX CHECKS FAILED ***')
