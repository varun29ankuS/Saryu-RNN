"""MQAR and multi-hop CHAIN generators for SPEC_shift_vs_gdn.md, with answerability self-checks.

Every design rule here was paid for by a bug found on 2026-09-14, and each is asserted rather
than trusted:

  LEFT-PAD.            reason_battery right-padded while the model read position -1, so it was
                       asked to answer from a PAD token with 11-46 PADs between the query and
                       the readout. episodic.py had ALREADY documented this exact bug. The
                       query must be the last non-pad token, and that is asserted.
  PAD OUTSIDE ENTITIES. PAD was 0, which is a legal entity id, so padding was not merely
                       uninformative but confusable with an answer.
  DEPTH AND INTERFERENCE ARE SEPARATE AXES. The chain builder previously took `nedge` and
                       `depth`, so distractors = nedge - depth: varying depth silently varied
                       interference, and I read the artefact as "deeper tasks are easier".
                       Here `depth` and `ratio` are independent arguments.
  DISTRACTORS AS A RATIO. Whole-edge granularity put the discriminative band BETWEEN two
                       settings - 0 distractors and everything saturates, 1 and everything
                       floors. A ratio gives a continuous dial.
  dist MUST SPLIT.     dist was len(seq), identical for every sample, so the median split left
                       the far bucket empty and `far` was silently nan at exactly the loads
                       whose ceiling was readable.

usage: python experiments/recall_tasks.py      # runs the self-checks, trains nothing

REFERENCES
  [Arora et al. 2023] Zoology: Measuring and Improving Recall in Efficient Language Models, arXiv 2312.04927
"""
from __future__ import annotations

import numpy as np

#   vocabulary: entities, then specials. PAD last so it can never be an entity.
NENT_DEFAULT = 64


def vocab(nent, nrel=0):
    """returns (SEP, QRY, PAD, V). Entities occupy [0, nent), relations [nent+3, nent+3+nrel).

    `nrel` ADDED AFTER AN AUDIT FOUND make_transport EMITTING OUT-OF-VOCABULARY TOKENS. It set
    rel0 = nent+3, which is exactly the V this function used to return, so relation ids ran from
    V upward - past the end of any embedding table sized V. Measured at nent=64, nrel=4: the
    generator emitted token 70 against a valid range of 0..66. Any Net would index out of bounds
    or silently wrap. Defaulting nrel=0 keeps every existing caller unchanged."""
    sep, qry, pad = nent, nent + 1, nent + 2
    return sep, qry, pad, nent + 3 + nrel


def _finish(seq, seqlen, pad):
    """LEFT-pad to seqlen. The query must end up at position -1."""
    if len(seq) > seqlen:
        return None                                   # caller decides; never silently truncate
    return [pad] * (seqlen - len(seq)) + seq


def make_mqar(rng, npairs, seqlen, nent=NENT_DEFAULT):
    """k1 v1 k2 v2 ... SEP k_i  ->  v_i.   Single hop: storage and addressing, no composition."""
    sep, qry, pad, _ = vocab(nent)
    #   MQAR needs 2*npairs DISTINCT entities. Asserting rather than letting numpy raise
    #   deep in the call stack, which is what happened at npairs=64 with nent=64.
    if 2 * npairs > nent:
        raise ValueError('npairs=%d needs %d distinct entities but nent=%d'
                         % (npairs, 2 * npairs, nent))
    ent = rng.choice(nent, size=2 * npairs, replace=False)
    ks, vs = ent[:npairs], ent[npairs:]
    seq = []
    for k, v in zip(ks, vs):
        seq += [int(k), int(v)]
    i = int(rng.integers(0, npairs))
    dist = len(seq) - 2 * i                           # varies with WHICH pair is queried
    seq += [sep, int(ks[i])]
    out = _finish(seq, seqlen, pad)
    return (out, int(vs[i]), dist) if out is not None else None


def make_transport(rng, length, nrel, seqlen, nent=NENT_DEFAULT, ordered=True):
    """ORDERED relation chain - the non-abelian TRANSPORT task, which make_chain does NOT test.

    Sequence: a0 r1 a1 r2 a2 ... rL aL, then SEP a0 r1 r2 ... rL QRY -> aL

    The relations arrive IN PATH ORDER, so a transport model composes as it reads and needs no
    storage: L relations are L group multiplications in a fixed state, not L facts to remember.
    reason_battery.py: "the signature of a transport model is accuracy that stays FLAT as the
    ordered chain gets longer, while a lookup model degrades as it runs out of slots."

    ordered=False shuffles the (node, relation) steps, which destroys transport - the path must be
    FOUND before it can be walked. That is the control, and it is the regime make_chain was in.

    Measured in nonabelian.py: ordered non-abelian stays 1.000 to L=64; shuffled falls to chance
    by L=8. So this generator and make_chain test DIFFERENT CLAIMS and must not be conflated.
    """
    #   vocab MUST be told about the relations, or rel0 lands exactly on V - see vocab()
    sep, qry, pad, _ = vocab(nent, nrel)
    #   relations live ABOVE the entity range so they can never be confused with a node
    rel0 = nent + 3
    if length + 1 > nent:
        return None
    nodes = rng.choice(nent, size=length + 1, replace=False)
    rels = rng.integers(0, nrel, size=length)
    steps = [(int(nodes[i]), rel0 + int(rels[i]), int(nodes[i + 1])) for i in range(length)]
    order = list(range(length))
    if not ordered:
        rng.shuffle(order)
    seq = []
    for i in order:
        a, r, b = steps[i]
        seq += [a, r, b]
    dist = len(seq)
    seq += [sep, int(nodes[0])] + [rel0 + int(r) for r in rels] + [qry]
    out = _finish(seq, seqlen, pad)
    return (out, int(nodes[length]), dist) if out is not None else None


def make_chain(rng, depth, ratio, seqlen, nent=NENT_DEFAULT, dmode='edge'):
    """Edges shuffled; query the node `depth` hops from the start.

    dmode='chain' (added 2026-09-15, THE FORM ANY NEW RESULT SHOULD USE). In the original
    dmode='edge' every distractor is a lone edge a->b whose source never appears as a target,
    while the answer's source path[depth-1] always does. So for depth >= 2 the answer is THE UNIQUE
    SINK WHOSE SOURCE IS ALSO A TARGET - a rule that never reads the query and needs no composition.
    Measured over 3000 samples it scores 1.000 at depths 2, 3, 4 and 6. dmode='chain' groups the
    same round(depth*ratio) distractor edges into chains of length `depth`, each from its own root,
    so every sink has a target as its source and only the query picks the right chain. The edge
    count and sequence length are unchanged; the guess-among-sinks floor becomes
    1/(n_chains + 1). round(depth*ratio) must be a multiple of depth, i.e. ratio a whole number.

    depth and ratio are INDEPENDENT: n_distractors = round(depth * ratio), so interference can
    be swept at fixed depth and depth swept at fixed interference. Shuffling is the point - the
    chain is not local, so no fixed window can solve it.

    THE FLOOR HERE IS NOT 1/nent, AND IT MOVES WITH DEPTH. Distractor edges are built from `rest`,
    which is disjoint from the path, so no node ever has two outgoing edges. That keeps the walk
    unambiguous, but it also means the nodes that never appear as a SOURCE are path[depth] plus
    every distractor target. At ratio=1.0 that is depth+1 candidates, measured over 3000 samples:

            depth   mean sinks   guess-among-sinks   1/nent
            2       3.00         0.333               0.016
            3       4.00         0.250               0.016
            4       5.00         0.200               0.016
            6       7.00         0.143               0.016

    So an accuracy of 0.25 at depth 3 is the HEURISTIC, not weak composition. And because the
    floor FALLS as depth rises, a flat accuracy curve across depth is not automatically a flat
    mechanism - it can be a model riding a floor that is itself moving. Anything reading these
    tasks across depth must print the floor next to the score; gpu_run does.

    (The older ratio=0.0 form was worse still: with only path edges there is exactly ONE sink and
    the heuristic is correct 100% of the time. That is why every claim-bearing cell uses ratio>=0.5.)

    AND A SECOND, SEPARATE GIFT TO US SPECIFICALLY - THE ADJACENCY PRIOR. Edges are emitted as
    `seq += [a, b]`, so a target ALWAYS immediately follows its source. A lag-1 tied map has
    exactly that prior built in: with a sharp tap, k_t is the previous percept and v_t the current
    one, so ONE hop is read straight off the surface form without any memory being consulted.
    smoke_corrected.py flagged this for MQAR - "keys sit immediately before their values, so a
    lag-1 tied map has an inductive bias matched to the task surface form" - and it applies here
    for the same reason, which I missed when reading chain results as composition.

    So an ABSOLUTE score on this task flatters a tied map, and "we beat the transformer" cannot be
    read off it. WHAT SURVIVES: adjacency buys one hop, not four. Composing across hops still
    requires S.S, which no surface prior supplies - so a comparison BETWEEN tied arms differing
    only in n_read (NL2 r1 vs NL2 r4) is unaffected, because both enjoy the same gift.

    Fixing it would mean shuffling the two tokens of each edge, or separating source and target
    with a relation token. Not done, because it would also change what the task tests; recorded so
    nobody reads the absolute numbers as composition.
    """
    sep, qry, pad, _ = vocab(nent)
    assert dmode in ('edge', 'chain'), dmode
    ndis = int(round(depth * ratio))
    if dmode == 'chain':
        if ndis % depth:
            raise ValueError('dmode=chain needs a whole-number ratio (depth=%d ratio=%s gives %d '
                             'distractor edges)' % (depth, ratio, ndis))
        nchains = ndis // depth
        need = (depth + 1) * (1 + nchains)
    else:
        need = (depth + 1) + 2 * ndis
    if need > nent:
        return None
    ent = rng.choice(nent, size=need, replace=False)
    path = ent[:depth + 1]
    edges = [(int(path[i]), int(path[i + 1])) for i in range(depth)]
    rest = ent[depth + 1:]
    if dmode == 'chain':
        for c in range(nchains):                      # each distractor is a full depth-long chain
            ch = rest[c * (depth + 1):(c + 1) * (depth + 1)]
            edges += [(int(ch[i]), int(ch[i + 1])) for i in range(depth)]
    else:
        for j in range(ndis):                         # distractors reuse NO path node as source
            edges.append((int(rest[2 * j]), int(rest[2 * j + 1])))
    order = list(range(len(edges)))
    rng.shuffle(order)
    seq = []
    for oi in order:
        a, b = edges[oi]
        seq += [a, b]
    #   dist = how far back the EARLIEST path edge sits. Varies per sample because the shuffle
    #   moves it, which is what lets the median split actually split.
    first_path_pos = min(order.index(i) for i in range(depth))
    dist = len(seq) - 2 * first_path_pos
    seq += [sep, int(path[0]), qry]
    out = _finish(seq, seqlen, pad)
    return (out, int(path[depth]), dist) if out is not None else None


def _chain_edges(seq, pad, sep):
    body = [t for t in seq if t != pad]
    cut = body.index(sep)
    return [(body[i], body[i + 1]) for i in range(0, cut, 2)]


def heuristic_floors(seq, tgt, pad, sep):
    """Expected accuracy of the two query-free heuristics on one make_chain sample.

      sink       guess uniformly among nodes that are never a source
      shortcut   guess among sinks whose source is itself a target (the dmode='edge' leak)

    A score is only evidence of composition above the HIGHER of the two."""
    es = _chain_edges(seq, pad, sep)
    srcs, tgts = {a for a, _ in es}, {b for _, b in es}
    sinks = tgts - srcs
    cut = [b for a, b in es if b in sinks and a in tgts] or list(sinks)
    return (1.0 / len(sinks) if tgt in sinks else 0.0,
            1.0 / len(cut) if tgt in cut else 0.0)


def swap_query(rng, seq, depth, pad, sep):
    """Replace the query's start node with ANOTHER root that also has a depth-hop walk, and return
    (new_seq, new_target), or None if no such root exists (always the case under dmode='edge').

    A model that reads the query must follow the swap; one riding a query-free heuristic keeps
    its old answer. Accuracy against new_target is therefore a direct test of query reading."""
    es = _chain_edges(seq, pad, sep)
    succ = dict(es)
    start = seq[-2]
    roots = [a for a, _ in es if a not in {b for _, b in es} and a != start]
    ok = []
    for r in roots:
        node = r
        for _ in range(depth):
            node = succ.get(node)
            if node is None:
                break
        if node is not None:
            ok.append((r, node))
    if not ok:
        return None
    r, end = ok[int(rng.integers(0, len(ok)))]
    return seq[:-2] + [r, seq[-1]], end


# ------------------------------------------------------------------ self-checks
def check(name, fn, kwargs, nent, trials=400):
    """Assert the task is ANSWERABLE and correctly shaped before any model sees it."""
    sep, qry, pad, V = vocab(nent)
    rng = np.random.default_rng(0)
    dists, fails = [], []
    n_ok = 0
    for _ in range(trials):
        r = fn(rng, **kwargs)
        if r is None:
            fails.append('sequence did not fit seqlen')
            continue
        seq, tgt, d = r
        n_ok += 1
        dists.append(d)
        if len(seq) != kwargs['seqlen']:
            fails.append('wrong length %d' % len(seq))
        if seq[-1] == pad:
            fails.append('FINAL POSITION IS PAD - the readout would see padding')
        if not (0 <= tgt < nent):
            fails.append('target %d outside entity range' % tgt)
        if tgt not in seq:
            fails.append('TARGET NOT PRESENT - task is unanswerable')
        if any(t == pad for t in seq[seq.index(sep):]):
            fails.append('PAD appears AFTER SEP - padding must be on the left only')
        if sep not in seq:
            fails.append('no SEP')
    uniq = len(set(dists))
    md = float(np.median(dists)) if dists else 0.0
    far = [d for d in dists if d > md]
    #   A constant `dist` is NOT a defect here. At ratio 0 the chain contains only path edges,
    #   so the earliest path edge is always at position 0 and the distance cannot vary. The
    #   right response is to say near/far is NOT MEASURED at this cell, not to fail the task -
    #   the failure that matters is an unanswerable or misshapen sequence. (reason_battery
    #   emitted a silent nan here instead, which is what made it quotable as a measurement.)
    split = 'OK (%d far)' % len(far) if far else 'no spread - near/far NOT MEASURED'
    print('  %-34s built %3d/%d  distinct dist %3d  %s'
          % (name, n_ok, trials, uniq, split))
    if fails:
        from collections import Counter
        for msg, c in Counter(fails).most_common(4):
            print('      *** %s  (x%d)' % (msg, c))
    return not fails


if __name__ == '__main__':
    nent = NENT_DEFAULT
    sep, qry, pad, V = vocab(nent)
    print('RECALL TASK SELF-CHECKS  (entities 0..%d, SEP=%d QRY=%d PAD=%d, vocab %d)'
          % (nent - 1, sep, qry, pad, V))
    print('  nothing is trained here - this only proves the tasks are answerable and well formed')
    print()
    ok = True
    print('MQAR - single hop (vocab scales with load: 2*npairs distinct entities needed):')
    for npairs, sl, ne in ((8, 64, 64), (16, 96, 64), (32, 160, 128), (64, 288, 256)):
        ok &= check('npairs=%d seqlen=%d nent=%d' % (npairs, sl, ne), make_mqar,
                    dict(npairs=npairs, seqlen=sl, nent=ne), ne)
    print()
    print('CHAIN - multi-hop, depth and interference INDEPENDENT:')
    for depth in (2, 3, 4):
        for ratio in (0.0, 0.5, 1.0, 2.0):
            ok &= check('depth=%d ratio=%.1f' % (depth, ratio), make_chain,
                        dict(depth=depth, ratio=ratio, seqlen=160, nent=nent), nent)
    print()
    print('CHAIN - query-free heuristic floors, dmode=edge (leaky) vs dmode=chain (fixed):')
    for depth in (2, 3, 4):
        for dmode, ratio in (('edge', 1.0), ('chain', 1.0), ('chain', 2.0)):
            kw = dict(depth=depth, ratio=ratio, seqlen=160, nent=nent, dmode=dmode)
            ok &= check('depth=%d ratio=%.1f dmode=%s' % (depth, ratio, dmode), make_chain, kw, nent)
            rng = np.random.default_rng(0)
            fl, sw_ok, sw_n = [], 0, 0
            for _ in range(1000):
                s, t, _ = make_chain(rng, **kw)
                fl.append(heuristic_floors(s, t, pad, sep))
                sw = swap_query(rng, s, depth, pad, sep)
                if sw is not None:
                    sw_n += 1
                    s2, t2 = sw
                    sw_ok += (t2 != t and len(s2) == len(s) and s2[-1] == qry)
            sink, short = np.mean(fl, axis=0)
            want = 1.0 / (1 + int(ratio)) if dmode == 'chain' else None
            good = (short == 1.0) if dmode == 'edge' else (abs(short - want) < 1e-9 and sw_ok == sw_n == 1000)
            ok &= bool(good)
            print('      sink floor %.3f  SHORTCUT floor %.3f  query swaps valid %d/%d   %s'
                  % (sink, short, sw_ok, sw_n,
                     ('leak present, as documented' if dmode == 'edge' else 'leak closed')
                     if good else '*** UNEXPECTED ***'))
    print()
    print('EXAMPLE (chain, depth 3, ratio 1.0) - read it and confirm it is solvable by hand:')
    r = make_chain(np.random.default_rng(7), 3, 1.0, 64, nent)
    seq, tgt, d = r
    body = [s for s in seq if s != pad]
    print('   tokens after left-pad :', seq[-24:])
    print('   body (pad stripped)   :', body)
    print('   edges                 :', [(body[i], body[i + 1]) for i in range(0, body.index(sep), 2)])
    print('   query start node      :', body[body.index(sep) + 1], ' -> answer', tgt, ' (3 hops)')
    print('   final token is the QRY marker:', seq[-1] == qry, '  dist =', d)
    print()
    print('ALL CHECKS PASSED' if ok else '*** SOME CHECKS FAILED - do not train on this ***')
