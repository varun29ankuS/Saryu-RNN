"""Recompute, from first principles, every number in paper B that is computable rather than measured.

A measured number (a training accuracy) can only be re-run; a DERIVED number -- the abelian ceiling,
the rung values, the character norms -- is a fact about the group and can be checked exactly. This
project has already published two numbers that turned out to be artefacts, so nothing derived enters
paper B without being reproduced here first.

    python evidence/verify_paperB.py

WHAT IT CHECKS
  1. The abelian ceiling. For a commuting transport the product depends only on the MULTISET of
     inputs, so the best achievable accuracy is obtained by exact enumeration: for every multiset,
     guess the most frequent product. Claimed in evidence/results/v2.txt: 1.0000 on Z_6, 0.3850 on
     S_3, at sequence length 8.
  2. The rung values. Rungs are 1/|N| over the normal subgroups N of G. Normal subgroups are unions
     of conjugacy classes containing e that are closed under the product, which is enumerable.
     Claimed in evidence/results/rungs.txt for S_4, Q_8, A_5.
  3. The character norms. <chi,chi> = (1/|G|) sum |chi(g)|^2 for a d-dimensional representation, and
     the claim in evidence/results/charnorm.txt is that 2.000 means faithful, 3 or 5 means kernel V_4
     and 8 means kernel A_4, for S_4 at d = 4. Those are the only values the multiplicities allow,
     which is checkable by decomposing every 4-dimensional rep of each quotient.
"""
from __future__ import annotations

import itertools
import os
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------- groups as permutations
Perm = tuple


def compose(a: Perm, b: Perm) -> Perm:
    """(a*b)(i) = a(b(i)) -- the convention used for the word problem."""
    return tuple(a[b[i]] for i in range(len(a)))


def sym(n: int) -> list[Perm]:
    return sorted(itertools.permutations(range(n)))


def alt(n: int) -> list[Perm]:
    def even(p):
        return sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p))) % 2 == 0
    return [p for p in sym(n) if even(p)]


def cyclic(n: int) -> list[Perm]:
    return [tuple((i + k) % n for i in range(n)) for k in range(n)]


def quaternion8() -> tuple[list[int], dict]:
    """Q_8 = {+-1, +-i, +-j, +-k} by its multiplication table, as indices 0..7."""
    names = ['1', '-1', 'i', '-i', 'j', '-j', 'k', '-k']
    sign = {'1': 1, '-1': -1}
    base = {'1': '1', 'i': 'i', 'j': 'j', 'k': 'k'}
    tbl = {('1', '1'): '1', ('1', 'i'): 'i', ('1', 'j'): 'j', ('1', 'k'): 'k',
           ('i', '1'): 'i', ('j', '1'): 'j', ('k', '1'): 'k',
           ('i', 'i'): '-1', ('j', 'j'): '-1', ('k', 'k'): '-1',
           ('i', 'j'): 'k', ('j', 'k'): 'i', ('k', 'i'): 'j',
           ('j', 'i'): '-k', ('k', 'j'): '-i', ('i', 'k'): '-j'}

    def split(nm):
        return (-1 if nm.startswith('-') else 1), nm.lstrip('-')

    def mul(x, y):
        sx, bx = split(names[x]); sy, by = split(names[y])
        r = tbl[(bx, by)]
        sr, br = split(r)
        s = sx * sy * sr
        return names.index(br if s > 0 else '-' + br)
    del sign, base
    return list(range(8)), mul


# ---------------------------------------------------------------------------- 1. the abelian ceiling
def abelian_ceiling(elems, mul, length):
    """Exact best accuracy for any function of the MULTISET of inputs.

    Enumerates every word, groups by multiset, and for each multiset takes the most frequent product.
    That is the best a commuting transport can do, by definition, with no training involved."""
    per_multiset = defaultdict(Counter)
    idx = {g: i for i, g in enumerate(elems)}
    for word in itertools.product(range(len(elems)), repeat=length):
        p = elems[word[0]]
        for w in word[1:]:
            p = mul(p, elems[w])
        per_multiset[tuple(sorted(word))][idx[p]] += 1
    total = len(elems) ** length
    best = sum(c.most_common(1)[0][1] for c in per_multiset.values())
    return best / total, len(per_multiset)


# ---------------------------------------------------------------------------- 2. the rung values
def normal_subgroups(elems, mul, inv):
    """Every normal subgroup, as a union of conjugacy classes containing e that is closed."""
    e = next(g for g in elems if all(mul(g, x) == x for x in elems))
    classes = []
    seen = set()
    for g in elems:
        if g in seen:
            continue
        cls = frozenset(mul(mul(x, g), inv(x)) for x in elems)
        classes.append(cls)
        seen |= cls
    ident = next(c for c in classes if e in c)
    rest = [c for c in classes if c is not ident]
    out = []
    for r in range(len(rest) + 1):
        for pick in itertools.combinations(rest, r):
            S = set(ident)
            for c in pick:
                S |= c
            if all(mul(a, b) in S for a in S for b in S):
                out.append(len(S))
    return sorted(out)


# ---------------------------------------------------------------------------- 3. character norms
def s4_irreps():
    """The five irreps of S_4 as (dimension, kernel as an explicit set of permutations).

    The kernels are the four normal subgroups, built here as actual subsets rather than assumed to
    sit in a chain, so the intersection below is computed and not reasoned about."""
    S4 = sym(4)
    e = tuple(range(4))
    V4 = {e, (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)}
    A4 = set(alt(4))
    assert V4 <= A4 <= set(S4), 'the kernels are not nested as assumed'
    return {'trivial': (1, set(S4)), 'sign': (1, A4), '2dim': (2, V4),
            'standard': (3, {e}), 'std*sign': (3, {e})}


def char_norms_at_dim(d=4):
    """Every way to build a d-dimensional rep of S_4, its <chi,chi> = sum m_i^2, and its kernel.

    The kernel of a direct sum is the intersection of the kernels of the parts, so the reachable
    <chi,chi> values split by kernel -- which is the claim the table-free diagnostic rests on."""
    irreps = s4_irreps()
    names = list(irreps)
    out = defaultdict(set)
    for mult in itertools.product(range(d + 1), repeat=len(names)):
        if sum(m * irreps[n][0] for m, n in zip(mult, names)) != d:
            continue
        used = [n for m, n in zip(mult, names) if m]
        if not used:
            continue
        ker = set.intersection(*(irreps[n][1] for n in used))
        out[len(ker)].add(sum(m * m for m in mult))
    return {k: sorted(v) for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------- reading the results
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def src(name):
    """Result files live in evidence/results/ in the public tree and archive/results/ privately."""
    for d in (('evidence', 'results'), ('archive', 'results')):
        q = os.path.join(ROOT, *d, name)
        if os.path.exists(q):
            return q
    raise FileNotFoundError(name)


def src_script():
    """The word-problem experiment: evidence/word_problem.py publicly, archive/ privately."""
    for q in (os.path.join(ROOT, 'evidence', 'word_problem.py'),
              os.path.join(ROOT, 'archive', 'experiments', 'rungs.py')):
        if os.path.exists(q):
            return q
    raise FileNotFoundError('word_problem.py')


def parse_rungs(path):
    """(group, seed, L=12, L=96, flat, |ker|) per run, read from the file itself."""
    rows, grp = [], None
    for line in open(path, encoding='utf-8'):
        if '|G|=' in line:
            grp = line.split()[0]
            continue
        f = line.split()
        if grp and len(f) >= 7 and f[0].isdigit() and f[3] in ('True', 'False'):
            rows.append((grp, int(f[0]), float(f[1]), float(f[2]), f[3] == 'True', float(f[4])))
    return rows


def parse_lattice(path):
    """(raw accuracy, {subgroup: coset score}) per run."""
    cols, rows = ['1 (faithful)', 'V_4', 'A_4', 'S_4'], []
    for line in open(path, encoding='utf-8'):
        f = line.split()
        if len(f) == 8:
            try:
                v = [float(x) for x in f]
            except ValueError:
                continue
            rows.append((v[3], dict(zip(cols, v[4:]))))
    return rows


def parse_quotient(path, task):
    """The L=96 accuracies for one task of quotient_task.txt."""
    out = []
    for line in open(path, encoding='utf-8'):
        f = line.split()
        if len(f) >= 5 and f[0] == task and f[1].isdigit() and f[4] in ('True', 'False'):
            out.append(float(f[3]))
    return out


def main():
    ok = True

    print('1. THE ABELIAN CEILING  (claimed in evidence/results/v2.txt, sequence length 8)')
    print(f'   {"group":>6} {"measured claim":>15} {"exact":>9} {"multisets":>10}')
    for name, elems, claim in (('Z_6', cyclic(6), 1.0000), ('S_3', sym(3), 0.3850)):
        acc, nm = abelian_ceiling(elems, compose, 8)
        agree = abs(acc - claim) < 5e-4
        ok &= agree
        print(f'   {name:>6} {claim:>15.4f} {acc:>9.4f} {nm:>10}   {"ok" if agree else "MISMATCH"}')

    print('\n2. THE RUNG VALUES  (claimed in archive/results/rungs.txt)')
    q_el, q_mul = quaternion8()
    q_inv = lambda x: next(y for y in q_el if q_mul(x, y) == 0)  # noqa: E731
    p_inv = lambda p: tuple(sorted(range(len(p)), key=lambda i: p[i]))  # noqa: E731
    claims = {
        'S_4': ([1, 4, 12, 24], sym(4), compose, p_inv),
        'Q_8': ([1, 2, 4, 4, 4, 8], q_el, q_mul, q_inv),
        'A_5': ([1, 60], alt(5), compose, p_inv),
    }
    for name, (claim_orders, elems, mul, inv) in claims.items():
        orders = normal_subgroups(elems, mul, inv)
        rungs = sorted({round(1 / o, 3) for o in orders})
        agree = orders == sorted(claim_orders)
        ok &= agree
        print(f'   {name:>4} |G|={len(elems):>2}  normal subgroup orders {orders}')
        print(f'          rungs 1/|N| = {rungs}   {"ok" if agree else "MISMATCH vs " + str(claim_orders)}')

    print('\n3. CHARACTER NORMS AT d = 4  (claimed in archive/results/charnorm.txt:')
    print('   2.000 = faithful, 3 or 5 = kernel V_4, 8 = kernel A_4)')
    got = char_norms_at_dim(4)
    label = {1: 'faithful', 4: 'kernel V_4', 12: 'kernel A_4', 24: 'kernel S_4'}
    for ker, vals in got.items():
        print(f'   kernel order {ker:>2} ({label.get(ker, "?"):>10}):  <chi,chi> in {vals}')
    checks = [(1, 2, 'faithful gives 2'), (4, 3, 'ker V_4 gives 3'), (4, 5, 'ker V_4 gives 5'),
              (12, 8, 'ker A_4 gives 8')]
    for ker, val, why in checks:
        good = val in got.get(ker, [])
        ok &= good
        print(f'     {why:<22} {"ok" if good else "MISMATCH"}')
    uniq = [v for v in got.get(1, []) if all(v not in got[k] for k in got if k != 1)]
    print(f'   values unique to a faithful rep, at FULL rank d = 4: {uniq}  '
          f'-> 2 is {"diagnostic" if uniq == [2] else "NOT uniquely diagnostic"}')
    print('   but a trained transport can let a dimension die, so the honest question is what')
    print('   <chi,chi> = 2 implies over ALL dimensions up to 4:')
    for dd in (1, 2, 3, 4):
        g2 = char_norms_at_dim(dd)
        who = [f'{label.get(k, k)}' for k, v in g2.items() if 2 in v]
        print(f'     d={dd}: <chi,chi> = 2 is reachable with kernel {who or "(not reachable)"}')
    allk = {k for dd in (1, 2, 3, 4) for k, v in char_norms_at_dim(dd).items() if 2 in v}
    print(f'   => across d <= 4, <chi,chi> = 2 admits kernels {sorted(allk)}: it is diagnostic')
    print('      ONLY if the representation is known to be full rank. Reported as conditional.')

    print('\n4. THE MEASURED ACCURACIES AGAINST THOSE RUNGS  (parsed from the result files, not')
    print('   retyped, so the transcription is checked along with the claim)')
    rows = parse_rungs(src('rungs.txt'))
    worst = 0.0
    bad_ker = 0
    for grp, sd, l12, l96, flat, ker in rows:
        orders = normal_subgroups(*claims[grp][1:])
        rungs = sorted({1 / o for o in orders})
        dev = min(abs(l96 - r) for r in rungs)
        worst = max(worst, dev)
        legal = ker in orders
        bad_ker += not legal
        print(f'   {grp:>4} sd{sd}  L=12 {l12:.3f}  L=96 {l96:.3f}  flat {flat!s:>5}  '
              f'nearest rung {min(rungs, key=lambda r: abs(l96 - r)):.3f}  dev {dev:.3f}   '
              f'|ker|={ker:g} {"" if legal else "<- NOT a normal subgroup order"}')
    ok &= worst < 0.03
    print(f'   worst deviation from a rung of the run\'s own group: {worst:.3f} over {len(rows)} runs')
    print(f'   rows whose reported |ker| is not a normal subgroup order: {bad_ker}/{len(rows)}'
          '  <- that column is unreliable and is not used')
    a5 = [r[3] for r in rows if r[0] == 'A_5']
    strictly_between = [a for a in a5 if 0.017 + 0.03 < a < 1.000 - 0.03]
    ok &= not strictly_between
    print(f'   registered A_5 falsifier (a flat accuracy strictly between 0.017 and 1.000): '
          f'{len(strictly_between)} hits -> {"not falsified" if not strictly_between else "FALSIFIED"}')

    print('\n5. COSET CONSISTENCY PICKS THE SAME N THE ACCURACY DOES')
    lat = parse_lattice(src('lattice.txt'))
    sizes = {'1 (faithful)': 1, 'V_4': 4, 'A_4': 12, 'S_4': 24}
    agree = 0
    for raw, scores in lat:
        n = next((sizes[k] for k in ['1 (faithful)', 'V_4', 'A_4', 'S_4'] if scores[k] > 0.99), None)
        pred = 1 / n if n else float('nan')
        good = n is not None and abs(raw - pred) < 0.03
        agree += good
        print(f'   raw {raw:.3f}   smallest N with coset ~1.000: |N|={n:>2}   predicts 1/|N|={pred:.3f}'
              f'   {"ok" if good else "MISMATCH"}')
    ok &= agree == len(lat)
    print(f'   {agree}/{len(lat)} runs: the subgroup identified by the ERRORS predicts the accuracy')

    print('\n6. THE LATTICE LAW APPLIES TO THE QUOTIENT TOO')
    print('   On the V_4 task the target is the V_4-coset, so the group to be learned is')
    print('   S_4/V_4 = S_3; on the A_4 task it is S_4/A_4 = Z_2. Their own rungs should then')
    print('   govern the failures -- the law applied one level down.')
    for task, grp, elems, inv in (('V_4', 'S_3', sym(3), p_inv), ('A_4', 'Z_2', cyclic(2), p_inv)):
        orders = normal_subgroups(elems, compose, inv)
        rungs = sorted({1 / o for o in orders})
        accs = parse_quotient(src('quotient_task.txt'), task)
        print(f'   task {task}: quotient {grp}, normal subgroup orders {orders}, '
              f'rungs {[round(r, 3) for r in rungs]}')
        for a in accs:
            dev = min(abs(a - r) for r in rungs)
            ok &= dev < 0.03
            print(f'     measured {a:.3f} -> nearest quotient rung '
                  f'{min(rungs, key=lambda r: abs(a - r)):.3f}  dev {dev:.3f}'
                  f'   {"ok" if dev < 0.03 else "MISMATCH"}')

    print('\n7. WHY THE REPORTED |ker| COLUMN IS NOISE')
    print('   word_problem.py estimates the kernel by clustering  M[g] = mean over random h of T(h,g).')
    print('   T is LINEAR in h and h is zero-mean, so M[g] is a Monte-Carlo zero for every g:')
    print('   the clustering then runs on sampling noise, which is why it returns orders that')
    print('   are not normal subgroup orders at all.')
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'rungs_exp', src_script())
        rx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rx)
        import torch
        torch.manual_seed(0)
        N, d = 24, 4
        m = rx.Model(N, d)
        with torch.no_grad():
            h = torch.randn(400, d)
            M = torch.stack([m.T(h, torch.full((400,), g)).mean(0) for g in range(N)])
            typ = torch.stack([m.T(h, torch.full((400,), g)).norm(dim=-1).mean()
                               for g in range(N)]).mean()
        print(f'   ||mean_h T(h,g)|| averaged over g : {M.norm(dim=-1).mean():.4f}')
        print(f'   mean_h ||T(h,g)||   averaged over g : {typ:.4f}')
        print(f'   ratio {M.norm(dim=-1).mean() / typ:.3f}  -- the signal the clustering sees is')
        print(f'   {M.norm(dim=-1).mean() / typ * 100:.1f}% of the transport it is meant to measure;')
        print(f'   the threshold used is 0.05 on distances scaled by sqrt(d).')
        ks, nc = rx.kernel_size(m, N)
        print(f'   on an UNTRAINED model, where all {N} transports are independent random draws')
        print(f'   and the true kernel is trivial, the estimator reports |ker| = {ks:.1f} '
              f'({nc} clusters)')
        ok &= True
    except Exception as exc:                                  # the diagnosis is not load-bearing
        print(f'   [could not import the archived script: {exc}]')

    print('\nVERDICT:', 'all derived numbers reproduce' if ok else 'AT LEAST ONE NUMBER DOES NOT REPRODUCE')


if __name__ == '__main__':
    main()
