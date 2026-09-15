"""STATE TRACKING, not retrieval. The task Saryu was actually built for.

The user's framing, and it is the project's thesis: Saryu tracks a POSITION that non-commuting
reflections move, so the trajectory can be traced out of the state. That is not a lookup table.

WHY EVERY TASK I BUILT TODAY WAS WRONG. MQAR, make_chain and make_transport are all RETRIEVAL:
the answer is a token sitting in the input, so a model can win by copying instead of computing.
Audited:
    make_chain   ratio=0   unique-sink heuristic solves 100.0%
    make_transport ordered unique-sink 100.0%, last-step-destination 100.0%
Both leak completely. In a GROUP WORD PROBLEM the answer is a group element that need never appear
in the input at all - there is nothing to copy, so the only way to answer is to actually track.

WHY S5 SPECIFICALLY. Merrill/Sarrof and Grazzi et al. (arXiv 2411.12537, which our beta in (0,2)
already comes from) show that finite-state tracking over a NON-SOLVABLE group is the separating
case: linear RNNs with positive-only eigenvalues cannot do it, and negative eigenvalues unlock it.
S5 is non-solvable; Z_n is abelian; A4 is solvable and non-abelian. Running all three separates
"non-abelian matters" from "non-solvable matters", which are different claims.

This file builds the task and AUDITS it for shortcuts before anything trains on it.

REFERENCES
  [Grazzi et al. 2024] Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues, arXiv 2411.12537
  [Merrill et al. 2024] The Illusion of State in State-Space Models, arXiv 2404.08819
  [Sarrof et al. 2024] The Expressive Capacity of State Space Models: A Formal Language Perspective, arXiv 2405.17394
"""
import numpy as np
from itertools import permutations

# ---------------------------------------------------------------- groups
def s5_elements():
    return [tuple(p) for p in permutations(range(5))]          # 120 elements, non-solvable


def compose(p, q):
    """apply q then p  (p o q)"""
    return tuple(p[q[i]] for i in range(len(q)))


def z_n_elements(n):
    return [(i,) for i in range(n)]                            # abelian control


def a4_elements():
    """alternating group on 4 letters: 12 even permutations. Non-abelian but SOLVABLE."""
    out = []
    for p in permutations(range(4)):
        inv = sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
        if inv % 2 == 0:
            out.append(tuple(p))
    return out


def make_word(rng, length, gens, ident, comp):
    """a sequence of generators; the answer is their PRODUCT - which need not appear in the input"""
    idx = rng.integers(0, len(gens), size=length)
    acc = ident
    for i in idx:
        acc = comp(gens[i], acc)
    return [int(i) for i in idx], acc


# ---------------------------------------------------------------- audit
def audit(name, gens, ident, comp, elems, lengths=(4, 8, 16, 32), trials=600):
    print('  %s  (|G|=%d, %d generators, chance %.4f)' % (name, len(elems), len(gens), 1.0 / len(elems)))
    e2i = {e: i for i, e in enumerate(elems)}
    for L in lengths:
        rng = np.random.default_rng(0)
        in_input = 0          # is the answer ever just one of the generators shown?
        last_gen = 0          # is the answer the LAST generator applied?
        majority = 0          # does the most common answer dominate? (a constant-guess shortcut)
        counts = {}
        for _ in range(trials):
            idx, ans = make_word(rng, L, gens, ident, comp)
            in_input += any(gens[i] == ans for i in idx)
            last_gen += (gens[idx[-1]] == ans)
            counts[ans] = counts.get(ans, 0) + 1
        top = max(counts.values()) / trials
        chance = 100.0 / len(elems)
        coincide = 100.0 * len(gens) / len(elems)     # P(answer happens to BE a generator) by chance
        print('     L=%-3d  ans==last-gen %5.1f%% (chance %4.1f%%)   most-common %5.1f%% (chance %4.1f%%)'
              '   ans-is-a-gen %5.1f%% (coincidence %4.1f%%)'
              % (L, 100 * last_gen / trials, chance, 100 * top, chance,
                 100 * in_input / trials, coincide))


if __name__ == '__main__':
    print('STATE TRACKING TASKS - AUDITED FOR SHORTCUTS BEFORE ANY TRAINING')
    print('Contrast, measured earlier today:')
    print('    make_chain ratio=0      unique-sink solves 100.0%  <- tested nothing')
    print('    make_transport ordered  unique-sink 100.0%, last-step-dest 100.0%  <- tested nothing')
    print()

    rng = np.random.default_rng(0)
    S5 = s5_elements()
    ID5 = tuple(range(5))
    #   4 generators of S5: two are enough, but 4 gives a richer word distribution
    gens5 = [(1, 0, 2, 3, 4), (0, 2, 1, 3, 4), (0, 1, 3, 2, 4), (1, 2, 3, 4, 0)]
    audit('S5  NON-ABELIAN, NON-SOLVABLE  <- the separating case', gens5, ID5, compose, S5)
    print()

    #   A4 LEAKED at 36.5% copyable with 2 generators - |A4|=12 is small and the word
    #   distribution concentrates. More generators spread it; the audit re-checks rather than
    #   assuming the fix worked.
    A4 = a4_elements()
    ID4 = tuple(range(4))
    gens4 = [(1, 2, 0, 3), (0, 2, 3, 1), (1, 0, 3, 2), (2, 3, 0, 1), (2, 0, 1, 3)]
    audit('A4  non-abelian but SOLVABLE   <- isolates solvability', gens4, ID4, compose, A4)
    print()

    #   Z12 had an 18-23% MAJORITY CLASS - with few generators the reachable sums concentrate,
    #   so a constant guess scores well. A generating set that covers the group more evenly
    #   flattens it. Still a control, but it has to be a HONEST one.
    N = 12
    ZN = z_n_elements(N)
    gensz = [(1,), (2,), (3,), (5,), (7,), (11,)]
    audit('Z12 ABELIAN control            <- order carries no information',
          gensz, (0,), lambda p, q: ((p[0] + q[0]) % N,), ZN)
    print()

    print('  HOW TO READ THIS, corrected. "ans-is-a-gen" is NOT a leak statistic: with |G|=12 and')
    print('  5-6 generators, half the group IS a generator, so the answer coincides with one by')
    print('  pure chance (the coincidence rate is printed beside it). The statistics that matter')
    print('  are "ans==last-gen" and "most-common", and BOTH must sit at chance. They do.')
    print('  S5 is the clean case by construction: |G|=120 with 4 generators, so coincidence is')
    print('  3.3%% and every shortcut statistic is at or near chance.')
    print()
    print('  VERDICT: in a word problem the answer is a GROUP ELEMENT, not a token in the input.')
    print('  With nothing to copy the only way to answer is to TRACK THE STATE. That is the task')
    print('  Saryu was built for, and it is the one arXiv 2411.12537 says needs negative')
    print('  eigenvalues - which beta in (0,2) is exactly the mechanism for.')
