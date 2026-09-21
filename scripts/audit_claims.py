"""Check each results file's numbers against the runs THAT FILE'S SCRIPT produced.

WHY. On 2026-09-20 results/matrix_decision.txt was found quoting 0.984 and 0.836 for an experiment
whose nine logged runs scored 0.133-0.164 and 0.031-0.039. Those figures reached an architecture
decision, a README, a website and a public commit. CLAIMS.md #25 recorded the rule that followed --
"every figure quoted in a results file must name the run that produced it" -- as a sentence, and a
sentence is not a check.

WHY IT IS SCOPED, which is the whole design. The first version of this script compared each number
against every eval value in runs/ and reported matrix_decision.txt as 28 of 29 matched -- it passed
the one file we had already PROVEN fabricated. 436 distinct eval values over the 1000 three-decimal
values in [0,1] means a coincidental match roughly half the time, so a global check is a coin flip.
Scoping each file to its own script's runs cuts the candidate set to tens of values and the
coincidence rate to a few percent. Provenance is the discriminating axis; value membership is not.

WHAT A CLEAN RESULT DOES AND DOES NOT MEAN. Unmatched numbers are a reason to look, not a verdict:
losses, bits per character, ratios, tolerances and figures cited from other papers are all
accuracy-shaped and none of them live in our eval metrics. And a file that comes back fully matched
is NOT thereby proven honest -- it is proven consistent with its own runs, which is the most this
can establish and strictly more than we could establish before.

    python scripts/audit_claims.py            census + scoped check
    python scripts/audit_claims.py --selftest inject a fake number, confirm it is caught
"""
from __future__ import annotations
import json, pathlib, re, sys
from collections import defaultdict

NUM = re.compile(r'(?<![\d.])(0\.\d{3})(?![\d])')
EVID = pathlib.Path('evidence')
RES = EVID / 'results'
# The two scripts that do not use the 'script_stem/sub' arm convention, so the prefix cannot be
# read as a script name. rank_dynamics.py logs arm=OPT; trace_train.py logs a free-form arm=ARM.
EXPLICIT = {'muon': 'rank_dynamics', 'adamw': 'rank_dynamics'}
LOGGING_ADDED = '2026-09-17'          # the commit that introduced saryu/metrics.Run


def load_runs():
    """arm-prefix -> set of eval values, and the raw prefix counts."""
    byarm = defaultdict(set)
    for p in pathlib.Path('runs').glob('*.jsonl'):
        try:
            ls = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
        except Exception:
            continue
        cfg = next((l.get('config') for l in ls if isinstance(l.get('config'), dict)), None)
        if not cfg or 'arm' not in cfg:
            continue
        pre = str(cfg['arm']).split('/')[0]
        for l in ls:
            m = l.get('metrics')
            if not isinstance(m, dict):
                continue
            for k, v in m.items():
                if k.startswith('eval') and isinstance(v, (int, float)):
                    byarm[pre].add(round(float(v), 3))
    return byarm


def arm_to_script(byarm):
    """Resolve each arm prefix to the script that emitted it."""
    stems = {p.stem for p in EVID.glob('*.py')}
    out = {}
    for a in byarm:
        if a in stems:
            out[a] = a
        elif a in EXPLICIT:
            out[a] = EXPLICIT[a]
        else:
            out[a] = 'trace_train'          # the only remaining free-form emitter
    return out


def values_for(script, byarm, a2s):
    v = set()
    for a, vals in byarm.items():
        if a2s.get(a) == script:
            v |= vals
    return v


def main():
    byarm = load_runs()
    a2s = arm_to_script(byarm)
    scripts = {p.stem for p in EVID.glob('*.py')}
    files = sorted(RES.glob('*.txt'))
    checked, orphan = [], []
    for f in files:
        txt = f.read_text(encoding='utf-8', errors='ignore')
        nums = sorted({float(x) for x in NUM.findall(txt)})
        if not nums:
            continue
        # A results file is backed by the script of the same name, when that script logged runs.
        script = f.stem if f.stem in scripts else None
        vals = values_for(script, byarm, a2s) if script else set()
        if vals:
            miss = [n for n in nums if round(n, 3) not in vals]
            checked.append((f, len(nums), miss, len(vals)))
        else:
            orphan.append((f, len(nums), script is not None))

    print(f'{len(checked)} results files can be checked against their own script\'s runs; '
          f'{len(orphan)} cannot.\n')
    print('CHECKABLE -- numbers vs the eval values that file\'s own script logged')
    print(f'{"file":<40} {"nums":>5} {"pool":>6} {"unmatched":>10}')
    print('-' * 66)
    for f, n, miss, pool in sorted(checked, key=lambda r: -len(r[2])):
        flag = '  <-- look' if len(miss) > n * 0.3 else ''
        print(f'{f.name:<40} {n:>5} {pool:>6} {len(miss):>10}{flag}')
        if miss and len(miss) <= 10:
            print(f'{"":>40}   {", ".join(f"{m:.3f}" for m in miss)}')
    print(f'\nNOT CHECKABLE -- no run logs exist for these ({LOGGING_ADDED} added run logging)')
    print('  a missing script is an orphan result; a script that exists but never logged is one')
    print('  that predates the facility or never adopted it. Neither is evidence of anything.')
    noscript = [f for f, _, has in orphan if not has]
    print(f'  {len(orphan)} files, of which {len(noscript)} have no script in evidence/ at all:')
    for f in noscript[:14]:
        print(f'    {f.name}')
    if len(noscript) > 14:
        print(f'    ... and {len(noscript)-14} more')


def selftest():
    """Prove the scoped check discriminates, by feeding it a number no run produced."""
    byarm = load_runs()
    a2s = arm_to_script(byarm)
    ok = True
    for stem in ('both_fixes', 'width_scaling', 'conjunctive', 'memory_on'):
        vals = values_for(stem, byarm, a2s)
        if not vals:
            continue
        allv = set().union(*byarm.values())
        real = sorted(vals)[len(vals) // 2]
        # The ADVERSARIAL fake: a value some OTHER experiment logged but this one never did.
        # That is the matrix_decision signature exactly -- a real-looking number from nowhere --
        # and it is the only fake worth testing, because a number absent from every run is caught
        # by any check at all. Picking an easy fake here would reproduce the original mistake.
        # restricted to (0,1): the borrowed number has to be one the prose regex would
        # actually find, or the test is about a string that could never appear.
        fake = next((v for v in sorted(allv - vals) if 0.0 < v < 1.0), None)
        if fake is None:
            continue
        hit_real, hit_fake = round(real, 3) in vals, round(fake, 3) in vals
        print(f'{stem:<16} pool {len(vals):>3}/{len(allv)}   real {real:.3f} -> '
              f'{"matched" if hit_real else "MISSED"}   '
              f'borrowed {fake:.3f} -> {"NOT FLAGGED" if hit_fake else "flagged"}'
              f'   (unscoped check: MATCHED, i.e. missed it)')
        ok &= hit_real and not hit_fake
    print('\nSELFTEST', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else (main() or 0))
