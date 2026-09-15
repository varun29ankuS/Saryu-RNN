"""THE FAIR COMPARISON: each attentional bias at ITS OWN best learning rate.

WHY THE PREVIOUS RUN WAS INVALID, found by reading my own code rather than the numbers:
    g for l1 is Sign(e), whose norm is sqrt(d) = 8.0 at d=64.
    g for l2 is e,       whose norm is 1.0.
I applied the SAME eta to both. l1 therefore took an 8x larger step, the state norm reached
9.53 against l2's 2.59, and I recorded "l1 diverges - the update is unstable at eta=1".
That was MY CONSTANT eta, not the objective. MIRAS's eta_t is data-dependent and learnable
(their Eq 5); fixing it at 1 across objectives whose gradients differ 8x in scale is not a
comparison.

Gradient norms at a unit-norm error, measured:
    l2 1.000 | huber d=0.5 1.000 | huber d=0.1 0.628 | robust 1.300 | lp p=1.5 3.833 | l1 8.000

So: sweep eta per bias, report each at its OWN optimum, and use the graded readout (residue,
state norm, cosine) that caught the previous artefact.

STILL NON-EVIDENCE whatever eta does: Huber d=0.5 is bit-identical to l2, because a unit-norm
error has per-coordinate |e_i| ~ 0.104 and never reaches delta=0.5. No learning rate changes that.

REFERENCES
  [Behrouz et al. 2025] It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization, arXiv 2504.13173
"""
import numpy as np

D, N, TRIALS = 64, 8, 150


def grad(e, bias, p=1.0, delta=0.5, Delta=0.3):
    if bias == 'l2':
        return e
    if bias == 'lp':
        return p * np.sign(e) * np.abs(e) ** (p - 1)
    if bias == 'l1':
        return np.sign(e)
    if bias == 'huber':
        return np.where(np.abs(e) <= delta, e, delta * np.sign(e))
    if bias == 'robust':
        return e + Delta * e / (np.linalg.norm(e) + 1e-12)
    raise ValueError(bias)


def run(bias, eta, trials=TRIALS, **kw):
    res, nrm, c1, c3 = [], [], [], []
    for _ in range(trials):
        rg = np.random.default_rng()
        phi = rg.normal(size=(N, D)); phi /= np.linalg.norm(phi, axis=1, keepdims=True)
        S = np.zeros((D, D))
        for t in range(N - 1):
            k, v = phi[t], phi[t + 1]
            S = S - eta * np.outer(grad(S @ k - v, bias, **kw), k)
        res.append(np.linalg.norm(S @ phi[N - 2] - phi[N - 1]))
        nrm.append(np.linalg.norm(S))
        for h, acc in ((1, c1), (3, c3)):
            x = phi[0].copy(); ok = True
            for _ in range(h):
                x = S @ x
                n = np.linalg.norm(x)
                if n < 1e-10:
                    ok = False; break
                x = x / n
            acc.append(float(x @ phi[h]) if ok else 0.0)
    return dict(res=np.mean(res), nrm=np.mean(nrm), c1=np.mean(c1), c3=np.mean(c3))


if __name__ == '__main__':
    ETAS = [0.03, 0.06, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5]
    CFG = [('l2', 'l2', {}), ('huber d=0.5', 'huber', dict(delta=0.5)),
           ('huber d=0.1', 'huber', dict(delta=0.1)), ('robust D=0.3', 'robust', {}),
           ('lp p=1.5', 'lp', dict(p=1.5)), ('lp p=3.0', 'lp', dict(p=3.0)),
           ('l1 (Sign)', 'l1', {})]
    print('EACH BIAS AT ITS OWN BEST LEARNING RATE   d=%d, chains of %d, %d trials' % (D, N, TRIALS))
    print('previous run used eta=1 for ALL of them, which handed l1 an 8x larger step')
    print()
    print('  %-14s %-8s %-10s %-9s %-9s %-9s' % ('bias', 'best eta', 'residue', '||S||', 'cos 1hop', 'cos 3hop'))
    print('  ' + '-' * 66)
    best = {}
    for lab, bias, kw in CFG:
        rows = [(run(bias, e, **kw), e) for e in ETAS]
        r, e = max(rows, key=lambda t: t[0]['c3'])          # pick by 3-hop composition
        best[lab] = (e, r)
        print('  %-14s %-8.3f %-10.3f %-9.2f %-9.3f %-9.3f' % (lab, e, r['res'], r['nrm'], r['c1'], r['c3']), flush=True)
    print()
    print('  FULL SWEEP for the two I previously called divergent:')
    for lab, bias, kw in [('l1 (Sign)', 'l1', {}), ('lp p=1.5', 'lp', dict(p=1.5))]:
        print('   %s' % lab)
        for e in ETAS:
            r = run(bias, e, **kw)
            print('     eta=%-6.3f residue %-8.3f ||S|| %-7.2f cos1 %-7.3f cos3 %.3f'
                  % (e, r['res'], r['nrm'], r['c1'], r['c3']), flush=True)
    b = best['l2'][1]['c3']
    print()
    print('  l2 baseline 3-hop cosine: %.3f' % b)
    for lab in best:
        if lab != 'l2':
            d = best[lab][1]['c3'] - b
            print('   %-14s %+.3f  %s' % (lab, d, 'holds' if d > -0.05 else 'genuinely worse'))
