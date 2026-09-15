"""DOES CHANGING THE ATTENTIONAL BIAS BREAK COMPOSITION?  MIRAS axis 2, on our matrix state.

MIRAS (2504.13173) says the field has only ever used dot-product or l2 objectives (their Remark 4),
and offers alternatives. Crucially it derives all of them FOR A MATRIX MEMORY - we are not giving
up the matrix:
    Eq 11  l_p :   W <- W - eta * p * (Sign(Wk - v) . |Wk - v|^(p-1)) k^T
    Eq 14  Huber:  l2 gradient where the error is small, Sign gradient where it is large
    Eq 17  robust: l2 gradient plus a normalised term, worst-case value perturbation

Our objective has always been plain l2: W <- W - eta (Wk - v) k^T, which is the delta rule.

THE QUESTION THAT DECIDES WHETHER WE CAN STACK THEM: the tied map's whole property is that S
becomes a TRANSITION OPERATOR, so S^d walks d hops. That property comes from the l2 delta rule
exactly replacing the value at a key. If an l1 or Huber update no longer replaces cleanly, the
composition dies and the two ideas are incompatible.

Pure numerics, no training. Baseline (l2, eta=1) must reproduce the known 1.000, or nothing below
is readable.

REFERENCES
  [Behrouz et al. 2025] It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization, arXiv 2504.13173
"""
import numpy as np

D, N, TRIALS = 64, 8, 200


def nearest(x, table):
    return int(np.argmax(table @ x / (np.linalg.norm(table, axis=1) * np.linalg.norm(x) + 1e-12)))


def walk(S, x, h):
    for _ in range(h):
        x = S @ x
        n = np.linalg.norm(x)
        if n < 1e-10:
            return None
        x = x / n
    return x


def build(rng, bias, eta=1.0, p=1.0, delta=0.5, n=N, d=D):
    """TIED chain written into a MATRIX state under different attentional biases."""
    phi = rng.normal(size=(n, d)); phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    S = np.zeros((d, d))
    for t in range(n - 1):
        k, v = phi[t], phi[t + 1]
        e = S @ k - v                                    # the error the objective acts on
        if bias == 'l2':
            g = e                                        # delta rule: exact replacement at eta=1
        elif bias == 'lp':
            g = p * np.sign(e) * np.abs(e) ** (p - 1)    # MIRAS Eq 11
        elif bias == 'l1':
            g = np.sign(e)                               # MIRAS Eq 12, "value-less" memory
        elif bias == 'huber':
            small = np.abs(e) <= delta                   # MIRAS Eq 14
            g = np.where(small, e, delta * np.sign(e))
        elif bias == 'robust':
            nrm = np.linalg.norm(e) + 1e-12              # MIRAS Eq 17 update
            g = e + 0.3 * e / nrm
        S = S - eta * np.outer(g, k)
    return S, phi


def score(bias, eta=1.0, **kw):
    acc = {h: 0 for h in (1, 2, 3, 7)}
    for _ in range(TRIALS):
        rng = np.random.default_rng()
        S, phi = build(rng, bias, eta, **kw)
        for h in acc:
            if h >= N:
                continue
            got = walk(S, phi[0], h)
            acc[h] += (got is not None and nearest(got, phi) == h)
    return {h: acc[h] / TRIALS for h in acc}


if __name__ == '__main__':
    print('ATTENTIONAL BIAS vs COMPOSITION   matrix state, tied map, d=%d, chance %.3f' % (D, 1.0 / N))
    print('MIRAS derives ALL of these for a MATRIX memory - the matrix is not being abandoned.')
    print()
    print('  %-34s %8s %8s %8s %8s' % ('attentional bias', '1 hop', '2 hops', '3 hops', 'depth 7'))
    print('  ' + '-' * 70)
    b = score('l2', 1.0)
    print('  %-34s %8.3f %8.3f %8.3f %8.3f' % ('l2, eta=1 (delta rule, BASELINE)', b[1], b[2], b[3], b[7]))
    if b[1] < 0.99:
        raise SystemExit('  *** baseline broken - nothing below is readable ***')
    for lab, bias, kw in [
        ('l2, eta=0.5 (partial replacement)', 'l2', dict(eta=0.5)),
        ('l_p, p=1.5', 'lp', dict(p=1.5)),
        ('l_p, p=3.0', 'lp', dict(p=3.0)),
        ('l1 / Sign (value-less memory)', 'l1', dict(eta=0.5)),
        ('Huber delta=0.5', 'huber', dict(delta=0.5)),
        ('Huber delta=0.1 (mostly l1)', 'huber', dict(delta=0.1)),
        ('robust (Eq 17)', 'robust', {}),
    ]:
        eta = kw.pop('eta', 1.0)
        r = score(bias, eta, **kw)
        print('  %-34s %8.3f %8.3f %8.3f %8.3f' % (lab, r[1], r[2], r[3], r[7]))
    print()
    print('  READ: l2 at eta=1 is the ONLY update that exactly replaces the value at a key, which is')
    print('  what makes S a clean transition operator. Any bias that under- or over-shoots leaves')
    print('  residue at the key, and residue compounds when you walk S^d.')
    print('  If the alternatives hold composition, MIRAS axis 2 and our tied map STACK.')
    print('  If they collapse, we must choose one - and that is worth knowing before building.')
