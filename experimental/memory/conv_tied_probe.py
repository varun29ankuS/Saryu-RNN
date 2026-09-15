"""STEP 0 of SPEC_shift_vs_gdn.md: does a LEARNED lag still compose?

compose_probe showed a HARD shift (v_t = k_{t+1}) composes to any depth while untied k/v
lands at chance past one hop. But a hard lag of 1 is task-specific - stage_c's node stride
is 4, the battery's is 2 - so it is not an architecture. The proposed fix keeps the maps
TIED and lets a depthwise causal conv learn which recent percept becomes the key.

This asks the only question that matters before any training: does a CONVEX BLEND of recent
percepts as the key still give a composable transition operator, or does mixing lags destroy
the algebra? No model, no training, float64, seconds - the same discipline that has paid
every time it was used today.

    untied     k and v from independent maps              what all 8 fla layers do
    shifted    v_t = k_{t+1}, one map                     the hard shift, known to compose
    conv-tied  k_t = sum_j w_j p_{t-j}, v_t = p_t         ONE map, LEARNED lag

usage: python experiments/conv_tied_probe.py
"""
import numpy as np

D, N, TRIALS = 64, 8, 200


def nearest(x, table):
    return int(np.argmax(table @ x / (np.linalg.norm(table, axis=1) * np.linalg.norm(x) + 1e-12)))


def build(mode, rng, w=None, n=N, d=D):
    """write a chain n0 -> n1 -> ... into a delta-rule state at beta=1."""
    phi = rng.normal(size=(n, d)) / np.sqrt(d)
    psi = rng.normal(size=(n, d)) / np.sqrt(d)
    S = np.zeros((d, d))
    for t in range(n - 1):
        if mode == 'untied':
            k, v = phi[t], psi[t + 1]
        elif mode == 'shifted':
            k, v = phi[t], phi[t + 1]
        else:
            #   conv-tied: the KEY is a weighted blend of recent percepts, the VALUE is the
            #   NEXT percept, both through the SAME map. w[0] weights p_t, w[1] p_{t-1}, ...
            #   TWO BUGS FIXED 2026-09-14, found because w=[0,1,0,0] is DEFINITIONALLY the
            #   `shifted` row and scored 0.000 against its 1.000 - an impossibility that could
            #   only be my code:
            #     1. OFF BY ONE. The label said [0,1,0,0] was the hard shift, but w[1] selects
            #        phi[t-1] while v is phi[t+1] - a stride-2 map, not shifted's k=phi[t].
            #        The hard shift is w=[1,0,0,0]. The labels below now match the code.
            #     2. DIVIDE BY ZERO. At small t every (t-j) can be negative, leaving k as the
            #        zero vector; k@k = 0 poisoned S with NaN for the whole chain, which is why
            #        every conv row read 0.000. Skip the write when no tap is in range.
            k = np.zeros(d)
            for j, wj in enumerate(w):
                if t - j >= 0:
                    k = k + wj * phi[t - j]
            nk = np.linalg.norm(k)
            if nk < 1e-9:
                continue
            k = k / nk
            v = phi[t + 1]
        S = S - np.outer(S @ k, k) / (k @ k) + np.outer(v, k) / (k @ k)
    return S, phi, psi


def walk(S, x, h, phi=None, cleanup=False):
    for _ in range(h):
        x = S @ x
        if cleanup:
            x = phi[nearest(x, phi)]
    return x


if __name__ == '__main__':
    print('DOES A LEARNED LAG STILL COMPOSE?  chains of %d, d=%d, %d trials, float64' % (N, D, TRIALS))
    print('  conv weights are over [p_t, p_{t-1}, p_{t-2}, p_{t-3}]; a HARD shift is [0,1,0,0]')
    print()
    CONFIGS = [
        ('untied',    None),
        ('shifted',   None),
        ('conv [1,0,0,0] == shifted', [1.0, 0.0, 0.0, 0.0]),
        ('conv [.9,.1,0,0] mild blend', [0.9, 0.1, 0.0, 0.0]),
        ('conv [.5,.25,.25,0] blurred', [0.5, 0.25, 0.25, 0.0]),
        ('conv [.4,.3,.2,.1] diffuse',  [0.4, 0.3, 0.2, 0.1]),
        ('conv [0,1,0,0] lag-2 stride', [0.0, 1.0, 0.0, 0.0]),
    ]
    print('  %-30s %8s %8s %8s %8s' % ('construction', '1 hop', '2 hops', '3 hops', 'depth 7'))
    print('  ' + '-' * 66)
    for lab, w in CONFIGS:
        mode = lab if lab in ('untied', 'shifted') else 'conv'
        acc = {h: 0 for h in (1, 2, 3, 7)}
        for _ in range(TRIALS):
            rng = np.random.default_rng()
            S, phi, psi = build(mode, rng, w)
            table = psi if mode == 'untied' else phi
            for h in acc:
                if h >= N:
                    continue
                acc[h] += (nearest(walk(S, phi[0], h), table) == h)
        print('  %-30s %8s %8s %8s %8s'
              % (lab, *['%.3f' % (acc[h] / TRIALS) for h in (1, 2, 3, 7)]))
    print()
    print('  READ: 1 hop is retrieval, every construction should manage it. 2+ hops is COMPOSITION.')
    print('  chance is 1/%d = %.3f' % (N, 1.0 / N))
    print('  SELF-CHECK FIRST: the [1,0,0,0] row is DEFINITIONALLY the `shifted` row. If it does')
    print('  not reproduce 1.000, the harness is broken and NO other row may be read.')
    print('  THE QUESTION: does a BLENDED key (what a conv actually learns) still compose, or is')
    print('  composition a property only of an exact one-token shift? If the blends collapse to')
    print('  chance, the learned-lag design is dead and SPEC_shift_vs_gdn.md stops here.')
