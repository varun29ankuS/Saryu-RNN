"""Does a SHARPNESS CONSTRAINT recover composition, and what is the sharpness threshold?

Step 0 showed a near-delta tap composes and a diffuse one does not. A free conv can learn
either. So: sweep tap sharpness continuously and find where composition breaks. If the
usable band is wide, a softmax-with-temperature tap is safe. If it is a knife edge, only a
hard argmax will do - and that changes the design.

Also re-tests the unexplained depth-7 row with a fixed evaluation to see if it survives.
"""
import numpy as np

D, N, TRIALS = 64, 8, 300


def nearest(x, table):
    return int(np.argmax(table @ x / (np.linalg.norm(table, axis=1) * np.linalg.norm(x) + 1e-12)))


def run(w, trials=TRIALS, n=N, d=D):
    acc = {h: 0 for h in (1, 2, 3, 5, 7)}
    valid = {h: 0 for h in acc}
    for _ in range(trials):
        rng = np.random.default_rng()
        phi = rng.normal(size=(n, d)) / np.sqrt(d)
        S = np.zeros((d, d))
        for t in range(n - 1):
            k = np.zeros(d)
            for j, wj in enumerate(w):
                if t - j >= 0:
                    k = k + wj * phi[t - j]
            nk = np.linalg.norm(k)
            if nk < 1e-9:
                continue
            k = k / nk
            v = phi[t + 1]
            S = S - np.outer(S @ k, k) + np.outer(v, k)
        for h in acc:
            if h >= n:
                continue
            x = phi[0]
            for _ in range(h):
                x = S @ x
            nx = np.linalg.norm(x)
            valid[h] += 1
            #   a decayed-to-nothing vector must NOT be scored as a hit by argmax on noise
            if nx > 1e-8:
                acc[h] += (nearest(x, phi) == h)
    return {h: acc[h] / max(valid[h], 1) for h in acc}


def softmax(z, tau):
    z = np.array(z, dtype=float) / tau
    e = np.exp(z - z.max())
    return e / e.sum()


if __name__ == '__main__':
    print('SHARPNESS SWEEP: softmax over 4 taps with logits [1,0,0,0], temperature tau.')
    print('tau -> 0 is a hard one-hot tap; large tau is uniform blending. chance = %.3f' % (1.0 / N))
    print()
    print('  %-8s %-34s %7s %7s %7s %7s' % ('tau', 'resulting tap weights', '1 hop', '2 hops', '3 hops', 'd7'))
    print('  ' + '-' * 76)
    for tau in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.5, 5.0):
        w = softmax([1, 0, 0, 0], tau)
        r = run(w)
        print('  %-8.2f %-34s %7.3f %7.3f %7.3f %7.3f'
              % (tau, '[' + ' '.join('%.3f' % x for x in w) + ']', r[1], r[2], r[3], r[7]))
    print()
    print('  max tap weight is the sharpness measure: 1.000 = pure delta, 0.250 = uniform.')
    print('  READ: the largest tau whose 2-3 hop accuracy stays high is the usable band. A wide')
    print('  band means a temperature-sharpened conv is safe to learn; a knife edge means only a')
    print('  hard argmax tap will do, and the spec must say so.')
