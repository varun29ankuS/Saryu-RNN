"""Why does cpu_precheck check 2 disagree at 2.7e-09? Triangulate before changing anything.

2.7e-09 is the wrong size for either likely story: a sign or factor error would be O(1), and
float64 round-off over 8 rank-1 updates should be O(1e-15). So something small and systematic
is there. The candidate I can see by reading robust_push against bias_fair.grad:

    numpy wants      -eta * Delta * e / (||e||   + eps)
    robust_push does  eta * Delta * u / (||u||   + eps)   with u = -eta*e, ||u|| = eta*||e||
                   = -eta * Delta * e / (||e|| + eps/eta)

The eps is divided by eta. With eps=1e-12 that is a ~1e-12 relative difference per step, so the
question is whether eight steps amplify it by ~1000x or whether something else is going on.

THREE CELLS SEPARATE THE STORIES:
    1 step,  eps=1e-12   -> if ~1e-12, the per-step difference is exactly the eps scaling
    1 step,  eps=0       -> if ~1e-16, eps is the ONLY difference and the formula is right
    8 steps, eps=0       -> if ~1e-16, there is no compounding bug either; 2.7e-09 is eps
                            amplified by the recursion, and is benign at the real eps of 1e-4
                            only if the same test at eps=1e-4 scales proportionally, which the
                            last row checks.

If cell 2 does NOT come back at round-off, the formula itself is wrong and eps is a red herring.

usage: cd experiments && python diag_check2.py
"""
from __future__ import annotations

import numpy as np
import torch

from bias_fair import grad as np_grad
from parity_block import robust_push

DELTA = 0.3


def run(nsteps, eps, d=16, seed=0):
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(nsteps + 1, d))
    phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    etas = rng.uniform(0.2, 0.9, size=nsteps)

    #   numpy reference, with eps threaded in so the two sides are compared at the SAME eps
    S_np = np.zeros((d, d))
    for t in range(nsteps):
        e = S_np @ phi[t] - phi[t + 1]
        g = e + DELTA * e / (np.linalg.norm(e) + eps)
        S_np = S_np - etas[t] * np.outer(g, phi[t])

    S_t = torch.zeros(1, 1, d, d, dtype=torch.float64)
    for t in range(nsteps):
        k = torch.tensor(phi[t], dtype=torch.float64).view(1, 1, d)
        v = torch.tensor(phi[t + 1], dtype=torch.float64).view(1, 1, d)
        eta = torch.full((1, 1, 1), float(etas[t]), dtype=torch.float64)
        u = eta * v - torch.einsum('bhvk,bhk->bhv', S_t, eta * k)
        u = robust_push(u, eta, DELTA, eps)
        S_t = S_t + u.unsqueeze(-1) * k.unsqueeze(-2)
    return float(np.abs(S_t.numpy().reshape(d, d) - S_np).max())


def run_bias_fair_path(nsteps, eps, d=16, seed=0):
    """Same, but the numpy side calls bias_fair.grad itself - which HARDCODES eps=1e-12."""
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(nsteps + 1, d))
    phi /= np.linalg.norm(phi, axis=1, keepdims=True)
    etas = rng.uniform(0.2, 0.9, size=nsteps)
    S_np = np.zeros((d, d))
    for t in range(nsteps):
        e = S_np @ phi[t] - phi[t + 1]
        S_np = S_np - etas[t] * np.outer(np_grad(e, 'robust', Delta=DELTA), phi[t])
    S_t = torch.zeros(1, 1, d, d, dtype=torch.float64)
    for t in range(nsteps):
        k = torch.tensor(phi[t], dtype=torch.float64).view(1, 1, d)
        v = torch.tensor(phi[t + 1], dtype=torch.float64).view(1, 1, d)
        eta = torch.full((1, 1, 1), float(etas[t]), dtype=torch.float64)
        u = eta * v - torch.einsum('bhvk,bhk->bhv', S_t, eta * k)
        u = robust_push(u, eta, DELTA, eps)
        S_t = S_t + u.unsqueeze(-1) * k.unsqueeze(-2)
    return float(np.abs(S_t.numpy().reshape(d, d) - S_np).max())


if __name__ == '__main__':
    print('DIAGNOSING cpu_precheck check 2   (reported 2.713e-09)')
    print()
    print('  A. eps threaded identically into BOTH sides:')
    print('     %-10s %-12s %s' % ('steps', 'eps', 'max err'))
    for nsteps in (1, 2, 8):
        for eps in (0.0, 1e-12, 1e-4):
            print('     %-10d %-12.0e %.3e' % (nsteps, eps, run(nsteps, eps)), flush=True)

    print()
    print('  B. against bias_fair.grad as-written (its eps is HARDCODED at 1e-12):')
    print('     %-10s %-12s %s' % ('steps', 'torch eps', 'max err'))
    for nsteps in (1, 8):
        for eps in (0.0, 1e-12, 1e-4):
            print('     %-10d %-12.0e %.3e' % (nsteps, eps, run_bias_fair_path(nsteps, eps)), flush=True)

    print()
    print('  READ: if column A at eps=0 is round-off at every depth, the FORMULA is correct and')
    print('  the only discrepancy is where eps sits (divided by eta on the torch side). If A at')
    print('  eps=0 is large, the formula is wrong and eps was never the issue.')
