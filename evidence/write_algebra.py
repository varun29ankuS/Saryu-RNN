"""Deposit or disposition? Four write rules at an EQUAL memory budget.

THE QUESTION. Everything in this project so far varied the READ. Two independent routes now say
the READ is not the axis. Fable's review, arguing from CDMA multiuser detection, put it on write
algebra: commutative accumulation (memory, order-free) versus non-commutative composition
(tracking, order-critical). The saryu recurrence is superb at the second and has no mechanism for
the first. Separately, the classical Indian account of memory says the same thing in different
words -- a samskara is not a stored copy placed in a container, it is a DISPOSITIONAL MODIFICATION
of the substrate, reactivated by resemblance. Deposit versus disposition is a claim about the
write.

It is not a foreign import. H(u,beta) = I - beta*u*u^T IS the erase half of a delta rule. Delta
writing is S <- S(I - beta*k*k^T) + beta*v*k^T; we have the first term and our write is + g*c with
c independent of the key. We built half the rule and never built the other half.

THE CONTROL. Every arm gets the SAME NUMBER OF STORED SCALARS, B. A vector arm uses d = B. A
matrix arm uses m x m with m = sqrt(B), so it is far narrower per axis. This is the comparison the
field usually skips: matrix states are assumed better because they hold pairs in separate rows,
but at equal budget the matrix is sqrt(B) wide while the vector is B wide.

    vec-add     h = sum_i k_i (*) v_i                       DEPOSIT, commutative
    vec-delta   h <- h - k (*) (k^* (*) h) + k (*) v         DISPOSITION on a vector: erase what is
                                                            already associated with this key, then
                                                            write. The delta rule in the
                                                            convolution algebra.
    mat-hebb    S <- S + v k^T                               DEPOSIT, commutative (Hebbian)
    mat-delta   S <- S(I - beta k k^T) + beta v k^T          DISPOSITION on a matrix (DeltaNet)

All four are read ONCE and LINEARLY -- no iteration, no oracle, no cleanup beyond argmax over the
value codebook. That is deliberate: this isolates the write.

REGISTERED PREDICTIONS, before running:
  P1  the two DISPOSITION arms beat their DEPOSIT counterparts, and the gap widens with n, because
      deposit accumulates interference from every earlier write while disposition overwrites.
  P2  mat-delta shows a STEP near n = m (keys become orthogonalised, so recall is exact until the
      rows run out), while mat-hebb shows a SLOPE. This one is not ours -- it follows from
      Amit-Gutfreund-Sompolinsky 1985 and the DeltaNet line, and is here as a CALIBRATION that the
      harness reproduces known results.
  P3  UNKNOWN AND THE INTERESTING ONE: at equal budget B, does the wide vector beat the narrow
      matrix? The field assumes matrix states win. A vector of width B has far more room per item
      than a sqrt(B)-wide matrix, so vec-delta may beat mat-delta at equal cost. If it does, the
      cheap architecture is a WIDE VECTOR WITH A DELTA WRITE, not a matrix state.

FALSIFIER for the disposition account: vec-delta ~ vec-add and mat-delta ~ mat-hebb. Then the write
rule is not what matters and the axis is wrong again.

    python evidence/write_algebra.py
Environment: BUDGET, PAIRS, TRIALS, SEED.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

BUDGET = [int(x) for x in os.environ.get('BUDGET', '1024,4096,16384').split(',')]
PAIRS = [int(x) for x in os.environ.get('PAIRS', '2,4,8,16,32,64,128').split(',')]
TRIALS = int(os.environ.get('TRIALS', 200))
SEED = int(os.environ.get('SEED', 0))


def unit(g, *s):
    return F.normalize(torch.randn(*s, generator=g), dim=-1)


def unitary(g, n, d):
    """Unitary HRR keys: unit magnitude at every frequency, so k^* (*) (k (*) v) = v EXACTLY.

    A random unit vector is NOT unitary -- its spectrum has varying magnitude, so correlating
    after convolving applies |K|^2 and distorts rather than cancelling. The delta write needs an
    exact inverse to erase what a key already addresses, so it needs unitary keys (Plate 1995)."""
    ph = torch.rand(n, d // 2 + 1, generator=g) * 2 * math.pi
    spec = torch.exp(1j * ph)
    spec[:, 0] = 1.0 + 0j                       # DC must be real for a real inverse transform
    if d % 2 == 0:
        spec[:, -1] = 1.0 + 0j                  # and Nyquist too, for even d
    return torch.fft.irfft(spec, n=d)


def cconv(a, b, d):
    return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=d)


def ccorr(a, b, d):
    """k^* (*) h -- the exact inverse of binding by k, in the convolution algebra."""
    return torch.fft.irfft(torch.conj(torch.fft.rfft(a)) * torch.fft.rfft(b), n=d)


def vec_add(g, n, B):
    d = B
    K, V = unitary(g, n, d), unit(g, n, d)
    h = torch.zeros(d)
    for i in range(n):
        h = h + cconv(K[i], V[i], d)                       # deposit
    j = int(torch.randint(0, n, (1,), generator=g))
    return int((V @ ccorr(K[j], h, d)).argmax()) == j


def vec_delta_naive(g, n, B):
    """The OBVIOUS vector delta rule, and it is ill-defined. Kept as the negative control.

    With unitary keys the binding inverse is exact, so k (*) (k^* (*) h) = h IDENTICALLY -- the
    'erase' term is the identity, not a projection onto what this key addresses, and it wipes the
    whole state at every write. The delta rule needs k k^T, a RANK-ONE PROJECTOR isolating one
    direction out of m. Convolution binding is full-rank invertible and has no such projector.
    A vector state with invertible binding therefore admits no erase-then-write."""
    d = B
    K, V = unitary(g, n, d), unit(g, n, d)
    h = torch.zeros(d)
    for i in range(n):
        old = cconv(K[i], ccorr(K[i], h, d), d)
        h = h - old + cconv(K[i], V[i], d)
    j = int(torch.randint(0, n, (1,), generator=g))
    return int((V @ ccorr(K[j], h, d)).argmax()) == j


def vec_delta_clean(g, n, B):
    """Disposition on a vector, done properly: erase by RECOGNISING what is stored there.

    Since no projector exists, the only way to know what this key currently addresses is to read
    it and match it against the codebook. That is a nonlinearity -- recognition by resemblance --
    and it is realisable in a real model because the value codebook IS the vocabulary embedding.
    The write becomes: read this address, recognise what sits there, subtract exactly that, write
    the new value."""
    d = B
    K, V = unitary(g, n, d), unit(g, n, d)
    h = torch.zeros(d)
    for i in range(n):
        r = ccorr(K[i], h, d)                               # read this address
        sim = V @ r
        if float(sim.max()) > 0.5:                          # recognised something there
            h = h - cconv(K[i], V[int(sim.argmax())], d)    # subtract exactly that
        h = h + cconv(K[i], V[i], d)                        # then write
    j = int(torch.randint(0, n, (1,), generator=g))
    return int((V @ ccorr(K[j], h, d)).argmax()) == j


def mat_hebb(g, n, B):
    m = max(2, int(math.isqrt(B)))
    K, V = unit(g, n, m), unit(g, n, m)
    S = torch.zeros(m, m)
    for i in range(n):
        S = S + torch.outer(V[i], K[i])                    # deposit
    j = int(torch.randint(0, n, (1,), generator=g))
    return int((V @ (S @ K[j])).argmax()) == j


def mat_delta(g, n, B, beta=1.0):
    m = max(2, int(math.isqrt(B)))
    K, V = unit(g, n, m), unit(g, n, m)
    S = torch.zeros(m, m)
    for i in range(n):
        k = K[i]
        S = S - beta * torch.outer(S @ k, k) + beta * torch.outer(V[i], k)   # erase, then write
    j = int(torch.randint(0, n, (1,), generator=g))
    return int((V @ (S @ K[j])).argmax()) == j


ARMS = [('vec-add', vec_add, 'deposit  '),
        ('vec-delta!', vec_delta_naive, 'ill-defined'),
        ('vec-delta+', vec_delta_clean, 'DISPOSITION'),
        ('mat-hebb', mat_hebb, 'deposit  '), ('mat-delta', mat_delta, 'DISPOSITION')]


def main():
    print(f'WRITE ALGEBRA  equal budget, single LINEAR read, no iteration, {TRIALS} trials')
    print('vector arms are B wide; matrix arms are sqrt(B) x sqrt(B) -- same stored scalars\n')
    for B in BUDGET:
        m = max(2, int(math.isqrt(B)))
        print(f'budget B = {B}   vector width {B}   matrix {m} x {m}')
        print(f'  {"arm":<11} {"kind":<12} ' + ' '.join(f'{"n="+str(n):>6}' for n in PAIRS))
        print('  ' + '-' * (24 + 7 * len(PAIRS)))
        for name, fn, kind in ARMS:
            row = []
            for n in PAIRS:
                g = torch.Generator().manual_seed(SEED)
                row.append(sum(fn(g, n, B) for _ in range(TRIALS)) / TRIALS)
            print(f'  {name:<11} {kind:<12} ' + ' '.join(f'{v:>6.3f}' for v in row), flush=True)
        print()
    print('READ')
    print('  disposition > deposit, widening with n   -> the WRITE is the axis')
    print('  mat-delta a step near n = sqrt(B)        -> harness reproduces the known result')
    print('  vec-delta > mat-delta at equal budget    -> cheap memory is a WIDE VECTOR with a')
    print('                                              delta write, not a matrix state')


if __name__ == '__main__':
    main()
