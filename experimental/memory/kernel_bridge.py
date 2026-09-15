"""CAN WE JUST USE fla's KERNELS? Written on CPU, runs on GPU, costs minutes not hours.

I have been calling our missing kernel "engineering debt" as though it were a fixed cost. fla is
MIT-licensed, has Triton kernels for the whole DeltaNet family, and was verified working on a T4
earlier in this project. The question is whether our block can call them, and that is testable.

THE ARGUMENT THAT IT SHOULD WORK: a chunkwise delta-rule kernel takes (q, k, v, beta, g) and runs
the recurrence. It does not know or care where k and v CAME from. Tying is a property of how we
BUILD k and v - a lag-weighted sum over the address stream - and that happens before the call,
elementwise and in parallel, with no recurrence of its own. Same for FHRR binding and the phase
rotation. So the incumbent kernel should run our variant unchanged.

That is not a new hope: chunkwise parallelism was already verified to survive tying at 3.33e-16
earlier in this project. This file turns that into a working call.

WHAT MAPS, AND WHAT DOES NOT
  bind / rotate / tap      elementwise and parallel, pre-kernel. Free.
  the delta recurrence     fla's chunk_(gated_)delta_rule, unchanged.
  n_read = 2               TWO kernel passes. The kernel returns o_t = S_t q_t for EVERY t, so
                           feeding o back in as the query yields S_t (S_t q_t). Costs 2x the
                           kernel, against L sequential python steps. Wasteful and still a rout.
  the reflection tracker   the only genuinely sequential piece. A product of Householders has a
                           WY form, prod(I - 2 u u^T) = I - U T U^T with T triangular, which is
                           the SAME UT transform fla already uses internally. Not attempted here;
                           this file measures whether it is the bottleneck before anyone builds it.

WHY THIS IS CHEAP: forward passes and a correctness comparison, no training. If it passes, the
real run gets 10-100x faster, so testing the kernel makes the main run MORE affordable, not less.

DEFENSIVE ABOUT THE API ON PURPOSE: fla's signatures have moved between versions and this cannot
be checked from CPU. The script INTROSPECTS what is available and reports, rather than assuming a
signature and failing with a traceback that looks like the idea is wrong.

usage: python kernel_bridge.py          # on a GPU box; on CPU it reports what it cannot do

REFERENCES
  [Yang et al. 2024a] Parallelizing Linear Transformers with the Delta Rule over Sequence Length, arXiv 2406.06484
  [Yang et al. 2024b] Gated Delta Networks: Improving Mamba2 with Delta Rule, arXiv 2412.06464
"""
from __future__ import annotations

import inspect
import time

import torch
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def report_env():
    print('torch %s   cuda %s   device %s' % (torch.__version__, torch.cuda.is_available(), DEV))
    if DEV == 'cuda':
        print('gpu: %s' % torch.cuda.get_device_name(0))
        cap = torch.cuda.get_device_capability(0)
        print('capability %s  -> bf16 tensor cores: %s (Turing/T4 is 7.5 = fp16 only)'
              % (cap, cap[0] >= 8))
    try:
        import fla
        print('fla %s' % getattr(fla, '__version__', 'unknown'))
    except Exception as e:                                    # noqa: BLE001
        print('fla NOT importable: %s' % e)
        return None
    ops = {}
    for name in ('chunk_delta_rule', 'fused_recurrent_delta_rule'):
        try:
            m = __import__('fla.ops.delta_rule', fromlist=[name])
            ops[name] = getattr(m, name)
        except Exception as e:                                # noqa: BLE001
            print('  %s unavailable: %s' % (name, e))
    for name in ('chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule'):
        try:
            m = __import__('fla.ops.gated_delta_rule', fromlist=[name])
            ops[name] = getattr(m, name)
        except Exception as e:                                # noqa: BLE001
            print('  %s unavailable: %s' % (name, e))
    for k, f in ops.items():
        try:
            print('  %-34s %s' % (k, inspect.signature(f)))
        except (ValueError, TypeError):
            print('  %-34s <no signature>' % k)
    return ops


# ------------------------------------------------------------------ our reference
def seq_delta(q, k, v, beta, g=None, n_read=1):
    """The sequential recurrence SaryuBlock runs, as the thing the kernel must reproduce.
    q,k,v: (B,L,H,D)   beta: (B,L,H)   g: (B,L,H) log-decay or None"""
    B, L, H, D = q.shape
    S = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    outs = []
    for t in range(L):
        kt, vt, qt = k[:, t].float(), v[:, t].float(), q[:, t].float()
        bt = beta[:, t].float().unsqueeze(-1)
        if g is not None:
            S = torch.exp(g[:, t].float()).unsqueeze(-1).unsqueeze(-1) * S
        u = vt * bt - torch.einsum('bhvk,bhk->bhv', S, kt * bt)
        S = S + u.unsqueeze(-1) * kt.unsqueeze(-2)
        r = qt
        for _ in range(n_read):
            r = torch.einsum('bhvk,bhk->bhv', S, r)
            if n_read > 1:
                r = F.normalize(r, dim=-1)
        outs.append(r)
    return torch.stack(outs, 1)


def tied_kv(a, tap, ntap=4):
    """Build the TIED key stream from the address stream. Pure elementwise + shift: this is the
    whole of `tying` as far as any kernel is concerned, and it is parallel."""
    L = a.shape[1]
    w = F.softmax(tap, dim=0)
    k = torch.zeros_like(a)
    for j in range(ntap):
        k = k + w[j] * F.pad(a, (0, 0, 0, 0, j + 1, 0))[:, :L]
    return F.normalize(k, dim=-1), a


def call_kernel(fn, q, k, v, beta, g):
    """Try the plausible calling conventions and return (out, how). Signatures move between
    versions, so this reports which one worked instead of hard-coding one."""
    attempts = []
    if g is not None:
        attempts.append(('g=,head_first=False', dict(g=g, head_first=False)))
        attempts.append(('g=', dict(g=g)))
    attempts.append(('head_first=False', dict(head_first=False)))
    attempts.append(('bare', {}))
    errs = []
    for how, kw in attempts:
        try:
            out = fn(q, k, v, beta=beta, **kw)
            if isinstance(out, tuple):
                out = out[0]
            return out, how
        except Exception as e:                                # noqa: BLE001
            errs.append('%s -> %s' % (how, str(e).splitlines()[0][:110]))
    return None, errs


if __name__ == '__main__':
    print('KERNEL BRIDGE - can fla run our tied variant unchanged?\n')
    ops = report_env()
    if DEV != 'cuda':
        print('\nfla is Triton/GPU-only, so nothing below can run here. This file is syntax-checked')
        print('on CPU and is meant to be run on the GPU box, where it costs minutes.')
        raise SystemExit(0)
    if not ops:
        raise SystemExit('no fla ops available')

    torch.manual_seed(0)
    B, L, H, D = 2, 256, 4, 64
    a = F.normalize(torch.randn(B, L, H, D, device=DEV), dim=-1)      # the ADDRESS stream
    q = F.normalize(torch.randn(B, L, H, D, device=DEV), dim=-1)
    beta = torch.sigmoid(torch.randn(B, L, H, device=DEV))
    g = -F.softplus(torch.randn(B, L, H, device=DEV)) * 0.05
    tap = torch.tensor([4.0, 0.0, 0.0, 0.0], device=DEV)
    k, v = tied_kv(a, tap)

    print('\n1. TIED STREAM IS A LEGAL KERNEL INPUT (k_t == v_(t-1) at a sharp tap)')
    e = float((k[:, 1:] - v[:, :-1]).abs().max())
    print('   max |k_t - v_(t-1)| = %.2e   %s' % (e, 'OK' if e < 1e-5 else '*** FAIL ***'))

    print('\n2. KERNEL vs OUR SEQUENTIAL REFERENCE, one read')
    ref = seq_delta(q, k, v, beta, g, n_read=1)
    for name, fn in ops.items():
        if 'chunk' not in name:
            continue
        out, how = call_kernel(fn, q, k, v, beta, g if 'gated' in name else None)
        if out is None:
            print('   %-30s could not call:' % name)
            for line in how:
                print('      %s' % line)
            continue
        err = float((out.float() - ref).abs().max())
        rel = err / max(float(ref.abs().max()), 1e-6)
        print('   %-30s [%s] max abs %.3e  rel %.3e  %s'
              % (name, how, err, rel, 'MATCHES' if rel < 2e-2 else '*** DIVERGES ***'))

    print('\n3. n_read=2 AS TWO KERNEL PASSES (feed o back as q)')
    ref2 = seq_delta(q, k, v, beta, g, n_read=2)
    for name, fn in ops.items():
        if 'chunk' not in name:
            continue
        o1, how = call_kernel(fn, q, k, v, beta, g if 'gated' in name else None)
        if o1 is None:
            continue
        o1n = F.normalize(o1.float(), dim=-1).to(q.dtype)
        o2, _ = call_kernel(fn, o1n, k, v, beta, g if 'gated' in name else None)
        if o2 is None:
            continue
        o2n = F.normalize(o2.float(), dim=-1)
        r2 = F.normalize(ref2, dim=-1)
        err = float((o2n - r2).abs().max())
        print('   %-30s two-pass vs sequential chase: max %.3e  %s'
              % (name, err, 'MATCHES' if err < 5e-2 else 'DIVERGES - chase needs per-step states'))

    print('\n4. SPEED, the reason any of this matters')
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        seq_delta(q, k, v, beta, g, n_read=1)
    torch.cuda.synchronize()
    t_seq = (time.time() - t0) / 3
    print('   sequential python loop (L=%d): %.3f s' % (L, t_seq))
    for name, fn in ops.items():
        if 'chunk' not in name:
            continue
        out, how = call_kernel(fn, q, k, v, beta, g if 'gated' in name else None)
        if out is None:
            continue
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            call_kernel(fn, q, k, v, beta, g if 'gated' in name else None)
        torch.cuda.synchronize()
        t_k = (time.time() - t0) / 3
        print('   %-30s %.4f s   speedup %.0fx' % (name, t_k, t_seq / max(t_k, 1e-9)))

    print('\n5. BACKWARD - a forward-only kernel would be useless for training')
    for name, fn in ops.items():
        if 'chunk' not in name:
            continue
        aa = a.clone().requires_grad_(True)
        kk, vv = tied_kv(aa, tap)
        out, _ = call_kernel(fn, q, kk, vv, beta, g if 'gated' in name else None)
        if out is None:
            continue
        try:
            out.float().pow(2).mean().backward()
            ok = aa.grad is not None and torch.isfinite(aa.grad).all()
            print('   %-30s grad flows to the ADDRESS stream: %s' % (name, 'yes' if ok else 'NO'))
        except Exception as ex:                               # noqa: BLE001
            print('   %-30s backward raised: %s' % (name, str(ex).splitlines()[0][:100]))

    print('\n  IF 1-3 AND 5 PASS: our block is a pre-kernel construction plus a stock kernel, the')
    print('  tracker is the only sequential piece left, and the main run gets the speedup in 4.')
