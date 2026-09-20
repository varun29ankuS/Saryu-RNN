"""Where does a Saryu training step actually spend its time on a GPU?

WHY THIS COMES BEFORE WRITING A FUSED KERNEL. The chunk-parallel kernel is exact but loops over
L/C chunks in Python, so the suspicion is that it is launch-bound on a GPU. A fused Triton kernel
(forward AND backward, since training is what we care about) is days of work and cannot be tested
on this laptop at all -- no CUDA, no Triton, and torch.compile has no MSVC to call. So measure
first: if the recurrence is 20% of the step, a perfect kernel buys 20%.

A CPU chunk-size sweep already said C is not the lever there (135ms at C=8, 130 at 16, 146 at 32,
194 at 64, all exact to 3e-6). GPU launch overhead is a different regime, so C is swept again here.

WHAT IS MEASURED, with CUDA events and proper warmup, at the 5M and 25M shapes:
  1. a full training step (forward + backward + optimiser)
  2. the recurrence alone, forward
  3. the recurrence alone, forward + backward
  4. the share of the step the recurrence accounts for
  5. the same across chunk sizes C in {8, 16, 32, 64} and contexts 128 and 512

READ IT AS: share of step = the ceiling on what any fused kernel can save. Below ~25% it is not
the first thing to fix; above ~50% it is.
"""
import os
import subprocess
import sys
import time

import torch

REPO = 'https://github.com/varun29ankuS/Saryu-RNN'
WORK = '/kaggle/working/saryu'

subprocess.run(['git', 'clone', '--depth', '1', REPO, WORK], check=True)
os.chdir(WORK)
sys.path.insert(0, WORK)
print('repo at', subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
                                text=True).stdout.strip(), flush=True)

from saryu.model import CHUNK, SaryuV3LM, SaryuV3Block, chunkwise, sequential  # noqa: E402

assert torch.cuda.is_available(), 'this profile is meaningless without a GPU'
DEV = 'cuda'
print(torch.cuda.get_device_name(0), '| torch', torch.__version__, flush=True)


def timed(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters          # milliseconds


def kernel_inputs(BH, L, dh, nh=2):
    u = torch.nn.functional.normalize(torch.randn(BH, L, nh, dh, device=DEV), dim=-1)
    beta = 1 - torch.cos(torch.rand(BH, L, nh, device=DEV) * 3.14159)
    a = 1 - 0.9 * torch.rand(BH, L, device=DEV)
    b = torch.randn(BH, L, dh, device=DEV) * 0.1
    h0 = torch.randn(BH, dh, device=DEV)
    return h0, u, beta, a, b


def bench_model(d, nl, V, B, L, label, chunk=CHUNK):
    """chunk MUST be passed through: the first version of this script built the model at the library
    default and called chunkwise() directly with an explicit C, so it measured the shipped
    configuration no matter what training actually uses, and could not show the end-to-end win."""
    torch.manual_seed(0)
    m = SaryuV3LM(V, d, nl, chunk=chunk).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, V, (B, L), device=DEV)
    y = torch.randint(0, V, (B, L), device=DEV)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        loss.backward()
        opt.step()

    step_ms = timed(step, iters=8)

    H, dh = 8, d // 8
    h0, u, beta, a, b = kernel_inputs(B * H, L, dh)
    fwd_ms = timed(lambda: chunkwise(h0, u, beta, a, b, chunk), iters=8)

    ug = u.clone().requires_grad_(True)
    bg = b.clone().requires_grad_(True)

    def fwd_bwd():
        out = chunkwise(h0, ug, beta, a, bg, chunk)
        out.sum().backward()
    fb_ms = timed(fwd_bwd, iters=8)

    per_layer = fb_ms
    share = 100.0 * per_layer * nl / step_ms
    print(f'{label:>12} d={d:<4} B={B:<3} L={L:<4} C={chunk:<3} | step {step_ms:7.1f} ms | '
          f'recurrence fwd {fwd_ms:6.2f} ms, fwd+bwd {fb_ms:6.2f} ms/layer | x{nl} layers = '
          f'{per_layer*nl:6.1f} ms = {share:4.1f}% of the step', flush=True)
    del m, opt
    torch.cuda.empty_cache()
    return share


def bench_chunks(d, B, L):
    H, dh = 8, d // 8
    h0, u, beta, a, b = kernel_inputs(B * H, L, dh)
    ug = u.clone().requires_grad_(True)
    bg = b.clone().requires_grad_(True)
    ref = None
    print(f'  chunk sweep at d={d} B={B} L={L} (dh={dh}, {B*H} rows)', flush=True)
    for C in (8, 16, 32, 64):
        if C > L:
            continue
        f = timed(lambda: chunkwise(h0, u, beta, a, b, C=C), iters=8)

        def fwd_bwd():
            out = chunkwise(h0, ug, beta, a, bg, C=C)
            out.sum().backward()
        fb = timed(fwd_bwd, iters=8)
        with torch.no_grad():
            out = chunkwise(h0, u, beta, a, b, C=C)
        if ref is None:
            ref, err = out, 0.0
        else:
            err = float((out - ref).abs().max())
        print(f'    C={C:<3} {L//C:>3} python iterations | fwd {f:6.2f} ms | fwd+bwd {fb:6.2f} ms '
              f'| max |diff vs C=8| {err:.1e}', flush=True)


t0 = time.time()
print('\n== exactness first: the kernel must equal the recurrence on this device')
with torch.no_grad():
    h0, u, beta, a, b = kernel_inputs(16, 128, 40)
    err = float((chunkwise(h0, u, beta, a, b) - sequential(h0, u, beta, a, b)).abs().max())
print(f'   max |chunkwise - sequential| = {err:.2e}', flush=True)

print('\n== how much of a training step is the recurrence, and what does the chunk size do to the'
      ' WHOLE step?')
shares = []
for d, nl, B, L, label in ((328, 4, 48, 128, 'saryu-5M'), (328, 4, 8, 512, 'saryu-5M'),
                           (744, 4, 48, 128, 'saryu-25M'), (744, 4, 8, 512, 'saryu-25M')):
    for C in (8, 32, 64):                    # 8 is what the released checkpoints trained with
        shares.append(bench_model(d, nl, 481, B, L, label, chunk=C))

print('\n== does the chunk size matter on a GPU?')
bench_chunks(328, 48, 128)
bench_chunks(744, 8, 512)

print(f'\nverdict: the recurrence is {min(shares):.0f}-{max(shares):.0f}% of a training step, so a '
      f'perfect fused kernel saves at most that.')
print(f'total {time.time() - t0:.0f}s')
