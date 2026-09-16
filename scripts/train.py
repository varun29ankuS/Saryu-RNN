"""saryu-25M: the scale run (from the repo root: python scripts/train.py). Every component pre-validated at 5M; nothing untested.

COMPOSITION (all measured at 5M, six seeds of base): v3 anatomy (worth 0.46 bpc) +
exact chunkwise kernel (7x; causal to float dust) + modern recipe (WSD, no decay on
1D params, residual init 1/sqrt(2nl), z-loss; Muon excluded -- measured marginal) +
train-short-run-long (ctx 128 is a pure compute choice; measured no context tax).
nl=4 WIDE, not deeper (v5depth: width beat depth at fixed budget; at 25M, d~740
gives head-dim ~92, far above the capacity floor).

REGISTERED, zero knobs to adjust afterward:
  bpc@128 in [1.55, 1.68] at 10,000 steps (small-char-LM scaling for 5x params;
  stated as a band because we have one scale point, not a fitted slope -- honest).
  bpc improves MONOTONICALLY with eval context (the architectural signature holds
  at scale) -- bpc@8192 < bpc@128 by >= 0.10.
  s/step in [0.9, 1.4] on the T4 (kernel arithmetic, ~5x the 5M cost).
  VULCAN-WATCH (named in advance): if the band is missed, the registered number
  rules -- no post-hoc story about steps or learning rate gets to rescue it.
Checkpoint saved for the register audit at scale (does L3H1-style localization
persist? do NEW registers appear at 5x capacity?).

REFERENCES
  [Liu et al. 2025] Muon is Scalable for LLM Training, arXiv 2502.16982
  [Wen et al. 2024] Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective, arXiv 2410.05192
"""
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET_PARAMS = int(os.environ.get('TARGET_PARAMS', 25_000_000))
CTX = int(os.environ.get('CTX', 128))
BS = int(os.environ.get('BS', 48))
NL, NH, NHH = int(os.environ.get('NL', 4)), 2, 8      # n_h reflections, NHH heads
CHUNK = 8
STEPS = int(os.environ.get('STEPS', 10000))
EVAL_LENS = [int(x) for x in os.environ.get('EVAL_LENS','128,512,2048,8192').split(',') if x.strip()]
SCORE_LAST = 128
EVAL_WINDOWS = 64
WALLCAP_S = float(os.environ.get('WALLCAP_S', 9000))

if DEV == 'cuda':
    cap = torch.cuda.get_device_capability(0)
    archs = [int(a[3:5]) for a in torch.cuda.get_arch_list() if a.startswith('sm_')]
    print('  {} sm_{}{}'.format(torch.cuda.get_device_name(0), cap[0], cap[1]), flush=True)
    if cap[0] * 10 + cap[1] < min(archs):
        print('  UNUSABLE GPU -> CPU fallback', flush=True)
        DEV = 'cpu'
        STEPS = min(STEPS, 1200)
print('device ' + DEV, flush=True)


def load(vocab_min=100):
    cands = glob.glob('/kaggle/input/**/enwik8', recursive=True)
    _local = os.environ.get('CORPUS', 'corpus/enwik8')
    _p = cands[0] if cands else (_local if os.path.exists(_local) else 'enwik8')
    raw = open(_p, 'rb').read().decode('utf-8', errors='ignore')
    if os.environ.get('CHARS'):        # laptop: cap the corpus to keep encode time sane
        raw = raw[:int(os.environ['CHARS'])]
    cps = np.frombuffer(raw.encode('utf-32-le'), dtype=np.uint32)
    uniq, counts = np.unique(cps, return_counts=True)
    keep = np.sort(uniq[counts >= vocab_min])
    pos = np.clip(np.searchsorted(keep, cps), 0, len(keep) - 1)
    return torch.tensor(np.where(keep[pos] == cps, pos + 1, 0).astype(np.int64)), len(keep) + 1


# model code: saryu/model.py (identical to what this script trained)
from saryu.model import chunkwise, sequential, RMSNorm, SaryuV3Block, SwiGLU, SaryuV3LM  # noqa: E402,F401


def _ns5(G, steps=5):
    """Newton-Schulz orthogonalization (Keller Jordan coefficients)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tall:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon for 2D hidden matrices: orthogonalized momentum + decoupled decay."""

    def __init__(self, params, lr=0.02, momentum=0.95, wd=0.01):
        super().__init__(params, dict(lr=lr, momentum=momentum, wd=wd))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if 'buf' not in st:
                    st['buf'] = torch.zeros_like(p)
                buf = st['buf']
                buf.mul_(g['momentum']).add_(p.grad)
                upd = _ns5(p.grad.add(buf, alpha=g['momentum']))
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.mul_(1 - g['lr'] * g['wd'])
                p.add_(upd, alpha=-g['lr'] * scale)


def wsd_lambda(step, total, warm=0.05, decay=0.2):
    w, d0 = int(total * warm), int(total * (1 - decay))
    if step < w:
        return step / max(w, 1)
    if step < d0:
        return 1.0
    return max(0.0, (total - step) / max(total - d0, 1))


class GRULM(nn.Module):
    kind = 'gru'

    def __init__(self, vocab, d, nl):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.rnn = nn.GRU(d, d, num_layers=nl, batch_first=True)
        self.head = nn.Linear(d, vocab)
        self.d, self.nl = d, nl

    def forward(self, idx):
        h, _ = self.rnn(self.emb(idx))
        return self.head(h)

    def state_floats(self, L=None):
        return self.d * self.nl


def sized(ctor, vocab, target, nl, lo=64, hi=2048):
    best = None
    while lo <= hi:
        mid = max(NHH, ((lo + hi) // 2 // NHH) * NHH)
        n = sum(p.numel() for p in ctor(vocab, mid, nl).parameters())
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (mid, n)
        if n < target:
            lo = mid + NHH
        else:
            hi = mid - NHH
    return best


def eval_ends(va, seed=99):
    """Scoring positions shared by every context length.

    A window of length L ENDS at one of these positions, so each L scores the SAME final
    SCORE_LAST characters with more or less context in front of them. (Until 2026-09-16 the window
    STARTS were drawn per length, so the columns of the results table scored different characters
    and were only loosely comparable.)

    The earliest position is set by EVAL_ANCHOR, not by the lengths this run happens to ask for, so
    two runs are comparable even if one evaluates at fewer lengths."""
    lo = int(os.environ.get('EVAL_ANCHOR', 8192)) + 1
    if lo >= len(va) - 1:                      # tiny corpora (smoke runs)
        lo = max(1, len(va) // 2)
    ge = torch.Generator().manual_seed(seed)
    return torch.randint(lo, len(va) - 1, (EVAL_WINDOWS,), generator=ge).tolist()


@torch.no_grad()
def bpc_at(m, va, L, ends):
    m.eval()
    if L > min(ends):
        m.train()
        return float('nan')                    # not enough text in front of the scoring positions
    per = max(1, min(32, 8192 // L))
    tot, ntok = 0.0, 0
    for i in range(0, len(ends), per):
        ch = ends[i:i + per]
        x = torch.stack([va[e - L:e] for e in ch]).to(DEV)
        y = torch.stack([va[e - L + 1:e + 1] for e in ch]).to(DEV)
        lg = m(x)[:, -SCORE_LAST:]
        yy = y[:, -SCORE_LAST:]
        tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), yy.reshape(-1),
                                     reduction='sum'))
        ntok += yy.numel()
    m.train()
    return tot / ntok / math.log(2)


# ------------------------------------------------------------ exactness self-check
torch.manual_seed(0)
blk = SaryuV3Block(64)
zt = torch.randn(2, 64, 64)
u, beta, a, b = blk.mix(zt)
uf = u.permute(0, 2, 1, 3, 4).reshape(2 * NHH, 64, NH, 64 // NHH)
bf_ = beta.permute(0, 2, 1, 3).reshape(2 * NHH, 64, NH)
af = a.permute(0, 2, 1).reshape(2 * NHH, 64)
bbf = b.permute(0, 2, 1, 3).reshape(2 * NHH, 64, 64 // NHH)
h0 = blk.h0[None].expand(2, NHH, 64 // NHH).reshape(2 * NHH, 64 // NHH)
with torch.no_grad():
    err = float((chunkwise(h0, uf, bf_, af, bbf)
                 - sequential(h0, uf, bf_, af, bbf)).abs().max())
print('kernel self-check on the v3 module: max err {:.2e}'.format(err), flush=True)
assert err < 1e-3, 'chunkwise != sequential on this module -- aborting'

# ------------------------------------------------------------ main
t0 = time.time()
data, V = load()
print('enwik8: {:,} chars, vocab {} ({:.0f}s)'.format(len(data), V, time.time() - t0),
      flush=True)
split = int(0.95 * len(data))
tr, va = data[:split], data[split:]

def modern_init(m, nl):
    with torch.no_grad():
        for blk in m.mix:
            blk.out.weight.mul_(1.0 / (2 * nl) ** 0.5)
        for f in m.ffn:
            f.w2.weight.mul_(1.0 / (2 * nl) ** 0.5)
    return m


def split_params(m):
    twoD, rest = [], []
    for name, p in m.named_parameters():
        if p.ndim == 2 and 'emb' not in name and 'head' not in name:
            twoD.append(p)
        else:
            rest.append(p)
    return twoD, rest


def parse_arms(spec, seeds):
    """ARMS='adamw@1e-3,muon@0.02' with SEEDS='0,1,2' -> one arm per (setting, seed).

    The learning rate after @ is AdamW's in adamw mode and Muon's in muon mode (where the
    embeddings, the head and every 1-D parameter stay on AdamW at 1e-3, as Muon requires)."""
    arms = []
    for seed in [int(s) for s in seeds.split(',') if s.strip()]:
        for item in [a for a in spec.split(',') if a.strip()]:
            mode, _, lr = item.partition('@')
            lr = float(lr) if lr else (0.02 if mode == 'muon' else 1e-3)
            arms.append(('{}@{:g} s{}'.format(mode, lr, seed), mode, lr, seed))
    return arms


ARMS = (parse_arms(os.environ['ARMS'], os.environ.get('SEEDS', '0'))
        if os.environ.get('ARMS') else [('saryu-25M', 'adamw', 1e-3, 0)])
results = []
# gate options (defaults reproduce the trained configuration exactly); see saryu/model.py
TIMESCALES = os.environ.get('GATE_TIMESCALES', '0') == '1'
WRITE_SCALE = os.environ.get('WRITE_SCALE', '0') == '1'
# with timescales on, the input term of the gate is scaled by this so the per-head bias ladder is
# not swamped; 1.0 reproduces the first (invalid) attempt
GATE_W_SCALE = float(os.environ.get('GATE_W_SCALE', 0.01))
FREEZE_GATE_BIAS = os.environ.get('FREEZE_GATE_BIAS', '0') == '1'
GATE_CEILING = os.environ.get('GATE_CEILING', '0') == '1'
# chunk size is exact at any value; 32-64 is 3-6x faster than 8 on a T4 (evidence/results/
# kernel_profile.txt). The released checkpoints were trained at 8.
CHUNK_TRAIN = int(os.environ.get('CHUNK', 32))


def _ctor(vocab, width, nl):
    return SaryuV3LM(vocab, width, nl, timescales=TIMESCALES, write_scale=WRITE_SCALE,
                     gate_w_scale=GATE_W_SCALE, freeze_gate_bias=FREEZE_GATE_BIAS,
                     gate_ceiling=GATE_CEILING, chunk=CHUNK_TRAIN)


for name, mode, arm_lr, arm_seed in ARMS:
    d, n = sized(_ctor, V, TARGET_PARAMS, NL)
    torch.manual_seed(arm_seed)
    m = _ctor(V, d, NL).to(DEV)
    if mode != 'old':
        modern_init(m, NL)
    if mode == 'old':
        opt = torch.optim.AdamW(m.parameters(), lr=arm_lr, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, arm_lr, total_steps=STEPS,
                                                  pct_start=0.05)
        opt2 = None
    else:
        twoD, rest = split_params(m)
        decay1d = [p for p in rest if p.ndim >= 2]
        nodecay = [p for p in rest if p.ndim < 2]
        opt = torch.optim.AdamW([{'params': decay1d, 'weight_decay': 0.01},
                                 {'params': nodecay, 'weight_decay': 0.0}],
                                lr=(1e-3 if mode == 'muon' else arm_lr))
        if mode == 'muon':
            opt2 = Muon(twoD, lr=arm_lr, momentum=0.95, wd=0.01)
        else:
            opt.add_param_group({'params': twoD, 'weight_decay': 0.01})
            opt2 = None
        base_lrs = [g['lr'] for g in opt.param_groups]
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda st: wsd_lambda(st, STEPS))
    g = torch.Generator().manual_seed(1 + arm_seed)
    _start = 0
    _r = os.environ.get('RESUME')
    if _r and os.path.exists(_r):
        _ck = torch.load(_r, map_location=DEV, weights_only=False)
        m.load_state_dict(_ck['state'])
        if 'opt' in _ck:
            opt.load_state_dict(_ck['opt'])
        _start = int(_ck.get('step', 0))
        for _ in range(_start):
            sch.step()          # replay the schedule so LR is correct on resume
        print('  RESUMED from {} at step {}'.format(_r, _start), flush=True)
    print('{}: d={} {:,} params'.format(name, d, n), flush=True)
    t1, done, skips = time.time(), _start, 0
    for s in range(_start, STEPS):
        i = torch.randint(0, len(tr) - CTX - 1, (BS,), generator=g)
        x = torch.stack([tr[j:j + CTX] for j in i]).to(DEV)
        y = torch.stack([tr[j + 1:j + CTX + 1] for j in i]).to(DEV)
        lg = m(x)
        loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        if mode != 'old':
            loss = loss + 1e-4 * torch.logsumexp(lg, -1).pow(2).mean()   # z-loss
        if not torch.isfinite(loss):
            skips += 1
            opt.zero_grad()
            sch.step()
            if skips == 25 and SaryuV3Block.USE_KERNEL:
                SaryuV3Block.USE_KERNEL = False
                print('  25 non-finite losses -> permanent fallback to sequential path',
                      flush=True)
            done = s + 1
            continue
        opt.zero_grad()
        if opt2 is not None:
            opt2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if opt2 is not None:
            fac = wsd_lambda(s, STEPS)
            for gg in opt2.param_groups:
                gg['lr'] = arm_lr * fac
            opt2.step()
        sch.step()
        done = s + 1
        if done % 1000 == 0:
            print('  step {:>5} train bpc {:.3f} {:.0f}s'.format(
                done, float(loss.detach()) / math.log(2), time.time() - t1), flush=True)
        _sv = os.environ.get('SAVE', '')
        if _sv and done % max(1, int(os.environ.get('SAVE_EVERY', 500))) == 0:
            torch.save({'state': m.state_dict(), 'arm': name, 'd': d, 'step': done,
                        'opt': opt.state_dict()}, _sv + '.tmp')
            os.replace(_sv + '.tmp', _sv)
            print('    saved at step {}'.format(done), flush=True)
        if time.time() - t1 > WALLCAP_S:
            print('  WALLCAP at {}'.format(done), flush=True)
            break
    ts = time.time() - t1
    ends = eval_ends(va)
    row = {}
    for L in EVAL_LENS:
        try:
            row[L] = bpc_at(m, va, L, ends)
        except RuntimeError as e:
            row[L] = float('nan')
            print('  L={} failed: {}'.format(L, str(e)[:80]), flush=True)
            if DEV == 'cuda':
                torch.cuda.empty_cache()
        print('  bpc@{:<5} {:.4f}'.format(L, row[L]), flush=True)
    print('  non-finite steps skipped: {}'.format(skips), flush=True)
    results.append((name, d, n, done, ts, row))
    _sv = os.environ.get('SAVE', '/kaggle/working/saryu_25m.pt')
    if _sv:
        torch.save({'state': m.state_dict(), 'arm': name, 'd': d, 'step': done,
                    'opt': opt.state_dict()}, _sv)
    del m, opt
    if DEV == 'cuda':
        torch.cuda.empty_cache()

print('=' * 92)
print('SARYU-25M: the scale run (seed 0)')
print('{:>14} {:>5} {:>10} {:>6} {:>8} '.format('arm', 'd', 'params', 'steps', 's/step')
      + ' '.join('{:>10}'.format('bpc@' + str(L)) for L in EVAL_LENS))
print('-' * 92)
for name, d, n, done, ts, row in results:
    print('{:>14} {:>5} {:>10,} {:>6} {:>8.3f} '.format(name, d, n, done, ts / max(done, 1))
          + ' '.join('{:>10.4f}'.format(row[L]) for L in EVAL_LENS))
print()
print('  reference: v3.1 across seeds 0/1: bpc@128 1.7867/1.7934, bpc@8192 1.6065/1.6180')
print('  v3 closing most of the gap -> the v1 loss was ANATOMY, not the transport')
print('  s/step is the kernel dividend; v1 could not finish its budget, v3 should')
