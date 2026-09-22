"""Does Saryu learn MORE FROM LESS DATA than a transformer or a GRU at equal parameters?

THIS IS THE EXPERIMENT THE PROJECT EXISTS TO RUN, and it has never been run. Saryu's stated goal is
cheap intelligence -- trainable on one CPU in days, far fewer parameters, learning from reading
something once or twice. Every one of the ~180 logged runs in this repository measured recall
capacity on synthetic MQAR, which is a CAPABILITY axis. Not one measured sample efficiency. An audit
on 2026-09-21 (scripts/audit_claims.py) found that 94 of 110 results files have no run behind them
at all, and every one of those is MQAR: so even in the best case where all of them are true, none
of them bears on the thesis. This does.

Pre-registered in docs/learning_machine.md section 4, falsifier and all, before this file existed.

WHAT IS MEASURED, and it is the whole design. Not bits-per-character at a fixed step count -- that
is a capability number and the archive script this harness comes from reported exactly that. The
quantity is LOSS AS A FUNCTION OF TOKENS SEEN, and the headline is TOKENS-TO-THRESHOLD: how much
data each architecture needs to reach the same bpc. Steps are not comparable across architectures
here (a recurrence and a transformer do different work per step); tokens are what "learning from
less" is about. Wall-clock is reported alongside, because a model that is sample-efficient and
ten times slower has not made intelligence cheap.

ARMS, parameter-matched by binary search on width so the only difference is architecture:

    saryu        the current model, matrix memory and gate lower bound ON -- what Saryu now is
    saryu-plain  the same model with both off -- does this session's work help EFFICIENCY, or
                 only the recall capacity it was built for and measured on?
    gru          the classical recurrent baseline
    transformer  the thing to beat

REGISTERED PREDICTIONS, before running:
  P1  THE ONE THAT MATTERS. saryu reaches any given bpc in FEWER TOKENS than the transformer at
      equal parameters. That is the efficiency claim, stated so it can fail.
  P2  saryu beats the GRU in tokens-to-threshold. If it does not, the gain is "a recurrence", not
      this recurrence, and the architecture is not what is doing the work.
  P3  saryu >= saryu-plain. The memory and gate bound were built for recall; whether they buy
      efficiency is a separate question and has never been asked.
  P4  saryu-hybrid beats both pure saryu AND the transformer in tokens-to-threshold. This is the
      configuration the whole field ships -- Qwen3-Next and Kimi Linear independently chose 3:1 --
      and if it does not win here, either our recurrence is not contributing what theirs does or
      this task and budget cannot see the difference. Note the hybrid GIVES UP the constant-state
      property: its state_floats at L=2048 is ~1000x the pure model's, so a win here is a win on
      sample efficiency bought with memory, and must be reported with both numbers.

FALSIFIER, quoted from the pre-registration so it cannot be softened afterwards: "If the transformer
reaches the same loss in the same or fewer tokens at equal parameters, the efficiency claim for this
architecture is dead as stated, and the case for Saryu rests on the group-theoretic results and the
constant-cost inference alone -- not on learning more from less."

A REAL RISK, NAMED IN ADVANCE so it is not used as an excuse afterwards. Three experiments in this
project have been voided by reading an under-trained run as a negative result, and the rule added to
CLAIMS.md after the third is that a curve still rising when stopped is not a result. Tokens-to-
threshold is measured at SEVERAL thresholds, and any arm that never reaches a threshold is reported
as "not reached within budget" rather than given a number. If no arm reaches the highest threshold,
that threshold is void and says nothing about any arm.

Every arm logs through metrics.Run, so this result is checkable by scripts/audit_claims.py -- which
is the point of having written that checker.

    MODE=probe python evidence/efficiency.py     build, match params, time 10 steps, cost the run
    MODE=full  python evidence/efficiency.py     the experiment
Environment: MODE, TARGET, STEPS, CTX, BS, LR, NL, SEEDS, ARMS, THREADS.
"""
from __future__ import annotations

import collections
import math
import pathlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from saryu.model import SaryuV3LM                                    # noqa: E402
from saryu.metrics import Run                                        # noqa: E402

CORPUS = os.path.join(ROOT, 'corpus', 'enwik8')
TARGET = int(os.environ.get('TARGET', 5_000_000))
CTX = int(os.environ.get('CTX', 128))
BS = int(os.environ.get('BS', 16))
STEPS = int(os.environ.get('STEPS', 3000))
LR = float(os.environ.get('LR', 1e-3))
NL = int(os.environ.get('NL', 4))
SEEDS = int(os.environ.get('SEEDS', 1))
MODE = os.environ.get('MODE', 'probe')
EVAL_EVERY = int(os.environ.get('EVAL_EVERY', 100))
CHUNKSZ = int(os.environ.get('CHUNKSZ', 32))
# Cap on how much of enwik8 is held in memory. A 1500-step run consumes 1500*16*128 = 3.07M
# tokens, so holding all 99.6M serves nothing and cost two runs to an OOM kill on a 15.7 GB
# machine with ~3 GB free. 20M leaves ~19M distinct window starts against 3.07M tokens consumed,
# so windows still effectively never repeat. MUST be identical across arms in a comparison --
# it is recorded in every run's config for exactly that reason.
MAXCHARS = int(os.environ.get('MAXCHARS', 0)) or None
torch.set_num_threads(int(os.environ.get('THREADS', max(1, (os.cpu_count() or 4) - 2))))

# bpc thresholds for tokens-to-threshold. Chosen loose-to-tight so that an arm failing the tight
# one still reports the loose ones rather than a single all-or-nothing number.
THRESHOLDS = [3.0, 2.5, 2.2, 2.0]


# ------------------------------------------------------------------ data
def load(vocab_min=100):
    """Encode enwik8 without ever materialising a 100M-element Python list.

    The first version of this did `np.array([stoi.get(ch, 0) for ch in raw])` and then stored the
    result as int64. That is a ~0.8 GB list of boxed ints, transiently, on top of a 0.8 GB tensor,
    and it killed a four-arm run partway through for want of memory. Encoded through a codepoint
    lookup table in 4 MB blocks and held as int16, the same data is ~0.2 GB with a bounded
    transient. Batches are cast to long at use, which costs nothing at 16x128.

    Identical output to the old path: same vocabulary rule, same ids, same OOV = 0."""
    raw = open(CORPUS, 'rb').read().decode('utf-8', errors='ignore')
    # VOCABULARY FROM THE FULL CORPUS, ALWAYS -- then slice. Building it from the truncated text
    # instead took the vocabulary from 481 to 224, because characters that are rare overall fall
    # under the frequency threshold within any slice. That silently changes the task: bits per
    # character over 224 symbols is not comparable to bits per character over 481, so the capped
    # runs would have looked better than the uncapped ones for no architectural reason at all.
    c = collections.Counter(raw)
    keep = sorted([ch for ch, n in c.items() if n >= vocab_min])
    stoi = {ch: i + 1 for i, ch in enumerate(keep)}          # 0 = OOV
    lut = np.zeros(max(ord(ch) for ch in keep) + 1, dtype=np.int16)
    for ch, i in stoi.items():
        lut[ord(ch)] = i
    if MAXCHARS:
        raw = raw[:MAXCHARS]                                  # slice only AFTER the vocab is fixed
    arr = np.empty(len(raw), dtype=np.int16)
    step = 1 << 22
    for s in range(0, len(raw), step):
        cp = np.frombuffer(raw[s:s + step].encode('utf-32-le'), dtype=np.uint32)
        np.copyto(arr[s:s + len(cp)], lut[np.minimum(cp, len(lut) - 1)],
                  where=cp < len(lut))                        # anything above the table is OOV 0
        arr[s:s + len(cp)][cp >= len(lut)] = 0
    return torch.from_numpy(arr), len(keep) + 1


# ------------------------------------------------------------------ arms
class SaryuArm(nn.Module):
    """The current model. `full` toggles the two components added on 2026-09-20/21."""

    def __init__(self, vocab, d, nl, full=True):
        super().__init__()
        # chunk 32 rather than the default 8. The chunk size is a performance knob and not an
        # approximation -- tests/test_model.py checks exactness at 4, 8, 16, 32 and 64, and all
        # pass -- and at ctx 128 it cuts 16 sequential Python chunk iterations per layer to 4,
        # measured at 2.25 -> 1.50 s/step. Running the arm at the slow default would have
        # reported a wall-clock number 1.5x worse than the architecture actually is.
        self.net = SaryuV3LM(vocab, d, nl, memory=full, mem_decay=full, mem_decouple=full,
                             gate_lb=full, chunk=CHUNKSZ, mem_chunk=CHUNKSZ)
        self.d, self.nl = d, nl

    def forward(self, idx):
        return self.net(idx)

    def state_floats(self, L=2048):
        return self.d * self.nl


class HybridArm(nn.Module):
    """3 recurrent layers to 1 attention layer -- the ratio Qwen3-Next and Kimi Linear both use.

    The bet this arm tests: the recurrence is the cheap substrate for evolving state, and 25%
    attention buys back the unbounded retrieval a fixed state cannot do. Our own ablations found
    those to be separate capabilities (transport is necessary for tracking and irrelevant to
    recall, and the reverse for the memory), so pairing them is the design the dissociation
    implies rather than a borrowed recipe."""

    def __init__(self, vocab, d, nl, full=True):
        # `full` toggles the matrix memory and gate bound, exactly as in SaryuArm. The plain
        # variant is the one that matters now: the first efficiency run falsified P3: those two
        # components cost 2x on tokens-to-loss (204,800 vs 409,600 to bpc 3.0), so a hybrid
        # carrying them would be measured through a known handicap. And the redundancy is
        # expected rather than surprising -- the attention layer does retrieval, which is the
        # job the matrix memory was added to do.
        super().__init__()
        self.net = SaryuV3LM(vocab, d, nl, memory=full, mem_decay=full, mem_decouple=full,
                             gate_lb=full, chunk=CHUNKSZ, mem_chunk=CHUNKSZ, attn_every=4)
        self.d, self.nl = d, nl

    def forward(self, idx):
        return self.net(idx)

    def state_floats(self, L=2048):
        return self.net.state_floats(L=L)       # NOT constant: the attention layers cache


class GRUArm(nn.Module):
    """Pre-norm and residual per layer, NOT a bare stacked nn.GRU.

    The archive harness this was consolidated from used nn.GRU(num_layers=nl) directly, with no
    normalisation and no residual connections. Stacked four deep that is a known-weak baseline,
    and beating it would measure our harness rather than the architectures: Saryu and the
    transformer arms both get pre-norm and residuals, so the GRU gets them too. Its first run
    under the bare version reached bpc 5.07 at 1M tokens where saryu-plain was at 2.20."""

    def __init__(self, vocab, d, nl, full=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(nl)])
        self.rnns = nn.ModuleList([nn.GRU(d, d, batch_first=True) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.d, self.nl = d, nl

    def forward(self, idx):
        x = self.emb(idx)
        for ln, rnn in zip(self.lns, self.rnns):
            h, _ = rnn(ln(x))
            x = x + h
        return self.head(self.lnf(x))

    def state_floats(self, L=2048):
        return self.d * self.nl


class TFBlock(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        T = x.shape[1]
        m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=m, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TransformerArm(nn.Module):
    """maxlen defaults to the TRAINING CONTEXT, and that is not a detail. The archive script this
    was consolidated from used maxlen=8192, which at d=240 puts 1,966,080 parameters -- 39.5% of a
    5M budget -- into positional embeddings for positions the model never sees at ctx 128. Matching
    on total parameters then gives Saryu ~5M of usable capacity against the transformer's ~3M, and
    any efficiency win we measured would have been partly that handicap. Measured, not reasoned:
    4,975,201 total / 1,966,080 positional at maxlen 8192, against 3,039,841 / 30,720 at 128."""

    def __init__(self, vocab, d, nl, full=True, nh=4, maxlen=None):
        maxlen = maxlen or CTX
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos = nn.Embedding(maxlen, d)
        nn.init.normal_(self.pos.weight, std=0.02)
        # width is searched over, so the head count has to divide it -- otherwise the binary
        # search silently skips every width that is not a multiple of 4 and the "matched"
        # parameter counts are matched only among the widths that happened to work.
        self.blocks = nn.ModuleList([TFBlock(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.d, self.nl = d, nl

    def forward(self, idx):
        T = idx.shape[1]
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

    def state_floats(self, L=2048):
        return 2 * L * self.d * self.nl                      # KV cache, grows with context


ARMS = {
    'saryu':        (SaryuArm, True),
    'saryu-plain':  (SaryuArm, False),
    'saryu-hybrid': (HybridArm, True),
    'saryu-hybrid-plain': (HybridArm, False),
    'gru':          (GRUArm, True),
    'transformer':  (TransformerArm, True),
}


def sized(ctor, full, vocab, target, nl, lo=64, hi=1024, mult=8):
    """Binary-search width for a parameter match. `mult` keeps d divisible by the head count."""
    best = None
    while lo <= hi:
        mid = max(mult, ((lo + hi) // 2 // mult) * mult)
        try:
            n = sum(p.numel() for p in ctor(vocab, mid, nl, full).parameters())
        except Exception:
            n = None
        if n is not None:
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (mid, n)
            if n < target:
                lo = mid + mult
            else:
                hi = mid - mult
        else:
            lo = mid + mult
    return best


# ------------------------------------------------------------------ training
@torch.no_grad()
def evaluate(m, va, V, n_batches=8, seed=7):
    g = torch.Generator().manual_seed(seed)
    m.eval()
    tot = 0.0
    for _ in range(n_batches):
        i = torch.randint(0, len(va) - CTX - 1, (BS,), generator=g)
        x = torch.stack([va[j:j + CTX] for j in i]).long()
        y = torch.stack([va[j + 1:j + CTX + 1] for j in i]).long()
        tot += float(F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))) / n_batches
    m.train()
    return tot / math.log(2)                                 # nats -> bits per character


def train(name, ctor, full, d, nl, npar, tr, va, V, seed):
    torch.manual_seed(seed)
    m = ctor(V, d, nl, full)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    g = torch.Generator().manual_seed(1000 + seed)
    log = Run(f'eff-{name}-s{seed}',
              config=dict(arm=f'efficiency/{name}', d=d, nl=nl, params=npar, ctx=CTX, bs=BS,
                          steps=STEPS, lr=LR, seed=seed, tokens_total=STEPS * BS * CTX,
                          corpus_chars=MAXCHARS or 0, chunk=CHUNKSZ,
                          thresholds=THRESHOLDS))
    # threshold -> (tokens seen, hours spent). BOTH, because the probe showed they disagree:
    # saryu costs 2.62 s/step against the transformer's 0.40 at equal parameters, so an arm can
    # win on tokens and still lose on wall-clock by 6.5x. "Cheap intelligence" needs both halves,
    # and reporting only the flattering one is the failure this repo just audited itself for.
    hit = {}
    t0, best = time.time(), 99.
    for s in range(STEPS):
        i = torch.randint(0, len(tr) - CTX - 1, (BS,), generator=g)
        x = torch.stack([tr[j:j + CTX] for j in i]).long()
        y = torch.stack([tr[j + 1:j + CTX + 1] for j in i]).long()
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % EVAL_EVERY == 0 or s == 0:
            bpc = evaluate(m, va, V)
            tokens = (s + 1) * BS * CTX
            best = min(best, bpc)
            for th in THRESHOLDS:
                if th not in hit and bpc <= th:
                    hit[th] = (tokens, (time.time() - t0) / 3600)
            log.log(s + 1, loss=float(loss),
                    **{'eval/bpc': bpc, 'eval/best_bpc': best, 'eval/tokens': tokens,
                       'eval/hours': (time.time() - t0) / 3600})
            print(f'    {name:<12} {tokens:>10,} tok  bpc {bpc:.3f}  '
                  f'({(time.time()-t0)/60:.0f}m)', flush=True)
    log.done()
    return dict(name=name, best=best, hit=hit, hours=(time.time() - t0) / 3600,
                params=npar, d=d)


def table_from_runs():
    """Rebuild the tokens-to-threshold table from runs/ rather than from one process's memory.

    Two four-arm runs were killed by the OS partway through, losing the arms that had already
    finished because the table was only printed at the end. Every arm logs through metrics.Run,
    so the table is recoverable from the logs -- which is what that logging was for. This also
    lets arms run as separate processes, so memory is released between them and a kill costs one
    arm instead of all of them.

    Arms are grouped by corpus_chars and chunk: comparing across a change in either is exactly
    the confound that truncating the corpus nearly introduced, so runs that differ are separated
    rather than silently pooled."""
    import json
    from collections import defaultdict
    runs = defaultdict(list)
    for p in pathlib.Path('runs').glob('*.jsonl'):
        try:
            ls = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
        except Exception:
            continue
        cfg = next((l.get('config') for l in ls if isinstance(l.get('config'), dict)), None)
        if not cfg or not str(cfg.get('arm', '')).startswith('efficiency/'):
            continue
        pts = [(m['eval/tokens'], m['eval/bpc'], m.get('eval/hours', 0.0))
               for m in (l.get('metrics') for l in ls)
               if isinstance(m, dict) and 'eval/bpc' in m and 'eval/tokens' in m]
        if not pts:
            continue
        key = (cfg.get('corpus_chars', 0), cfg.get('chunk', 0), cfg.get('steps', 0))
        runs[key].append((str(cfg['arm']).split('/', 1)[1], cfg.get('seed', 0),
                          cfg.get('params', 0), sorted(pts)))
    if not runs:
        print('no efficiency runs logged yet')
        return 0
    for key in sorted(runs):
        chars, ch, steps = key
        print(f'\ncorpus {chars or "full":>9}  chunk {ch}  steps {steps}')
        print(f'{"arm":<20}{"seed":>5}{"params":>11} '
              + ' '.join(f'{f"<={t}":>12}' for t in THRESHOLDS) + f'{"best":>8}')
        print('-' * (44 + 13 * len(THRESHOLDS)))
        for arm, seed, npar, pts in sorted(runs[key]):
            hit, best = {}, 99.
            for tok, bpc, _ in pts:
                best = min(best, bpc)
                for th in THRESHOLDS:
                    if th not in hit and bpc <= th:
                        hit[th] = tok
            cells = ' '.join(f'{hit[t]:>12,.0f}' if t in hit else f'{"--":>12}'
                             for t in THRESHOLDS)
            print(f'{arm:<20}{seed:>5}{npar:>11,} {cells}{best:>8.3f}')
    print('\n  "--" = never reached within budget. NOT a score.')
    return 0


def main():
    t0 = time.time()
    if MODE == 'table':
        return table_from_runs()
    if not os.path.exists(CORPUS):
        print(f'corpus not found at {CORPUS}')
        return 1
    data, V = load()
    print(f'EFFICIENCY  enwik8: {len(data):,} chars, vocab {V} '
          f'({time.time()-t0:.0f}s to encode)')
    print(f'target {TARGET:,} params, ctx {CTX}, batch {BS}, {NL} layers, '
          f'{STEPS} steps = {STEPS*BS*CTX:,} tokens\n')
    split = int(0.95 * len(data))
    tr, va = data[:split], data[split:]

    want = [a for a in os.environ.get('ARMS', ','.join(ARMS)).split(',') if a in ARMS]
    specs = []
    print(f'{"arm":<14} {"d":>5} {"params":>12}  parameter match')
    print('-' * 52)
    for name in want:
        ctor, full = ARMS[name]
        d, n = sized(ctor, full, V, TARGET, NL)
        specs.append((name, ctor, full, d, n))
        print(f'{name:<14} {d:>5} {n:>12,}  {n/TARGET:>5.2f}x target', flush=True)

    if MODE == 'probe':
        print(f'\nthroughput probe (10 steps each, ctx {CTX}, batch {BS})')
        g = torch.Generator().manual_seed(0)
        total = 0.0
        for name, ctor, full, d, n in specs:
            m = ctor(V, d, NL, full)
            opt = torch.optim.AdamW(m.parameters(), lr=LR)
            i = torch.randint(0, len(tr) - CTX - 1, (BS,), generator=g)
            x = torch.stack([tr[j:j + CTX] for j in i]).long()
            y = torch.stack([tr[j + 1:j + CTX + 1] for j in i]).long()
            t1 = time.time()
            for _ in range(10):
                loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
            per = (time.time() - t1) / 10
            hrs = per * STEPS / 3600
            total += hrs * SEEDS
            print(f'  {name:<14} {per:>6.2f} s/step   {hrs:>5.1f} h x {SEEDS} seed(s)   '
                  f'state/cache {m.state_floats():>9,} floats', flush=True)
        print(f'\n  FULL RUN COST: {total:.1f} h for {len(specs)} arms x {SEEDS} seed(s).')
        print('  set MODE=full to run it. Reduce with STEPS, TARGET, BS, or ARMS.')
        return 0

    print()
    res = []
    for name, ctor, full, d, n in specs:
        for sd in range(SEEDS):
            res.append(train(name, ctor, full, d, NL, n, tr, va, V, sd))
    print(f'\n{time.time()-t0:.0f}s total\n')

    for unit, idx, fmt in (('TOKENS', 0, lambda v: f'{v:>13,.0f}'),
                           ('HOURS', 1, lambda v: f'{v:>13.2f}')):
        print(f'{unit} TO REACH EACH bpc THRESHOLD  (lower is better)')
        print(f'{"arm":<14} {"params":>10} ' + ' '.join(f'{f"<={t}":>13}' for t in THRESHOLDS)
              + f'{"best":>8}')
        print('-' * (32 + 14 * len(THRESHOLDS)))
        for r in res:
            cells = ' '.join(fmt(r['hit'][t][idx]) if t in r['hit'] else f'{"--":>13}'
                             for t in THRESHOLDS)
            print(f'{r["name"]:<14} {r["params"]:>10,} {cells}{r["best"]:>8.3f}')
        print('  "--" means the arm never reached that bpc within the budget. It is NOT a score.\n')
    print('  The two tables are the two halves of the goal. Saryu winning the first and losing')
    print('  the second means it learns from less data but does not make intelligence cheaper,')
    print('  which is a materially different result from winning outright -- say so if it happens.')

    print('\nREAD, against the registered predictions')
    by = {r['name']: r for r in res}
    for th in THRESHOLDS:
        reached = [n for n in by if th in by[n]['hit']]
        if len(reached) < 2:
            print(f'  bpc <= {th}: reached by {len(reached)} arm(s) -- VOID, says nothing.')
            continue
        order = sorted(reached, key=lambda n: by[n]['hit'][th][0])
        print(f'  bpc <= {th} by tokens: '
              + '  <  '.join(f'{n} ({by[n]["hit"][th][0]:,.0f})' for n in order))
    if 'saryu' in by and 'transformer' in by:
        common = [t for t in THRESHOLDS if t in by['saryu']['hit'] and t in by['transformer']['hit']]
        if not common:
            print('  P1 UNDECIDED: saryu and the transformer share no threshold within budget.')
        else:
            t = common[-1]
            (s, sh), (x, xh) = by['saryu']['hit'][t], by['transformer']['hit'][t]
            print(f'  P1 at bpc<={t}: saryu {s:,.0f} vs transformer {x:,.0f} tokens '
                  f'-> {"HOLDS" if s < x else "FALSIFIED"}')
            if s >= x:
                print('     The pre-registered falsifier: the efficiency claim is dead as stated,')
                print('     and the case rests on the group-theoretic results and constant-cost')
                print('     inference alone -- not on learning more from less.')
            print(f'  and on WALL-CLOCK at the same threshold: saryu {sh:.2f} h vs transformer '
                  f'{xh:.2f} h -> {"cheaper" if sh < xh else "MORE EXPENSIVE"}')
            if s < x <= 0 or (s < x and sh >= xh):
                print('     Saryu learns from fewer tokens but costs more wall-clock to get there.')
                print('     That is sample efficiency WITHOUT cheap intelligence: the data half of')
                print('     the goal holds and the compute half does not. Report it as exactly that.')
    if 'saryu' in by and 'saryu-plain' in by:
        common = [t for t in THRESHOLDS
                  if t in by['saryu']['hit'] and t in by['saryu-plain']['hit']]
        if common:
            t = common[-1]
            a, b = by['saryu']['hit'][t][0], by['saryu-plain']['hit'][t][0]
            print(f'  P3 at bpc<={t}: memory+gate {a:,.0f} vs plain {b:,.0f} tokens -> '
                  f'{"HOLDS" if a <= b else "FALSIFIED -- this session\'s additions COST efficiency"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
