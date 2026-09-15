"""Glass-box inference UI for the 5M-parameter Saryu LM (v4b, enwik8, bpc ~1.78).

Watch the state while it writes. A stdlib-only HTTP server (no Flask, no pip) serves a
single embedded page; behind it runs a STREAMING stateful forward -- one token at a time,
O(1) state, no re-reading the context -- that reports, per generated character:

    erasure      -Sum_{layers,heads} dh * log(1-g)   (log-volume the write path destroys)
    bracket      L3H1 raw state projected on a calibrated "inside [[...]]" direction
    gates        mean per-head gate g, one number per layer
    entropy      prediction entropy of the T=1 softmax, in bits

The server holds ONE persistent session. Each POST /gen ingests the new prompt into the
state the previous turn left behind and continues from there; the model never re-reads a
character it has already seen, so turn k costs exactly what turn 1 cost. The transcript
lives in the browser -- the model's only memory of it is the state vectors. POST /reset
puts the state back to h0.

Usage:
    python scripts/serve.py --selftest      # parity + continuity + smoke, writes the cache
    python scripts/serve.py --smoke         # live-server smoke test only
    python scripts/serve.py                 # serve on http://127.0.0.1:8471
    python scripts/serve.py --port 9000

Conventions: layer/head indices are 0-based, so "L3H1" is mix[3], head 1 -- the same
labelling lm_tda.py prints. The bracket state is read PRE-hnorm (the raw recurrent state),
also as in lm_tda.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
torch.set_grad_enabled(False)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(ROOT, 'checkpoints', 'saryu_v4b_last.pt')
CORPUS = os.path.join(ROOT, 'corpus', 'enwik8')
CACHE = os.path.join(HERE, 'saryu_ui_cache.npz')
VOCAB = os.path.join(ROOT, 'saryu', 'enwik8_vocab.json')        # so the corpus is optional
CALIB = os.path.join(HERE, 'serve_calibration.json')             # bracket direction, same reason
SMOKE_PORT = 8931                # spare port used only by --smoke / --selftest

# ------------------------------------------------------------------ model classes
sys.path.insert(0, ROOT)
import saryu.model as _m  # noqa: E402


# ------------------------------------------------------------------ vocab (cached)
def keep_from_corpus(vocab_min=100):
    raw = open(CORPUS, 'rb').read().decode('utf-8', errors='ignore')
    cps = np.frombuffer(raw.encode('utf-32-le'), dtype=np.uint32)
    uniq, counts = np.unique(cps, return_counts=True)
    return raw, np.sort(uniq[counts >= vocab_min])


def load_cache():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    return {}


def save_cache(d):
    np.savez(CACHE, **d)


_cache = load_cache()
_raw_text = None
if 'keep' in _cache:
    keep = _cache['keep']
elif os.path.exists(VOCAB):
    keep = np.array(json.load(open(VOCAB))['codepoints'], dtype=np.uint32)
else:
    print('building vocab from corpus (first run)...', flush=True)
    _raw_text, keep = keep_from_corpus()
    _cache['keep'] = keep
    save_cache(_cache)

stoi = {int(c): i + 1 for i, c in enumerate(keep)}
itos = {i + 1: chr(int(c)) for i, c in enumerate(keep)}
itos[0] = '?'
V = len(keep) + 1

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
model = _m.SaryuV3LM(V, ck['d'], 4, use_vemb=True)
model.load_state_dict(ck['state'])
model.eval()
D = ck['d']
NL = model.nl
H = _m.NHH
DH = D // H
NH = _m.NH
NPARAM = sum(p.numel() for p in model.parameters())
print('loaded arm {!r}: {:,} params, vocab {}, d={} L={} H={} dh={}'.format(
    ck['arm'], NPARAM, V, D, NL, H, DH), flush=True)


def encode(text):
    return [stoi.get(ord(c), 0) for c in text]


# ------------------------------------------------------------------ streaming forward
class StreamState:
    """Per-layer recurrent state s [H,dh] plus the rolling 4-slot causal conv buffer.

    The batch path is Conv1d(d,d,4,padding=3,groups=d) sliced to [:, :, :L], i.e.
        y[t] = sum_k w[k] * z[t-3+k] + bias,   z[j<0] = 0
    so a zero-initialised buffer (slot 0 oldest = t-3, slot 3 newest = t) reproduces
    the batch output exactly from position 0 -- no warm-up carve-out.
    """

    def __init__(self, mdl):
        self.m = mdl
        self.blocks = list(mdl.mix)
        self.ffns = list(mdl.ffn)
        # precompute the depthwise conv weight as [4, d] so a token costs one broadcast mul
        self.convw = [b.conv.weight[:, 0, :].t().contiguous() for b in self.blocks]
        self.convb = [b.conv.bias for b in self.blocks]
        self.reset()

    def reset(self):
        self.s = [b.h0.detach().clone() for b in self.blocks]          # [H,dh] each
        self.buf = [torch.zeros(4, self.m.d) for _ in self.blocks]     # [4,d] each
        self.n = 0

    def clone(self):
        o = StreamState.__new__(StreamState)
        o.m, o.blocks, o.ffns = self.m, self.blocks, self.ffns
        o.convw, o.convb = self.convw, self.convb
        o.s = [t.clone() for t in self.s]
        o.buf = [t.clone() for t in self.buf]
        o.n = self.n
        return o

    def step(self, tok, telemetry=True):
        """Advance one token. Returns (logits [V], telemetry dict or None)."""
        ti = torch.tensor([tok])
        x = self.m.emb(ti)[0]
        ve = self.m.vemb(ti)[0] if self.m.vemb is not None else None
        erase = 0.0
        gates = []
        for li, (blk, ffn) in enumerate(zip(self.blocks, self.ffns)):
            z0 = blk.ln(x)
            buf = self.buf[li]
            buf[:3] = buf[1:].clone()
            buf[3] = z0
            z = (self.convw[li] * buf).sum(0) + self.convb[li]         # [d]
            u = F.normalize(blk.v_proj(z).view(H, NH, DH), dim=-1)
            beta = 1.0 - torch.cos(blk.b_proj(z).view(H, NH))
            g = ((1.0 - torch.cos(blk.g_proj(z))) / 2.0).clamp(max=0.90)   # [H]
            cz = blk.c_proj(z)
            if ve is not None:
                cz = cz + ve
            a = 1.0 - g
            b = g[:, None] * cz.view(H, DH)
            s = self.s[li]
            for i in range(NH):                                        # n_h reflections
                ui = u[:, i, :]
                s = s - beta[:, i:i + 1] * (s * ui).sum(-1, keepdim=True) * ui
            s = a[:, None] * s + b                                     # the gated write
            self.s[li] = s
            sn = blk.hnorm(s).reshape(self.m.d)
            x = x + blk.out(sn * F.silu(blk.og(z)))
            x = x + ffn(x)
            if telemetry:
                erase += DH * float(-torch.log(a).sum())
                gates.append(float(g.mean()))
        logits = self.m.head(self.m.lnf(x))
        self.n += 1
        tel = None
        if telemetry:
            p = torch.softmax(logits, -1)
            ent = float(-(p * torch.log2(p.clamp_min(1e-12))).sum())
            tel = {'erasure': erase, 'gates': gates, 'entropy': ent,
                   'bracket': float(self.s[3][1] @ BDIR) if BDIR is not None else 0.0}
        return logits, tel

    def l3h1(self):
        return self.s[3][1].clone()


# ------------------------------------------------------------------ bracket calibration
BDIR = None      # unit vector in R^dh; L3H1 state . BDIR ~ "am I inside [[...]]"
BDIR_DPRIME = float('nan')
NW_CAL, LW_CAL, BURN_CAL = 12, 256, 32


def bracket_labels(t):
    """lm_tda.py's labelling: depth of [[...]] nesting, tracked from the raw text."""
    depth, lab, i = 0, [], 0
    while i < len(t):
        if t[i:i + 2] == '[[':
            depth += 1
            i += 2
            lab += [min(depth, 3)] * 2
            continue
        if t[i:i + 2] == ']]' and depth > 0:
            depth -= 1
            i += 2
            lab += [min(depth + 1, 3)] * 2
            continue
        lab.append(min(depth, 3))
        i += 1
    return np.array(lab[:len(t)])


@torch.no_grad()
def collect_l3h1(idx):
    """Mirror SaryuV3LM.forward, capturing layer-3 head-1 raw (pre-hnorm) states."""
    x = model.emb(idx)
    ve = model.vemb(idx) if model.vemb is not None else None
    out = None
    for li, (blk, ffn) in enumerate(zip(model.mix, model.ffn)):
        B, L, _ = x.shape
        z = blk.ln(x)
        z = blk.conv(z.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u, beta, a, b = blk.mix(z, ve)
        uf = u.permute(0, 2, 1, 3, 4).reshape(B * blk.H, L, blk.nh, blk.dh)
        bf = beta.permute(0, 2, 1, 3).reshape(B * blk.H, L, blk.nh)
        af = a.permute(0, 2, 1).reshape(B * blk.H, L)
        bbf = b.permute(0, 2, 1, 3).reshape(B * blk.H, L, blk.dh)
        h0 = blk.h0[None].expand(B, blk.H, blk.dh).reshape(B * blk.H, blk.dh)
        s = _m.chunkwise(h0, uf, bf, af, bbf).view(B, blk.H, L, blk.dh)
        if li == 3:
            out = s[:, 1].clone()                                    # [B,L,dh]
        sn = blk.hnorm(s.permute(0, 2, 1, 3)).reshape(B, L, blk.d)
        x = x + blk.out(sn * F.silu(blk.og(z)))
        x = x + ffn(x)
    return out


def calibrate(force=False):
    """Direction = normalize(mean(inside [[..]]) - mean(outside)) for the L3H1 state."""
    global BDIR, BDIR_DPRIME, _raw_text
    if not force and 'bdir' in _cache:
        BDIR = torch.tensor(_cache['bdir'].astype(np.float32))
        BDIR_DPRIME = float(_cache.get('bdprime', np.array(float('nan'))))
        print('bracket direction: loaded from cache (dprime={:.2f})'.format(BDIR_DPRIME), flush=True)
        return
    if not force and os.path.exists(CALIB):
        c = json.load(open(CALIB))
        BDIR = torch.tensor(np.array(c['bdir'], dtype=np.float32))
        BDIR_DPRIME = float(c['dprime'])
        print('bracket direction: loaded from {} (dprime={:.2f})'.format(
            os.path.basename(CALIB), BDIR_DPRIME), flush=True)
        return
    t0 = time.time()
    if _raw_text is None:
        print('reading corpus for calibration...', flush=True)
        _raw_text, _ = keep_from_corpus()
    raw = _raw_text
    rng = np.random.default_rng(0)
    val0 = int(0.95 * len(raw))
    texts, labs = [], []
    for _ in range(NW_CAL):
        s0 = int(rng.integers(val0, len(raw) - LW_CAL - 1))
        t = raw[s0:s0 + LW_CAL]
        texts.append(t)
        labs.append(bracket_labels(t))
    idx = torch.tensor([encode(t) for t in texts])
    S = collect_l3h1(idx)[:, BURN_CAL:].reshape(-1, DH).numpy()
    lab = np.concatenate([l[BURN_CAL:] for l in labs])
    A, B_ = S[lab == 0], S[lab >= 1]
    if len(A) < 10 or len(B_) < 10:
        raise RuntimeError('calibration windows contained too few bracket positions')
    diff = B_.mean(0) - A.mean(0)
    sp = 0.5 * (A.std(0).mean() + B_.std(0).mean()) + 1e-9
    BDIR_DPRIME = float(np.linalg.norm(diff) / sp)
    BDIR = torch.tensor((diff / (np.linalg.norm(diff) + 1e-9)).astype(np.float32))
    _cache['bdir'] = BDIR.numpy()
    _cache['bdprime'] = np.array(BDIR_DPRIME)
    save_cache(_cache)
    print('bracket direction calibrated on {}x{} chars in {:.1f}s: dprime={:.2f}, '
          'inside n={} outside n={}'.format(NW_CAL, LW_CAL, time.time() - t0,
                                            BDIR_DPRIME, len(B_), len(A)), flush=True)


# ------------------------------------------------------------------ session
PROMPT_MAX = 4000
STATE_FLOATS = NL * (H * DH + 4 * D)
STATE_DESC = '{}L x (s[{},{}] + conv[4,{}]) = {:,} floats'.format(NL, H, DH, D, STATE_FLOATS)


class Session:
    """One continuing conversation. The past survives ONLY as recurrent state.

    Persisted across /gen calls: the StreamState (per-layer s [H,dh] and the rolling
    depthwise-conv buffer [4,d]), plus `last` -- the logits and telemetry of the most
    recent step, which is exactly what decides the next character. No text is kept
    server-side: the model never re-reads its own transcript, so the cost of turn k
    is independent of k. That is the whole point of the O(1) state.
    """

    def __init__(self):
        self.st = StreamState(model)
        self.reset()

    def reset(self):
        self.st.reset()
        self.last = None            # (logits [V], telemetry dict) of the newest step
        self.chars = 0              # steps taken since reset == chars ingested
        self.turns = 0
        self.erasure = 0.0          # cumulative nats destroyed since reset
        self.born = time.time()

    def ingest(self, tok, rec=None, kind=0, capture=None):
        """One step of the SAME state. kind 0 = user prompt char, 1 = model's own char."""
        lg, tel = self.st.step(tok, telemetry=True)
        self.last = (lg, tel)
        self.chars += 1
        self.erasure += tel['erasure']
        if capture is not None:
            capture.append(lg)
        if rec is not None:
            rec['erasure'].append(tel['erasure'])
            rec['bracket'].append(tel['bracket'])
            rec['gates'].append(tel['gates'])
            rec['entropy'].append(tel['entropy'])
            rec['kind'].append(kind)
        return lg, tel

    def stats(self):
        return {'chars_ingested': self.chars, 'turns': self.turns,
                'session_erasure': self.erasure, 'state_floats': STATE_FLOATS,
                'state_desc': STATE_DESC, 'age_s': time.time() - self.born}


SESSION = Session()


# ------------------------------------------------------------------ generation
def generate(sess, prompt, n_chars=200, temperature=0.8, top_k=40, seed=None,
             capture=None, toks_out=None):
    """Continue `sess`: ingest the new prompt into the EXISTING state, then write.

    `capture` (list) collects the logits of every step performed, in order; `toks_out`
    collects the token ids stepped. Together they are the continuity theorem's witness:
    replaying toks_out through a fresh StreamState must reproduce capture exactly.
    """
    n_chars = max(1, min(400, int(n_chars)))
    temperature = float(min(3.0, max(0.05, temperature)))
    top_k = int(min(V, max(1, top_k)))
    if seed is not None:
        torch.manual_seed(int(seed))
    prompt = (prompt or '')[:PROMPT_MAX]
    if not prompt and sess.last is None:
        prompt = '\n'                          # a fresh state has nothing to predict from
    step0 = sess.chars
    rec = {'erasure': [], 'bracket': [], 'gates': [], 'entropy': [], 'kind': []}
    for t in encode(prompt):                   # the new prompt enters the SAME state
        sess.ingest(t, rec, 0, capture)
        if toks_out is not None:
            toks_out.append(t)
    chars, era, brk, gts, ent = [], [], [], [], []
    t0 = time.time()
    for _ in range(n_chars):
        logits, tel = sess.last                # the step that DECIDES this character
        era.append(tel['erasure'])
        brk.append(tel['bracket'])
        gts.append(tel['gates'])
        ent.append(tel['entropy'])
        lg = logits / temperature
        v, _ix = torch.topk(lg, top_k)
        lg = lg.masked_fill(lg < v[-1], -1e30)
        nxt = int(torch.multinomial(torch.softmax(lg, -1), 1))
        chars.append(itos.get(nxt, '?'))
        sess.ingest(nxt, rec, 1, capture)      # what it just wrote becomes state too
        if toks_out is not None:
            toks_out.append(nxt)
    dt = time.time() - t0
    sess.turns += 1
    out = {'prompt': prompt, 'text': ''.join(chars), 'erasure': era, 'bracket': brk,
           'gates': gts, 'entropy': ent, 'ms_per_token': 1000.0 * dt / n_chars,
           'dprime': BDIR_DPRIME, 'steps': rec, 'step0': step0,
           'n_prompt': len(prompt), 'n_gen': n_chars}
    out.update(sess.stats())
    return out


# ------------------------------------------------------------------ verification
def selftest():
    print('\n=== (a) ast.parse ===', flush=True)
    import ast
    ast.parse(open(os.path.abspath(__file__), encoding='utf-8').read())
    print('ast.parse: OK', flush=True)

    print('\n=== (b) PARITY batch vs streaming ===', flush=True)
    text = ('The [[history of India]] begins with the [[Indus Valley]] civilisation, '
            'and continues in 1947 CE.')[:96]
    assert len(text) == 96, len(text)
    idx = torch.tensor([encode(text)])
    lb = model(idx)[0]                                   # [96,V]
    st = StreamState(model)
    rows = []
    for t in idx[0].tolist():
        lg, _ = st.step(t, telemetry=False)
        rows.append(lg)
    ls = torch.stack(rows)
    d = (ls - lb).abs()
    print('test string ({} chars): {!r}'.format(len(text), text), flush=True)
    print('max |logits_stream - logits_batch| over ALL positions 0..95 : {:.3e}'.format(
        float(d.max())), flush=True)
    print('max |diff| over positions 8..95                             : {:.3e}'.format(
        float(d[8:].max())), flush=True)
    print('max |diff| at position 0                                    : {:.3e}'.format(
        float(d[0].max())), flush=True)
    print('per-position max diff, first 8: ' + ' '.join(
        '{:.1e}'.format(float(x)) for x in d[:8].max(-1).values), flush=True)
    assert float(d.max()) < 1e-3, 'PARITY FAILED'
    print('PARITY: PASS (< 1e-3 from position 0, no warm-up carve-out)', flush=True)

    print('\n=== (c) streaming speed, 200 tokens ===', flush=True)
    st = StreamState(model)
    st.step(stoi.get(ord('T'), 0), telemetry=True)        # warm
    toks = encode('the quick brown fox ' * 10)[:200]
    t0 = time.time()
    for t in toks:
        st.step(t, telemetry=True)
    dt = time.time() - t0
    print('{:.2f} ms/token  ({:.2f}s for 200 tokens, torch threads={})'.format(
        1000 * dt / 200, dt, torch.get_num_threads()), flush=True)

    print('\n=== calibration ===', flush=True)
    calibrate()
    print('cache: {}'.format(CACHE), flush=True)

    continuity_test()
    smoke(port=SMOKE_PORT)
    print('\nALL SELFTESTS PASSED', flush=True)
    return 1000 * dt / 200


def continuity_test():
    """The core theorem of the session feature.

    Two /gen calls against ONE session must be indistinguishable from a single fresh
    stateful pass over the concatenated character stream P1 + genA + P2 + genB. Compared
    on token ids (never on re-encoded display text: itos maps the unknown id 0 to '?',
    which would round-trip to a different id and hide a real divergence).
    """
    print('\n=== (d) CONTINUITY PARITY: session == fresh pass over P1+genA+P2 ===', flush=True)
    p1 = 'The [[history of India]] begins with '
    p2 = ' Meanwhile, in [[Europe]] the '
    sess = Session()
    cap, toks = [], []
    a = generate(sess, p1, n_chars=64, temperature=0.05, top_k=1, seed=1234,
                 capture=cap, toks_out=toks)
    na = len(cap)
    b = generate(sess, p2, n_chars=64, temperature=0.05, top_k=1, seed=5678,
                 capture=cap, toks_out=toks)
    stream = p1 + a['text'] + p2 + b['text']
    print('call A: prompt {!r} -> {} gen chars'.format(p1, len(a['text'])), flush=True)
    print('call B: prompt {!r} -> {} gen chars (SAME session)'.format(p2, len(b['text'])),
          flush=True)
    print('session steps: {} (A) + {} (B) = {}; chars_ingested={}; stream len={}'.format(
        na, len(cap) - na, len(cap), sess.chars, len(stream)), flush=True)
    assert len(cap) == len(toks) == sess.chars == len(stream), 'step bookkeeping mismatch'
    assert toks == encode(stream), 'token stream != encode(displayed stream)'
    assert b['step0'] == na, 'turn B does not start where turn A ended'

    ref = StreamState(model)                       # one fresh pass over the whole stream
    rows = [ref.step(t, telemetry=False)[0] for t in toks]
    ls = torch.stack(rows)
    lc = torch.stack(cap)
    d = (lc - ls).abs()
    print('max |logits_session - logits_fresh| over ALL {} positions : {:.3e}'.format(
        len(toks), float(d.max())), flush=True)
    print('max |diff| inside turn A (0..{})                          : {:.3e}'.format(
        na - 1, float(d[:na].max())), flush=True)
    print('max |diff| inside turn B ({}..{})                       : {:.3e}'.format(
        na, len(toks) - 1, float(d[na:].max())), flush=True)
    print('max |diff| at the turn boundary (position {})             : {:.3e}'.format(
        na, float(d[na].max())), flush=True)
    assert float(d.max()) < 1e-3, 'CONTINUITY PARITY FAILED'

    ds = max(float((sess.st.s[li] - ref.s[li]).abs().max()) for li in range(NL))
    db = max(float((sess.st.buf[li] - ref.buf[li]).abs().max()) for li in range(NL))
    print('max |s_session - s_fresh|   over all {} layers            : {:.3e}'.format(NL, ds),
          flush=True)
    print('max |buf_session - buf_fresh| over all {} layers          : {:.3e}'.format(NL, db),
          flush=True)
    assert ds < 1e-3 and db < 1e-3, 'CARRIED STATE DIVERGED'
    print('CONTINUITY: PASS (< 1e-3 at every position; the past is only state)', flush=True)

    print('\n=== (d2) RESET restores h0 exactly ===', flush=True)
    sess.reset()
    fresh = StreamState(model)
    rs = max(float((sess.st.s[li] - fresh.s[li]).abs().max()) for li in range(NL))
    rb = max(float((sess.st.buf[li] - fresh.buf[li]).abs().max()) for li in range(NL))
    print('max |s_reset - h0| = {:.3e}, max |buf_reset - 0| = {:.3e}, chars={}, turns={}, '
          'erasure={:.1f}'.format(rs, rb, sess.chars, sess.turns, sess.erasure), flush=True)
    assert rs == 0.0 and rb == 0.0 and sess.chars == 0 and sess.last is None
    print('RESET: PASS (bit-exact h0, zero conv buffer, counters cleared)', flush=True)


def smoke(port=8931, timeout=240.0):
    """Start a real server in a subprocess: GET /, POST /gen twice, POST /reset."""
    import socket
    import subprocess
    import tempfile
    import urllib.request
    print('\n=== (e) SMOKE: live server on 127.0.0.1:{} ==='.format(port), flush=True)

    def post(path, obj):
        req = urllib.request.Request('http://127.0.0.1:{}{}'.format(port, path),
                                     data=json.dumps(obj).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'},
                                     method='POST')
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode('utf-8'))

    fd, logp = tempfile.mkstemp(prefix='saryu_smoke_', suffix='.log')
    lf = os.fdopen(fd, 'w+', encoding='utf-8', errors='replace')
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), '--port', str(port)],
                            cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError('server exited early, rc={}'.format(proc.returncode))
            try:
                socket.create_connection(('127.0.0.1', port), 0.5).close()
                break
            except OSError:
                if time.time() - t0 > timeout:
                    raise RuntimeError('server did not listen within {}s'.format(timeout))
        print('server pid {} listening after {:.1f}s'.format(proc.pid, time.time() - t0),
              flush=True)

        with urllib.request.urlopen('http://127.0.0.1:{}/'.format(port), timeout=30) as r:
            page = r.read().decode('utf-8')
        for needle in ('NEW SESSION', 'id="transcript"', 'id="sesschars"', 'id="sesserase"'):
            assert needle in page, 'page missing {!r}'.format(needle)
        print('GET /            : {} bytes, transcript + NEW SESSION present'.format(len(page)),
              flush=True)

        z = post('/reset', {})
        assert z['ok'] and z['chars_ingested'] == 0 and z['turns'] == 0
        print('POST /reset      : chars_ingested=0 turns=0 state={}'.format(z['state_desc']),
              flush=True)

        pa = 'The [[history of India]] begins with '
        a = post('/gen', {'prompt': pa, 'n_chars': 32, 'temperature': 0.05, 'top_k': 1})
        print('POST /gen #1     : step0={} n_prompt={} steps={} chars_ingested={} '
              'turns={} session_erasure={:.1f}'.format(
                  a['step0'], a['n_prompt'], len(a['steps']['kind']), a['chars_ingested'],
                  a['turns'], a['session_erasure']), flush=True)
        print('           text  : {!r}'.format(a['text']), flush=True)
        assert a['step0'] == 0 and a['chars_ingested'] == len(pa) + 32
        assert len(a['steps']['kind']) == len(pa) + 32 == len(a['steps']['erasure'])
        assert sum(a['steps']['kind']) == 32 and len(a['erasure']) == 32

        pb = ' and then '
        b = post('/gen', {'prompt': pb, 'n_chars': 32, 'temperature': 0.05, 'top_k': 1})
        print('POST /gen #2     : step0={} n_prompt={} steps={} chars_ingested={} '
              'turns={} session_erasure={:.1f}'.format(
                  b['step0'], b['n_prompt'], len(b['steps']['kind']), b['chars_ingested'],
                  b['turns'], b['session_erasure']), flush=True)
        print('           text  : {!r}'.format(b['text']), flush=True)
        assert b['step0'] == a['chars_ingested'], 'turn 2 did not resume where turn 1 ended'
        assert b['chars_ingested'] == a['chars_ingested'] + len(pb) + 32
        assert b['turns'] == 2 and b['session_erasure'] > a['session_erasure']
        assert len(b['steps']['kind']) == len(pb) + 32

        post('/reset', {})
        c = post('/gen', {'prompt': pa, 'n_chars': 32, 'temperature': 0.05, 'top_k': 1})
        assert c['text'] == a['text'], 'reset did not restore the initial state'
        assert c['chars_ingested'] == a['chars_ingested'] and c['turns'] == 1
        print('POST /reset+/gen : greedy replay reproduces turn #1 exactly -> reset is real',
              flush=True)

        post('/reset', {})
        d2 = post('/gen', {'prompt': pb, 'n_chars': 32, 'temperature': 0.05, 'top_k': 1})
        print('history matters  : same prompt {!r} with vs without history ->'.format(pb),
              flush=True)
        print('           with  : {!r}'.format(b['text']), flush=True)
        print('           without: {!r}'.format(d2['text']), flush=True)
        print('SMOKE: PASS', flush=True)
    finally:
        proc.kill()
        try:
            proc.wait(timeout=15)
        except Exception:                                            # noqa: BLE001
            pass
        lf.flush()
        lf.seek(0)
        tail = lf.read().strip().splitlines()[-4:]
        lf.close()
        try:
            os.remove(logp)
        except OSError:
            pass
        print('server log tail:', flush=True)
        for ln in tail:
            print('  | ' + ln, flush=True)


# ------------------------------------------------------------------ the page
PAGE = '''<!doctype html>
<html><head><meta charset="utf-8"><title>Saryu 5M - a session with no context window</title>
<style>
:root{--bg:#0c0e11;--panel:#14181d;--line:#232a32;--fg:#c9d3de;--dim:#6b7887;
      --hot:#ff8a4c;--cool:#4cc9f0;--gate:#8ce99a;--warn:#f7b955;
      --user:#4cc9f0;--model:#ffb37a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:13px/1.5 "Cascadia Mono",Consolas,"DejaVu Sans Mono",monospace}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
       align-items:baseline;gap:14px;flex-wrap:wrap}
h1{font-size:14px;margin:0;letter-spacing:.14em;text-transform:uppercase;font-weight:600}
.sub{color:var(--dim);font-size:11px}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:18px;padding:18px 20px;
     align-items:start}
@media(max-width:900px){main{grid-template-columns:1fr}}
textarea{width:100%;background:#0f1216;color:var(--fg);border:1px solid var(--line);
         border-radius:4px;padding:9px;font:inherit;resize:vertical;min-height:56px}
.row{display:flex;gap:14px;align-items:center;margin:10px 0;flex-wrap:wrap}
label{color:var(--dim);font-size:11px;letter-spacing:.06em}
input[type=range]{width:120px;accent-color:var(--hot)}
input[type=number]{width:70px;background:#0f1216;color:var(--fg);border:1px solid var(--line);
                   border-radius:3px;padding:4px;font:inherit}
button{background:var(--hot);color:#12161a;border:0;border-radius:4px;padding:8px 18px;
       font:inherit;font-weight:700;letter-spacing:.08em;cursor:pointer}
button.ghost{background:transparent;color:var(--dim);border:1px solid var(--line);
             font-weight:600}
button.ghost:hover:enabled{color:var(--fg);border-color:#3a4550}
button:disabled{background:#39424c;color:#7d8895;cursor:default}
button.ghost:disabled{background:transparent;color:#4a545f;border-color:#1b2129}
#transcript{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:14px;
     white-space:pre-wrap;word-break:break-word;min-height:300px;max-height:58vh;
     overflow:auto;font-size:13.5px}
#transcript .u{color:var(--user)}
#transcript .m{color:var(--model)}
#transcript .p{color:var(--dim)}
#transcript .cur{background:var(--hot);color:#12161a}
.legend{display:flex;gap:16px;color:var(--dim);font-size:10.5px;margin:8px 0 2px}
.legend b{font-weight:600}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;
    vertical-align:-1px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:5px;
       padding:12px;margin-bottom:14px}
.panel.sess{border-color:#2c3a44}
.ptitle{font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--dim);
        display:flex;justify-content:space-between;margin-bottom:9px}
.val{color:var(--fg);font-variant-numeric:tabular-nums}
.big{font-size:22px;color:var(--cool);font-variant-numeric:tabular-nums;line-height:1.15}
.kv{display:flex;justify-content:space-between;margin:3px 0;font-size:11px}
.kv b{font-weight:600;color:var(--fg);font-variant-numeric:tabular-nums}
.kv span{color:var(--dim)}
.meter{height:16px;background:#0b0e11;border:1px solid var(--line);border-radius:3px;
       overflow:hidden}
.meter i{display:block;height:100%;width:0;background:linear-gradient(90deg,#7a3b1e,var(--hot));
         transition:width .06s linear}
.scale{display:flex;justify-content:space-between;color:var(--dim);font-size:10px;margin-top:4px}
.gaterow{display:flex;align-items:center;gap:8px;margin:5px 0}
.gaterow span{width:22px;color:var(--dim);font-size:10px}
.gbar{flex:1;height:11px;background:#0b0e11;border:1px solid var(--line);border-radius:2px;
      overflow:hidden}
.gbar i{display:block;height:100%;width:0;background:var(--gate);transition:width .06s linear}
.gnum{width:44px;text-align:right;font-size:10.5px;color:var(--fg);font-variant-numeric:tabular-nums}
svg{display:block;width:100%}
svg.tall{height:78px}
svg.short{height:46px}
.depth{color:var(--warn)}
.note{color:var(--dim);font-size:10.5px;line-height:1.55;margin-top:6px}
.note b{color:var(--fg);font-weight:600}
.err{color:#ff6b6b;font-size:11px;margin-left:8px}
</style></head><body>
<header>
  <h1>Saryu 5M &mdash; a session with no context window</h1>
  <span class="sub" id="meta"></span>
</header>
<main>
 <section>
  <div id="transcript"><span class="p">new session: state = h0, nothing carried.</span></div>
  <div class="legend">
    <span><i class="sw" style="background:#4cc9f0"></i><b>you</b> &mdash; ingested into the state</span>
    <span><i class="sw" style="background:#ffb37a"></i><b>model</b> &mdash; written from the state</span>
  </div>
  <textarea id="prompt">The [[history of India]] begins with </textarea>
  <div class="row">
    <button id="go">GENERATE</button>
    <button id="new" class="ghost">NEW SESSION</button>
    <label>temp <span class="val" id="tv">0.80</span></label>
    <input type="range" id="temp" min="0.05" max="2" step="0.05" value="0.8">
    <label>chars</label><input type="number" id="n" value="200" min="1" max="400">
    <label>top-k</label><input type="number" id="k" value="40" min="1" max="512">
    <span class="err" id="err"></span>
  </div>
  <div class="note"><b>NOTE &mdash; this session has ingested
    <span class="val" id="sesschars">0</span> characters since reset, and the model has
    re-read none of them.</b> Each turn continues from the state the last turn left behind;
    the transcript above lives in your browser, not in the model. Memory is
    <span id="statedesc">O(1)</span> no matter how long the session runs, and the cost of
    character 10,000 equals the cost of character 1.</div>
 </section>
 <aside>
  <div class="panel sess">
    <div class="ptitle"><span>session</span><span class="val" id="sessturns">0</span></div>
    <div class="big"><span id="sesserase">0</span> <span class="sub">nats destroyed</span></div>
    <div class="kv"><span>chars ingested since reset</span><b id="sesschars2">0</b></div>
    <div class="kv"><span>turns</span><b id="sessturns2">0</b></div>
    <div class="kv"><span>carried state</span><b id="statefloats">-</b></div>
    <div class="note">Cumulative erasure over every step since NEW SESSION &mdash; the total
      log-volume this session has thrown away. It only ever goes up; the state is finite,
      so remembering is paid for by forgetting.</div>
  </div>
  <div class="panel">
    <div class="ptitle"><span>erasure &middot; this char</span><span class="val" id="ev">-</span></div>
    <div class="meter"><i id="ebar"></i></div>
    <div class="scale"><span>0</span><span id="emax">-</span></div>
    <svg class="short" id="espark" viewBox="0 0 320 46" preserveAspectRatio="none">
      <polyline id="epline" fill="none" stroke="#ff8a4c" stroke-width="1.2" points=""/>
    </svg>
    <div class="scale"><span>session timeline</span><span id="emaxs">-</span></div>
    <div class="note">Log-volume the gated write destroys this step:
      &minus;&Sigma;<sub>layers,heads</sub> d<sub>h</sub>&thinsp;log(1&minus;g).
      The trace runs across the <b>whole session</b>, not just this turn.</div>
  </div>
  <div class="panel">
    <div class="ptitle"><span>bracket register (L3H1)</span>
      <span class="val">depth <span class="depth" id="dv">0</span></span></div>
    <svg class="tall" id="spark" viewBox="0 0 320 78" preserveAspectRatio="none">
      <line x1="0" y1="39" x2="320" y2="39" stroke="#232a32" stroke-width="1"/>
      <g id="ribbon"></g>
      <polyline id="spline" fill="none" stroke="#4cc9f0" stroke-width="1.4" points=""/>
      <circle id="sdot" cx="0" cy="39" r="2.6" fill="#4cc9f0"/>
    </svg>
    <div class="scale"><span>step 0</span><span id="tlen">-</span></div>
    <div class="note">Layer-3 head-1 raw state on the calibrated inside-[[&hellip;]] direction,
      on the same session-long axis. The bar underneath is who wrote each step
      (teal you, amber model); faint verticals are turn boundaries. The amber number is
      the true bracket depth counted from the transcript.</div>
  </div>
  <div class="panel">
    <div class="ptitle"><span>mean gate per layer</span>
      <span class="val" id="hv">-</span></div>
    <div id="gates"></div>
    <div class="note">g = write strength, 0 carry &rarr; 0.9 overwrite. Right-hand number
      above is prediction entropy in bits (T=1).</div>
  </div>
 </aside>
</main>
<script>
var NS='http://www.w3.org/2000/svg';
var S,G=null,i=0,timer=null;
var gwrap=document.getElementById('gates'),gbars=[],gnums=[];
for(var L=0;L<4;L++){
  var r=document.createElement('div');r.className='gaterow';
  r.innerHTML='<span>L'+L+'</span><div class="gbar"><i></i></div><b class="gnum">-</b>';
  gwrap.appendChild(r);
  gbars.push(r.querySelector('i'));gnums.push(r.querySelector('b'));
}
document.getElementById('prompt').addEventListener('keydown',function(e){
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('go').click();}});
document.getElementById('temp').oninput=function(){
  document.getElementById('tv').textContent=(+this.value).toFixed(2);};

function el(id){return document.getElementById(id);}
function fresh(){return {era:[],brk:[],kind:[],bounds:[],shown:0,eraSum:0,
                         html:'',text:'',turns:0,base:0,np:0};}
S=fresh();

function depthOf(t){var d=0,j=0;while(j<t.length){
  if(t.substr(j,2)=='[['){d++;j+=2;continue;}
  if(t.substr(j,2)==']]'&&d>0){d--;j+=2;continue;}
  j++;}return d;}

function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

function reveal(n){while(S.shown<n){S.eraSum+=S.era[S.shown];S.shown++;}}

function stats(){
  var c=S.shown.toLocaleString();
  el('sesschars').textContent=c;el('sesschars2').textContent=c;
  el('sesserase').textContent=Math.round(S.eraSum).toLocaleString();
  el('sessturns').textContent='turn '+S.turns;el('sessturns2').textContent=S.turns;
}

function ribbon(den){
  var g=el('ribbon');while(g.firstChild)g.removeChild(g.firstChild);
  var n=S.era.length,j=0,k,j2,q;
  while(j<n){k=S.kind[j];j2=j;while(j2<n&&S.kind[j2]===k)j2++;
    q=document.createElementNS(NS,'rect');
    q.setAttribute('x',(320*j/den).toFixed(1));q.setAttribute('y','70');
    q.setAttribute('width',Math.max(0.7,320*(j2-j)/den).toFixed(1));
    q.setAttribute('height','6');
    q.setAttribute('fill',k?'#ffb37a':'#4cc9f0');
    q.setAttribute('opacity',j<S.shown?'0.95':'0.20');
    g.appendChild(q);j=j2;}
  for(var t=1;t<S.bounds.length;t++){
    var x=(320*S.bounds[t]/den).toFixed(1);
    var l=document.createElementNS(NS,'line');
    l.setAttribute('x1',x);l.setAttribute('x2',x);
    l.setAttribute('y1','3');l.setAttribute('y2','66');
    l.setAttribute('stroke','#33404b');l.setAttribute('stroke-width','1');
    g.appendChild(l);}
}

function timeline(){
  var tot=S.era.length,n=S.shown,j,v;
  el('tlen').textContent=tot?('step '+tot):'-';
  if(n<1){el('spline').setAttribute('points','');el('epline').setAttribute('points','');
    while(el('ribbon').firstChild)el('ribbon').removeChild(el('ribbon').firstChild);return;}
  var bm=1e-6,em=1e-6;
  for(j=0;j<n;j++){v=S.brk[j];if(v<0)v=-v;if(v>bm)bm=v;if(S.era[j]>em)em=S.era[j];}
  var den=Math.max(1,tot-1),st=Math.max(1,Math.ceil(n/400)),bp=[],ep=[],px;
  for(j=0;j<n;j+=st){
    px=(320*j/den).toFixed(1);
    bp.push(px+','+(39-33*(S.brk[j]/bm)).toFixed(1));
    ep.push(px+','+(43-40*(S.era[j]/em)).toFixed(1));
  }
  el('spline').setAttribute('points',bp.join(' '));
  el('epline').setAttribute('points',ep.join(' '));
  var dot=el('sdot');
  dot.setAttribute('cx',(320*(n-1)/den).toFixed(1));
  dot.setAttribute('cy',(39-33*(S.brk[n-1]/bm)).toFixed(1));
  el('emaxs').textContent=em.toFixed(0)+' nats peak';
  ribbon(den);
}

function paint(){
  var out=el('transcript'),so=G.text.slice(0,i+1);
  out.innerHTML=S.html+'<span class="m">'+esc(so.slice(0,-1))+
    '</span><span class="cur">'+esc(so.slice(-1))+'</span>';
  out.scrollTop=out.scrollHeight;
}

function draw(){
  reveal(S.base+S.np+i+1);
  paint();
  var e=G.erasure[i];
  el('ev').textContent=e.toFixed(1)+' nats';
  el('ebar').style.width=(100*e/G.emax).toFixed(1)+'%';
  el('emax').textContent=G.emax.toFixed(0)+' nats';
  el('dv').textContent=depthOf(S.text+G.text.slice(0,i+1));
  var g=G.gates[i];
  for(var L=0;L<4;L++){gbars[L].style.width=(100*g[L]/0.9).toFixed(1)+'%';
    gnums[L].textContent=g[L].toFixed(3);}
  el('hv').textContent=G.entropy[i].toFixed(2)+' bits';
  timeline();stats();
}

function busy(b){
  el('go').disabled=b;el('new').disabled=b;
  el('go').textContent=b?'RUNNING...':'GENERATE';
}

function meta(d){
  el('meta').textContent=d.info+(d.ms_per_token?
    (' | '+d.ms_per_token.toFixed(1)+' ms/char streamed'):'');
  el('statedesc').textContent=d.state_desc;
  el('statefloats').textContent=d.state_floats.toLocaleString()+' floats';
}

function run(){
  busy(true);el('err').textContent='';
  if(timer){clearInterval(timer);timer=null;}
  fetch('/gen',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prompt:el('prompt').value,
      n_chars:+el('n').value,
      temperature:+el('temp').value,
      top_k:+el('k').value})})
  .then(function(r){if(!r.ok)throw new Error('HTTP '+r.status);return r.json();})
  .then(function(d){
    if(d.error)throw new Error(d.error);
    G=d;i=0;
    S.base=S.era.length;S.bounds.push(S.base);S.turns=d.turns;S.np=d.n_prompt;
    var k=d.steps.kind;
    for(var j=0;j<k.length;j++){
      S.era.push(d.steps.erasure[j]);S.brk.push(d.steps.bracket[j]);S.kind.push(k[j]);}
    if(d.prompt.length){S.html+='<span class="u">'+esc(d.prompt)+'</span>';S.text+=d.prompt;}
    reveal(S.base+S.np);                       // the prompt is already inside the state
    G.emax=Math.max.apply(null,G.erasure)*1.02+1e-6;
    meta(d);timeline();stats();
    el('prompt').value='';
    timer=setInterval(function(){
      if(i>=G.text.length){
        clearInterval(timer);timer=null;
        S.html+='<span class="m">'+esc(G.text)+'</span>';S.text+=G.text;
        reveal(S.era.length);
        var out=el('transcript');out.innerHTML=S.html;out.scrollTop=out.scrollHeight;
        timeline();stats();busy(false);el('prompt').focus();return;}
      draw();i++;},32);
  })
  .catch(function(e){el('err').textContent=e.message;busy(false);});
}

function newsession(){
  busy(true);el('err').textContent='';
  if(timer){clearInterval(timer);timer=null;}
  fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})
  .then(function(r){if(!r.ok)throw new Error('HTTP '+r.status);return r.json();})
  .then(function(d){
    S=fresh();G=null;i=0;
    el('transcript').innerHTML='<span class="p">new session: state = h0, nothing carried.</span>';
    el('ev').textContent='-';el('ebar').style.width='0%';el('emax').textContent='-';
    el('hv').textContent='-';el('dv').textContent='0';
    for(var L=0;L<4;L++){gbars[L].style.width='0%';gnums[L].textContent='-';}
    meta(d);timeline();stats();busy(false);el('prompt').focus();
  })
  .catch(function(e){el('err').textContent=e.message;busy(false);});
}

el('go').onclick=run;
el('new').onclick=newsession;
newsession();
</script></body></html>
'''


# ------------------------------------------------------------------ server
import http.server                                                   # noqa: E402


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = 'saryu-glassbox/1.0'

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split('?')[0] in ('/', '/index.html'):
            self._send(200, 'text/html; charset=utf-8', PAGE.encode('utf-8'))
        else:
            self._send(404, 'text/plain; charset=utf-8', b'not found')

    def do_POST(self):
        path = self.path.split('?')[0]
        if path not in ('/gen', '/reset'):
            self._send(404, 'text/plain; charset=utf-8', b'not found')
            return
        try:
            n = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(n).decode('utf-8')) if n else {}
            if path == '/reset':
                SESSION.reset()                      # back to h0, zero conv buffer
                r = dict(SESSION.stats())
                r['ok'] = True
            else:
                if req.get('reset'):
                    SESSION.reset()
                r = generate(SESSION, req.get('prompt', ''),
                             n_chars=req.get('n_chars', 200),
                             temperature=req.get('temperature', 0.8),
                             top_k=req.get('top_k', 40),
                             seed=req.get('seed'))
            r['info'] = 'saryu-v3 {} {:,} params  d={} L={} H={} dh={}  bracket d\'={:.2f}'.format(
                ck['arm'], NPARAM, D, NL, H, DH, BDIR_DPRIME)
            body = json.dumps(r).encode('utf-8')
            self._send(200, 'application/json; charset=utf-8', body)
        except Exception as exc:                                     # noqa: BLE001
            body = json.dumps({'error': '{}: {}'.format(type(exc).__name__, exc)}).encode()
            self._send(500, 'application/json; charset=utf-8', body)

    def log_message(self, fmt, *a):
        sys.stderr.write('  %s\n' % (fmt % a))


def serve(port):
    calibrate()
    srv = http.server.HTTPServer(('127.0.0.1', port), Handler)       # loopback only
    print('\nSaryu glass-box UI: http://127.0.0.1:{}/   (Ctrl+C to stop)'.format(port),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nbye', flush=True)
    finally:
        srv.server_close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8471)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--smoke', action='store_true',
                    help='only the live-server smoke test (spare port {})'.format(SMOKE_PORT))
    ap.add_argument('--recalibrate', action='store_true')
    args = ap.parse_args()
    if args.recalibrate:
        calibrate(force=True)
    if args.selftest:
        selftest()
    elif args.smoke:
        calibrate()
        smoke(port=SMOKE_PORT)
    else:
        serve(args.port)
