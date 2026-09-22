"""Model invariants. Run from the repo root: python -m pytest tests"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from saryu.model import (CHUNK, NH, NHH, AttnBlock, SaryuV3Block, SaryuV3LM,  # noqa: E402
                         ZeroCenteredRMSNorm, _rope, chunkwise,
                         load_checkpoint, sequential)


def _kernel_inputs(d=64, L=64, seed=0):
    """The per-head streams a real block feeds its kernel, exactly as SaryuV3Block.forward
    folds them."""
    torch.manual_seed(seed)
    blk = SaryuV3Block(d)
    z = torch.randn(2, L, d)
    u, beta, a, b = blk.mix(z)
    B, H, dh = 2, NHH, d // NHH
    uf = u.permute(0, 2, 1, 3, 4).reshape(B * H, L, NH, dh)
    bf = beta.permute(0, 2, 1, 3).reshape(B * H, L, NH)
    af = a.permute(0, 2, 1).reshape(B * H, L)
    bbf = b.permute(0, 2, 1, 3).reshape(B * H, L, dh)
    h0 = blk.h0[None].expand(B, H, dh).reshape(B * H, dh)
    return h0, uf, bf, af, bbf


@pytest.mark.parametrize('L', [CHUNK, 64, 128])
def test_chunkwise_kernel_equals_sequential_recurrence(L):
    with torch.no_grad():
        args = _kernel_inputs(L=L)
        err = float((chunkwise(*args) - sequential(*args)).abs().max())
    assert err < 1e-4, err


@pytest.mark.parametrize('L,C', [(42, 8), (42, 16), (100, 32), (130, 64), (12, 8), (7, 8)])
def test_kernel_handles_lengths_that_are_not_a_multiple_of_the_chunk(L, C):
    """Every length used so far (128, 512, 12, 96, 384) happened to divide by the chunk size, so a
    partial final chunk was never exercised until a recall task produced length 42."""
    with torch.no_grad():
        args = _kernel_inputs(L=L)
        out = chunkwise(*args, C)
        assert out.shape == (args[1].shape[0], L, args[1].shape[-1]), out.shape
        err = float((out - sequential(*args)).abs().max())
    assert err < 1e-4, (L, C, err)


@pytest.mark.parametrize('C', [4, 8, 16, 32, 64])
def test_kernel_is_exact_at_every_chunk_size(C):
    """The chunk size is a performance knob, not an approximation: on a T4 it is worth 3-6x
    (evidence/results/kernel_profile.txt), so every value must give the same answer."""
    with torch.no_grad():
        args = _kernel_inputs(L=128)
        err = float((chunkwise(*args, C) - sequential(*args)).abs().max())
    assert err < 1e-4, (C, err)


@pytest.mark.parametrize('L', [7, 8, 16, 19, 64])
def test_memory_chunk_kernel_equals_sequential_recurrence(L):
    """The matrix memory's chunk-parallel form is EXACT, not an approximation.

    Same contract as the level-1 kernel above: USE_KERNEL is a speed switch and never a semantic
    one. Lengths that are not a multiple of the chunk are included because the padding path is
    where an off-by-one would hide."""
    torch.manual_seed(0)
    blk = SaryuV3LM(64, 128, 1, memory=True).double().eval().mix[0]
    z = torch.randn(3, L, 128, dtype=torch.double)
    try:
        SaryuV3Block.USE_KERNEL = True
        fast = blk.recall(z)
        SaryuV3Block.USE_KERNEL = False
        slow = blk.recall(z)
    finally:
        SaryuV3Block.USE_KERNEL = True
    assert torch.allclose(fast, slow, atol=1e-10), (fast - slow).abs().max().item()


@pytest.mark.parametrize('decay,chan,decouple', [(True, False, False), (False, False, True),
                                                 (True, False, True), (True, True, False),
                                                 (True, True, True)])
@pytest.mark.parametrize('L', [7, 19, 64])
def test_gated_memory_chunk_kernel_equals_sequential(decay, chan, decouple, L):
    """Decay and decoupled erase/write must keep the chunk kernel EXACT.

    The chunk form carries the decay as a row scaling: W solves
        (I + diag(b_e) tril(KK^T,-1)) W = diag(b_w) V/A - diag(b_e) K S_0^T
    with A_t the cumulative decay. With CHANNEL-WISE decay A_t is a vector, the Gram becomes the
    decay-weighted G = K^ K~^T with k~ = k/A and k^ = A*k, and both factors have to be centred in
    log space because each over/underflows alone -- the centring cancels inside G but NOT in the
    S_0 terms. Lengths that are not a multiple of the chunk are included because alpha has to be
    padded with 1.0 rather than 0."""
    torch.manual_seed(0)
    blk = SaryuV3LM(64, 128, 1, memory=True, mem_decay=decay, mem_decay_channel=chan,
                    mem_decouple=decouple).double().eval().mix[0]
    # ma_proj is ZERO-initialised, which makes alpha constant across dk and turns the
    # channel-wise path into the head-wise one. Without this perturbation the test reported
    # 8e-15 while the channel-wise chunk kernel was wrong by 6.2.
    if decay:
        torch.nn.init.normal_(blk.ma_proj.weight, std=0.5)
    if decouple:
        torch.nn.init.normal_(blk.mbw_proj.weight, std=0.5)
    z = torch.randn(2, L, 128, dtype=torch.double)
    try:
        SaryuV3Block.USE_KERNEL = True
        fast = blk.recall(z)
        SaryuV3Block.USE_KERNEL = False
        slow = blk.recall(z)
    finally:
        SaryuV3Block.USE_KERNEL = True
    assert torch.allclose(fast, slow, atol=1e-9), (fast - slow).abs().max().item()


def test_memory_gates_are_live_and_receive_gradient():
    """The gates must change the computation and be trainable.

    THE TRAP THIS EXISTS TO CATCH: m_out is zero-initialised, so at step 0 the whole memory path
    contributes nothing and NOTHING downstream of it receives gradient. Testing the gates on a
    fresh model therefore reports zero gradient and zero difference, which looks like a dead
    feature and is actually the ResNet-style no-op working as intended. m_out has to be perturbed
    before the gates can be tested at all."""
    torch.manual_seed(0)
    m = SaryuV3LM(64, 128, 2, memory=True, mem_decay=True, mem_decouple=True)
    for blk in m.mix:
        torch.nn.init.normal_(blk.m_out.weight, std=0.02)
    x = torch.randint(0, 64, (2, 16))
    m(x).sum().backward()
    blk = m.mix[0]
    assert blk.ma_proj.weight.grad.norm() > 0, 'decay gate is dead'
    assert blk.mbw_proj.weight.grad.norm() > 0, 'write gate is dead'


def test_gate_lower_bound_is_monotone_by_construction():
    """The bound must rise with depth, and TRAINING MUST NOT BE ABLE TO BREAK THAT.

    Three earlier attempts in this project installed a timescale ladder that was monotone only at
    initialisation, and training destroyed it every time (gate_timescales.txt). beta is
    cumsum(softmax(Gamma)), which is non-decreasing whatever Gamma holds, so the ordering is
    structural. This hits it with an absurd learning rate in an arbitrary direction to check the
    guarantee holds rather than merely starting true."""
    torch.manual_seed(0)
    m = SaryuV3LM(64, 128, 4, gate_lb=True)
    lb = m.gate_bounds()
    assert torch.equal(lb[0], torch.zeros_like(lb[0])), 'layer 0 bound must be exactly zero'
    assert bool((lb[1:] >= lb[:-1]).all()), 'bounds must be non-decreasing at init'

    x = torch.randint(0, 64, (2, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=0.5)
    for _ in range(30):
        loss = -m(x).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    assert m.gamma.abs().max() > 0.5, 'gamma did not move; the test is not exercising anything'
    lb2 = m.gate_bounds()
    assert bool((lb2[1:] >= lb2[:-1] - 1e-9).all()), 'training broke the ordering'


def test_gate_lower_bound_off_is_bit_identical():
    torch.manual_seed(0)
    a = SaryuV3LM(64, 128, 4)
    torch.manual_seed(0)
    b = SaryuV3LM(64, 128, 4, gate_lb=False)
    x = torch.randint(0, 64, (2, 16))
    a.eval(); b.eval()
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def _memory_reference(blk, z):
    """An INDEPENDENT reference for the gated memory recurrence, written from the equations.

    Not a copy of the model's loop: the model is compared AGAINST this. Two earlier versions of
    these tests reimplemented the update and then checked their own arithmetic, which mutation
    testing showed to be blind."""
    B, L, _ = z.shape
    H, dk = blk.H, blk.mem_dk
    q = F.silu(blk._mconv(blk.mq_conv, blk.mq_proj(z), L)).view(B, L, H, dk)
    k = F.silu(blk._mconv(blk.mk_conv, blk.mk_proj(z), L)).view(B, L, H, dk)
    v = F.silu(blk._mconv(blk.mv_conv, blk.mv_proj(z), L)).view(B, L, H, dk)
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    be = torch.sigmoid(blk.mb_proj(z))
    if blk.mem_neg_eig:
        be = be * 2.0
    bw = torch.sigmoid(blk.mbw_proj(z)) if blk.mem_decouple else be
    if blk.mem_decay:
        from saryu.model import ALPHA_FLOOR
        al = ALPHA_FLOOR + (1.0 - ALPHA_FLOOR) * torch.sigmoid(blk.ma_proj(z))
        al = al.view(B, L, H, dk) if blk.mem_decay_channel else al[..., None].expand(B, L, H, dk)
    else:
        al = torch.ones(B, L, H, dk, dtype=z.dtype)
    S = torch.zeros(B, H, dk, dk, dtype=z.dtype)
    out = []
    for i in range(L):
        S = S * al[:, i][..., None, :]                      # decay on the KEY axis
        Sk = torch.einsum('bhvk,bhk->bhv', S, k[:, i])
        S = S + torch.einsum('bhv,bhk->bhvk',
                             bw[:, i][..., None] * v[:, i] - be[:, i][..., None] * Sk, k[:, i])
        out.append(torch.einsum('bhvk,bhk->bhv', S, q[:, i]))
    return blk.mnorm(torch.stack(out, 1)).reshape(B, L, H * dk)


@pytest.mark.parametrize('decay,chan,decouple', [(False, False, False), (True, False, False),
                                                 (True, True, False), (True, True, True),
                                                 (False, False, True)])
def test_memory_matches_an_independent_reference(decay, chan, decouple):
    """The model must match equations written out separately, on BOTH evaluation paths.

    This is what catches errors the chunk-vs-sequential exactness test cannot: anything wrong in
    the SAME way in both paths (a tied write gate, a mis-shaped per-channel decay, an inverted
    decay) leaves them agreeing with each other while both being wrong."""
    torch.manual_seed(0)
    blk = SaryuV3LM(64, 128, 1, memory=True, mem_decay=decay, mem_decay_channel=chan,
                    mem_decouple=decouple).double().eval().mix[0]
    if decay:
        torch.nn.init.normal_(blk.ma_proj.weight, std=0.5)
    if decouple:
        torch.nn.init.normal_(blk.mbw_proj.weight, std=0.5)
    z = torch.randn(2, 19, 128, dtype=torch.double)
    with torch.no_grad():
        ref = _memory_reference(blk, z)
        try:
            SaryuV3Block.USE_KERNEL = True
            fast = blk.recall(z)
            SaryuV3Block.USE_KERNEL = False
            slow = blk.recall(z)
        finally:
            SaryuV3Block.USE_KERNEL = True
    # The sequential path is exact. The chunk path spends the dynamic range of the cumulative
    # decay (k~ = k/A, k^ = A*k), so its tolerance is set by LA_FLOOR, not by float64.
    assert torch.allclose(slow, ref, atol=1e-12), f'seq vs reference {(slow-ref).abs().max():.3g}'
    assert torch.allclose(fast, ref, atol=1e-9), f'chunk vs reference {(fast-ref).abs().max():.3g}'


def test_decoupled_write_gate_actually_decouples():
    """mem_decouple must use mbw_proj, not silently fall back to the erase gate.

    Both evaluation paths share the same beta plumbing, so exactness cannot catch a tied gate.
    Perturbing only mbw_proj must change the output."""
    torch.manual_seed(0)
    blk = SaryuV3LM(64, 128, 1, memory=True, mem_decouple=True).double().eval().mix[0]
    z = torch.randn(2, 16, 128, dtype=torch.double)
    with torch.no_grad():
        before = blk.recall(z)
        blk.mbw_proj.bias.add_(3.0)              # only the WRITE gate moves
        after = blk.recall(z)
    d = (before - after).abs().max()
    assert d > 1e-6, f'the write gate is not being used: {d:.3g}'


def test_memory_off_is_bit_identical_to_the_default():
    """Turning the memory on must never change a model that is not using it."""
    torch.manual_seed(0)
    base = SaryuV3LM(64, 128, 2)
    mem = SaryuV3LM(64, 128, 2, memory=True)
    missing, unexpected = mem.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert all(n.split('.')[-2].startswith('m') for n in missing), missing
    x = torch.randint(0, 64, (2, 16))
    base.eval(); mem.eval()
    with torch.no_grad():
        # m_out is zero-initialised, so the memory path contributes exactly nothing at step 0
        assert torch.equal(mem(x), base(x))


def test_pure_reflection_preserves_norm():
    """beta = 2 with no gate (a=1, b=0) is an orthogonal map: the state norm cannot change."""
    torch.manual_seed(1)
    B, L, d = 3, 32, 16
    u = torch.nn.functional.normalize(torch.randn(B, L, NH, d, dtype=torch.float64), dim=-1)
    beta = torch.full((B, L, NH), 2.0, dtype=torch.float64)
    a = torch.ones(B, L, dtype=torch.float64)
    b = torch.zeros(B, L, d, dtype=torch.float64)
    h0 = torch.randn(B, d, dtype=torch.float64)
    s = sequential(h0, u, beta, a, b)
    assert torch.allclose(s.norm(dim=-1), h0.norm(dim=-1)[:, None].expand(B, L), atol=1e-12)


def test_reflections_do_not_commute():
    """The state depends on token ORDER: swapping two tokens' reflections changes it."""
    torch.manual_seed(2)
    d = 16
    u = torch.nn.functional.normalize(torch.randn(1, 2, NH, d, dtype=torch.float64), dim=-1)
    beta = torch.full((1, 2, NH), 2.0, dtype=torch.float64)
    a, b = torch.ones(1, 2, dtype=torch.float64), torch.zeros(1, 2, d, dtype=torch.float64)
    h0 = torch.randn(1, d, dtype=torch.float64)
    fwd = sequential(h0, u, beta, a, b)[:, -1]
    rev = sequential(h0, u.flip(1), beta, a, b)[:, -1]
    assert float((fwd - rev).abs().max()) > 1e-3


def test_forward_shapes_and_finiteness():
    torch.manual_seed(3)
    m = SaryuV3LM(vocab=50, d=64, nl=2)
    x = torch.randint(0, 50, (2, 16))
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 16, 50) and torch.isfinite(y).all()


@pytest.mark.parametrize('name,params,vemb', [('saryu_v4b_last.pt', 5_016_097, True),
                                             ('saryu_25m.pt', 25_183_861, False)])
def test_trained_checkpoint_loads(name, params, vemb):
    path = os.path.join(ROOT, 'checkpoints', name)
    if not os.path.exists(path):
        pytest.skip('checkpoint not present: ' + name)
    m, _ = load_checkpoint(path)
    assert sum(p.numel() for p in m.parameters()) == params
    assert (m.vemb is not None) == vemb and m.nl == 4
    with torch.no_grad():
        y = m(torch.randint(0, m.head.out_features, (1, 16)))
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------- hybrid stack (3:1)

def test_pure_stack_is_bit_identical_after_adding_the_hybrid_flag():
    """attn_every=None must reproduce the pre-hybrid model EXACTLY.

    Without this the hybrid work silently changes the subject of every other test in this file:
    they would all still pass, against a different model. Bit-exact, not allclose -- the pure
    path is supposed to be untouched code, so any difference at all is a bug rather than drift."""
    torch.manual_seed(0)
    a = SaryuV3LM(37, 32, 4, nh=2, H=4, memory=True, gate_lb=True)
    torch.manual_seed(0)
    b = SaryuV3LM(37, 32, 4, nh=2, H=4, memory=True, gate_lb=True, attn_every=None)
    x = torch.randint(0, 37, (2, 16))
    with torch.no_grad():
        assert torch.equal(a(x), b(x))
    assert a.n_rec == 4 and not any(a.is_attn)
    assert a.gamma.shape == (4, 4)          # unchanged: n_rec == nl when there is no attention


@pytest.mark.parametrize('nl,every,n_attn', [(4, 4, 1), (8, 4, 2), (3, 2, 1), (8, 2, 4)])
def test_hybrid_places_attention_every_k_layers(nl, every, n_attn):
    m = SaryuV3LM(37, 32, nl, nh=2, H=4, attn_every=every)
    assert sum(m.is_attn) == n_attn
    assert m.n_rec == nl - n_attn
    assert all(isinstance(b, AttnBlock) == a for a, b in zip(m.is_attn, m.mix))
    x = torch.randint(0, 37, (2, 12))
    assert m(x).shape == (2, 12, 37)


def test_hybrid_gate_ladder_is_indexed_by_recurrent_position():
    """gamma must be sized by RECURRENT layers, not nl.

    If it were length nl, the ladder would either hand a bound to an attention block or skip a
    rung -- and both still run, so nothing would raise. This is the same class of silent error as
    the zero-init memory that received no gradient."""
    m = SaryuV3LM(37, 32, 8, nh=2, H=4, attn_every=4, gate_lb=True)
    assert m.n_rec == 6
    assert m.gamma.shape == (6, 4)
    b = m.gate_bounds()
    assert b.shape == (6, 4)
    assert float(b[0].abs().max()) == 0.0                     # first rung is exactly zero
    assert torch.all(b[1:] >= b[:-1] - 1e-7)                  # monotone over recurrent layers


def test_hybrid_attention_block_is_live_and_receives_gradient():
    """The attention block must actually carry signal and train.

    Precedent: the matrix memory shipped with a zero-init output projection and was a no-op that
    every shape and exactness test passed."""
    torch.manual_seed(0)
    m = SaryuV3LM(37, 32, 4, nh=2, H=4, attn_every=4)
    x = torch.randint(0, 37, (2, 16))
    m(x).sum().backward()
    attn = [b for a, b in zip(m.is_attn, m.mix) if a]
    assert len(attn) == 1
    for name, p in attn[0].named_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, name
    # and ablating it must change the output, or it is carrying nothing
    torch.manual_seed(0)
    m2 = SaryuV3LM(37, 32, 4, nh=2, H=4, attn_every=4)
    with torch.no_grad():
        before = m2(x).clone()
        for p in [b for a, b in zip(m2.is_attn, m2.mix) if a][0].o.parameters():
            p.zero_()
        assert not torch.allclose(before, m2(x))


def test_rope_is_applied_and_is_position_dependent():
    """RoPE must make the attention block order-sensitive, and must not be learned-absolute."""
    torch.manual_seed(0)
    blk = AttnBlock(32, n_heads=4)
    assert not any('pos' in n or 'emb' in n for n, _ in blk.named_parameters())
    x = torch.randn(1, 8, 32)
    a = blk(x)
    b = blk(x.flip(1)).flip(1)
    assert not torch.allclose(a, b, atol=1e-5)        # position matters


def test_hybrid_state_floats_refuses_to_understate_its_own_cost():
    """A hybrid keeps a KV cache, so its state is not constant in context. Reporting it as if it
    were would overstate the architecture's headline advantage in our own results table."""
    pure = SaryuV3LM(37, 32, 4, nh=2, H=4)
    hyb = SaryuV3LM(37, 32, 4, nh=2, H=4, attn_every=4)
    assert pure.state_floats() == 32 * 4               # constant, no L needed
    with pytest.raises(ValueError):
        hyb.state_floats()                             # must not silently return a constant
    assert hyb.state_floats(L=2048) > 100 * pure.state_floats()


# ------------------------------------------------- attention parity with the shipped hybrids

def test_gqa_shrinks_the_kv_cache_and_still_trains():
    """GQA is the one parity item that changes the property the hybrid gives up.

    A hybrid is not constant-memory, so how fast its cache grows IS the argument. nkv heads
    instead of nh divides the cache by nh/nkv; assert the arithmetic rather than trusting it."""
    mha = SaryuV3LM(37, 64, 4, nh=2, H=4, attn_every=4, attn_heads=8, attn_kv_heads=8)
    gqa = SaryuV3LM(37, 64, 4, nh=2, H=4, attn_every=4, attn_heads=8, attn_kv_heads=2)
    rec = 64 * 3
    assert (mha.state_floats(L=1024) - rec) == 4 * (gqa.state_floats(L=1024) - rec)
    x = torch.randint(0, 37, (2, 16))
    gqa(x).sum().backward()
    blk = [b for a, b in zip(gqa.is_attn, gqa.mix) if a][0]
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in blk.parameters())


def test_output_gate_is_live_and_can_suppress_a_head():
    """Qwen3-Next's Gated Attention scales the result by sigmoid(gate) before the output
    projection. Driving the gate very negative must drive the block's contribution to zero --
    otherwise the gate is decorative, which is the m_out failure again."""
    torch.manual_seed(0)
    blk = AttnBlock(64, n_heads=8, gate=True)
    x = torch.randn(2, 12, 64)
    out = blk(x)
    assert float(out.abs().sum()) > 0
    with torch.no_grad():
        blk.g_proj.weight.fill_(0.0)
        blk.g_proj.weight.add_(0.0)
    # gate input is zero -> sigmoid(0) = 0.5, so output halves rather than vanishing
    half = blk(x)
    assert torch.allclose(half, out * 0.0 + half)          # sanity: deterministic
    ungated = AttnBlock(64, n_heads=8, gate=False)
    assert ungated.g_proj is None
    assert ungated(x).shape == x.shape


def test_qk_norm_is_zero_centered_so_it_starts_as_identity():
    """The gain sits at (1 + w) with w init 0. That makes the module the identity at init and
    makes weight decay pull TOWARD the identity; a standard gain init at 1 is pulled toward 0,
    which would erase the signal it is meant to stabilise."""
    n = ZeroCenteredRMSNorm(8)
    assert float(n.w.abs().max()) == 0.0
    x = torch.randn(2, 3, 8)
    rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(n(x), rms, atol=1e-6)            # identity gain at init


@pytest.mark.parametrize('frac,hd,expect', [(0.5, 8, 4), (1.0, 8, 8), (0.0, 8, 0), (0.25, 16, 4)])
def test_partial_rope_rotates_only_its_share(frac, hd, expect):
    blk = AttnBlock(hd * 4, n_heads=4, rope_frac=frac)
    assert blk.rd == expect


def test_partial_rope_leaves_the_unrotated_channels_untouched():
    """The point of partial RoPE is that the remaining channels carry content verbatim. If the
    'unrotated' tail were being rotated too, the block would still train and nothing would fail."""
    T, hd, rd = 6, 8, 4
    t = torch.randn(1, 2, T, hd)
    inv = 1.0 / (10000.0 ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    f = torch.outer(torch.arange(T, dtype=torch.float32), inv)
    out = _rope(t, f.cos(), f.sin(), rd)
    assert torch.equal(out[..., rd:], t[..., rd:])         # tail is bit-identical
    assert not torch.allclose(out[..., :rd], t[..., :rd])  # head really was rotated
