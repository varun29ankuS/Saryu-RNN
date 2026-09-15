"""Model invariants. Run from the repo root: python -m pytest tests"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from saryu.model import (CHUNK, NH, NHH, SaryuV3Block, SaryuV3LM, chunkwise,  # noqa: E402
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
