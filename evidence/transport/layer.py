"""Saryu: a generalized-Householder transport layer.

Every design choice below is fixed by a measurement from this project, not by taste.
Each is annotated with the experiment and the number that decided it.

    choice                          evidence
    ------------------------------  ---------------------------------------------------
    beta INITIALISED AT 2           beta_act.py: beta = 2*sigmoid(raw) with raw init 0
                                    gives beta = 1.000 (a PROJECTION) and it never moves:
                                    0/4 solved. Init at 2 (a reflection): 2/4-3/6.
    beta = 1 - cos(theta)           beta_activation.py 2026-09-01: across S_4 and A_4 at
                                    n_h 2 and 3, cos scores 9/16 vs free 9/16, frozen 5/16,
                                    hard-clamp 2/16 and sigmoid 0/16. It TIES free on solve
                                    rate but is bounded to [0,2] by construction, where free
                                    is unbounded and reached |h| = 434,718 at L=4096. It
                                    reaches beta = 2.000 exactly when the parity selection
                                    rule allows (A_4 n_h=2: grp-law 0.0028 vs frozen 0.0025)
                                    and slides to 1.336 when it does not (A_4 n_h=3: 4/4
                                    where frozen is 0/4). sin(theta) vanishes ONLY at pi, so
                                    the reflection is sticky, and the projection at theta =
                                    pi/2 sits at MAXIMUM gradient -- the known failure mode
                                    becomes the least stable point on the curve.
    beta LEARNABLE PER TOKEN        beta_act.py: fixed beta = 2 is 0/4, because n_h pure
                                    reflections force det(T) = (-1)^n_h for EVERY token,
                                    and a group with both parities cannot be represented.
    beta in (0, 2)                  beta_act.py: beta = 1*sigmoid caps below 1 -- only
                                    contractions, no reflections -- 0/3, and beta drifts
                                    DOWN to 0.33-0.60. Non-abelian structure needs eig -1.
    CORRELATED householder init     random unit vectors are near-orthogonal in high d, and
                                    orthogonal reflections COMMUTE: the commutator ratio
                                    falls 0.912 (d=4) -> 0.017 (d=256) as 1/sqrt(d). At
                                    fla's defaults the two householders commute to 0.4%,
                                    so the model starts almost abelian.
    LINEAR readout                  missing.py: linear 0.556 vs MLP 0.059 at 8x train
                                    length. An MLP decodes a degenerate state and hides
                                    the transport's failure; a linear head cannot.
    NO state normalisation          flipflop.py: permutations lie on an affine hyperplane,
                                    so the TRIVIAL summand supplies Krohn-Rhodes resets for
                                    free (linear 0.946). Projecting it out kills the reset
                                    (0.307) unless an affine write term is added (0.996).
                                    RMSNorm on the recurrent state would remove it.
    soft KERNEL aux loss            kernel_pen.py: 5/6 vs 3/6 baseline, and TABLE-FREE.
                                    For a homomorphism, faithful <=> only the identity maps
                                    to I, so |G|-1 constraints against a fixed reference
                                    beat |G|^2 mutual ones (the margin hinge was harmful).

UNVALIDATED: no real data, no GPU, no reproduction of DeltaProduct's published numbers.
This is a hypothesis with measured components, not a benchmarked architecture.

REFERENCES
  [Siems et al. 2025] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products, arXiv 2502.10297
  [Grazzi et al. 2024] Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues, arXiv 2411.12537
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def householder_init(n_tokens, n_h, dim, correlation=0.5, generator=None):
    """Householder directions with DELIBERATE correlation between the n_h steps.

    Independent draws are near-orthogonal in high dim, and orthogonal reflections commute,
    so a product of them is nearly abelian -- the exact structure the architecture exists to
    avoid. We draw the first direction freely and tilt each subsequent one toward it by
    `correlation` (the cosine), which fixes the commutator scale independently of dim.
    """
    v = torch.randn(n_tokens, n_h, dim, generator=generator)
    v = F.normalize(v, dim=-1)
    if n_h > 1 and correlation != 0.0:
        anchor = v[:, :1]                                   # (n_tokens, 1, dim)
        rest = v[:, 1:]
        perp = rest - (rest * anchor).sum(-1, keepdim=True) * anchor
        perp = F.normalize(perp, dim=-1)
        tilted = correlation * anchor + math.sqrt(max(0.0, 1 - correlation ** 2)) * perp
        v = torch.cat([anchor, tilted], dim=1)
    return v


class SaryuTransport(nn.Module):
    """h <- prod_i (I - beta_i u_i u_i^T) h, per-token learnable directions and betas.

    vocab=...        free transport per vocabulary token (the setting every measurement in
                     this project was made in).
    hidden_size=...  directions and betas projected from a hidden state, as DeltaNet-family
                     layers do.
    """

    def __init__(self, dim, n_h=3, vocab=None, hidden_size=None,
                 correlation=0.5, beta_init=2.0, beta_param='cos'):
        """beta_param:
            'free'    beta IS the parameter, initialised at beta_init. This is what every
                      working measurement used, and beta_init=2.0 (a pure reflection) is
                      reachable.
            'sigmoid' beta = 2*sigmoid(raw), fla's form. NOTE it approaches 2 only
                      ASYMPTOTICALLY -- a pure reflection is UNREACHABLE, and beta_init=2.0
                      is therefore rejected here. That is a plausible reason fla-style
                      initialisation sits at a projection: beta_act.py found every variant
                      starting at beta <= 1.0 scored 0/4 and never moved.
        """
        super().__init__()
        if (vocab is None) == (hidden_size is None):
            raise ValueError('give exactly one of vocab (free per-token) or '
                             'hidden_size (input-dependent)')
        if beta_param not in ('free', 'sigmoid', 'cos'):
            raise ValueError("beta_param must be 'free', 'sigmoid' or 'cos'")
        if beta_param == 'sigmoid' and not 0.0 < beta_init < 2.0:
            raise ValueError('with beta_param="sigmoid", beta = 2*sigmoid(raw) never '
                             'reaches 2, so beta_init must lie strictly in (0, 2). '
                             'Use beta_param="free" for a true reflection at init.')
        self.dim, self.n_h, self.beta_param = dim, n_h, beta_param
        self.input_dependent = hidden_size is not None
        if beta_param == 'sigmoid':
            raw0 = math.log(beta_init / (2.0 - beta_init))
        elif beta_param == 'cos':
            # theta with 1 - cos(theta) = beta_init; beta_init = 2 gives theta = pi
            raw0 = math.acos(max(-1.0, min(1.0, 1.0 - beta_init)))
        else:
            raw0 = beta_init
        if self.input_dependent:
            self.v_proj = nn.Linear(hidden_size, n_h * dim, bias=False)
            self.b_proj = nn.Linear(hidden_size, n_h, bias=True)
            nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
            nn.init.zeros_(self.b_proj.weight)
            nn.init.constant_(self.b_proj.bias, raw0)       # start AT beta_init, not at 1
        else:
            self.v = nn.Parameter(householder_init(vocab, n_h, dim, correlation))
            self.raw = nn.Parameter(torch.full((vocab, n_h), raw0))

    def directions_and_beta(self, tok_or_hidden):
        if self.input_dependent:
            h = tok_or_hidden
            v = self.v_proj(h).view(*h.shape[:-1], self.n_h, self.dim)
            raw = self.b_proj(h)
        else:
            v, raw = self.v[tok_or_hidden], self.raw[tok_or_hidden]
        if self.beta_param == 'sigmoid':
            beta = 2.0 * torch.sigmoid(raw)
        elif self.beta_param == 'cos':
            beta = 1.0 - torch.cos(raw)
        else:
            beta = raw
        return F.normalize(v, dim=-1), beta

    def step(self, state, tok_or_hidden):
        u, beta = self.directions_and_beta(tok_or_hidden)
        for i in range(self.n_h):
            ui = u[..., i, :]
            state = state - beta[..., i:i + 1] * (state * ui).sum(-1, keepdim=True) * ui
        return state

    def forward(self, tokens, state):
        """Roll the transport over a sequence; returns every intermediate state."""
        out = []
        for t in range(tokens.shape[1]):
            state = self.step(state, tokens[:, t])
            out.append(state)
        return torch.stack(out, 1)


class FlipFlop(nn.Module):
    """Krohn-Rhodes reset: h <- (1 - g) h + g c(x),  g = (1 - cos(phi)) / 2.

    An orthogonal transport is a BIJECTION -- it permutes information and never destroys
    it. That is exactly right for groups and wrong for everything else, because reality is
    mostly irreversible. Measured today:
        monoid task (S_4 + RESET)   pure transport 0.337 -> with flip-flop 1.000, and
                                    beta STAYED at 1.980, so exactness was not traded away
        Taxi-v3 (0/6 actions invertible)  0.086 -> 0.482 (5.6x)
    Krohn-Rhodes: every finite monoid decomposes into groups wreathed with flip-flops.
    The transport is the group half; this is the other half.

    g = (1 - cos(phi))/2 rather than a sigmoid so the gate is DISCRETE by construction:
    dg/dphi = sin(phi)/2 vanishes at both phi=0 (hold) and phi=pi (reset), and is maximal
    at the half-blend -- so it settles on hold-or-reset and is repelled from the mushy
    middle. A flip-flop is a reset automaton, not a soft average of remembering and
    forgetting. per_dim=True gives one angle per dimension (selective forgetting), which a
    GRU gets from its per-dimension forget gates and a scalar gate cannot express.
    """

    def __init__(self, dim, hidden_size, per_dim=True, gate='cos'):
        super().__init__()
        if gate not in ('cos', 'sigmoid'):
            raise ValueError("gate must be 'cos' or 'sigmoid'")
        self.dim, self.gate, self.per_dim = dim, gate, per_dim
        self.g_proj = nn.Linear(hidden_size, dim if per_dim else 1)
        self.c_proj = nn.Linear(hidden_size, dim)
        nn.init.constant_(self.g_proj.bias, 0.4 if gate == 'cos' else -1.0)

    def gate_value(self, h):
        r = self.g_proj(h)
        return torch.sigmoid(r) if self.gate == 'sigmoid' else (1.0 - torch.cos(r)) / 2.0

    def forward(self, state, h):
        g = self.gate_value(h)
        return (1.0 - g) * state + g * self.c_proj(h)


class SaryuLayer(nn.Module):
    """Transport + LINEAR readout, and no state normalisation.

    The missing normalisation is deliberate: permutation-like representations carry a
    trivial summand that supplies resets for free, and normalising the recurrent state
    removes exactly that component.
    """

    def __init__(self, dim, n_classes, n_h=3, vocab=None, hidden_size=None,
                 correlation=0.5, beta_init=2.0):
        super().__init__()
        self.transport = SaryuTransport(dim, n_h, vocab, hidden_size,
                                        correlation, beta_init)
        self.h0 = nn.Parameter(torch.randn(dim) * 0.3)
        self.readout = nn.Linear(dim, n_classes)            # LINEAR, deliberately
        self.dim, self.vocab = dim, vocab

    def states(self, tokens):
        s = self.h0.expand(tokens.shape[0], self.dim)
        return self.transport(tokens, s)

    def forward(self, tokens):
        return self.readout(self.states(tokens))

    def kernel_loss(self, n_probe=128, sigma=0.5):
        """Table-free auxiliary objective: penalise transports sitting near the identity.

        For a homomorphism, faithful <=> only the identity maps to I. Each token gets ONE
        constraint against a FIXED reference, rather than |G|^2 mutual separations -- the
        all-pairs versions (margin hinge, Hopfield) fought the algebra and were harmful.
        No group table, no character table, no labels.
        """
        if self.transport.input_dependent:
            raise RuntimeError('kernel_loss requires the free per-token transport')
        dev_h = self.h0.device
        h = torch.randn(n_probe, self.dim, device=dev_h)
        devs = []
        for g in range(self.vocab):
            tok = torch.full((n_probe,), g, device=dev_h, dtype=torch.long)
            devs.append((self.transport.step(h, tok) - h).pow(2).mean())
        return torch.exp(-torch.stack(devs) / sigma ** 2).sum()
