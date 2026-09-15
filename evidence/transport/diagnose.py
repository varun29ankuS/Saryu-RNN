"""Diagnostics: read off WHICH algebraic structure a transport learned.

Accuracy alone conflates two different things -- whether the transport is a homomorphism
at all, and which quotient it landed on. These separate them, cheaply, with no
long-sequence evaluation.

    group_law_error(...)     is it a homomorphism?   (needs the Cayley table)
    character(...)           chi(g) = tr(T_g)        (needs NOTHING)
    character_norm(...)      <chi,chi> = sum of squared multiplicities  (needs nothing)
    irrep_multiplicities()   which frequencies are present  (needs the character table)
    coset_consistency(...)   which normal subgroup is the kernel  (needs the table)

The measured behaviour these rest on:

  * failure is QUANTISED. On S_4, models land on 1.000 / 0.250 / 0.083 / 0.042 and nothing
    between -- exactly 1/|N| over the normal subgroup lattice. Verified by coset consistency
    jumping to ~1.000 at one subgroup and sitting at chance below it, and by cross-group
    exclusivity (Q_8 produces a 0.500 plateau S_4's lattice forbids, and vice versa).

  * <chi,chi> catches what group-law error CANNOT. A seed with a respectable group law of
    0.0964 -- a good homomorphism onto the WRONG quotient -- reads 5.495 here, against 2.00
    for every solved seed. Group-law error is blind to the kernel; the character is not.

  * integer multiplicities mean it IS a representation; fractional ones mean it is not.
    Solved seeds read 0.999 / 1.002 on exactly two irreps; failures read 0.490, 0.662, 2.149.

CAVEAT: <chi,chi> is an excellent DIAGNOSTIC and a worthless OBJECTIVE. Minimising it
scores 2/6 against a 3/6 baseline, because the bound sum(m^2) >= 2 holds only ON the
representation manifold and the penalty pushes models off it -- one run ended at 1.993,
the perfect faithful value, with accuracy 0.056. Measure with it; do not train on it.
"""
from __future__ import annotations

import torch


def transport_matrices(layer, vocab):
    """T_g as explicit dim x dim matrices, by pushing the basis through the transport."""
    dev = layer.h0.device
    eye = torch.eye(layer.dim, device=dev)
    cols = []
    for j in range(layer.dim):
        row = eye[j].unsqueeze(0).expand(vocab, layer.dim)
        cols.append(layer.transport.step(row, torch.arange(vocab, device=dev)))
    return torch.stack(cols, dim=-1)                       # (vocab, dim, dim)


def character(layer, vocab):
    """chi(g) = tr(T_g). Needs no table of any kind, and is basis-independent -- which is
    why it is comparable across seeds whose weight matrices are not."""
    with torch.no_grad():
        M = transport_matrices(layer, vocab)
        return M.diagonal(dim1=-2, dim2=-1).sum(-1)


def character_norm(layer, vocab):
    """<chi,chi> = (1/|G|) sum_g chi(g)^2 = sum of squared irrep multiplicities.

    For a 4-dim representation of S_4 the exact values are:
        faithful (1+3 splits)  2.000 | ker V_4  3.000 or 5.000 | ker A_4  8.000 | trivial 16
    Monotone in the kernel, so lower means closer to faithful -- ON the representation
    manifold. Off it the bound does not hold.
    """
    return float((character(layer, vocab) ** 2).mean())


def group_law_error(layer, table, n_probe=300, reps=10):
    """max over probes of ||T_b(T_a(h)) - T_ab(h)||, normalised. 0 => a homomorphism.

    Blind to the kernel by construction: every quotient satisfies the group law exactly.
    """
    vocab = table.shape[0]
    dev = layer.h0.device
    worst = 0.0
    with torch.no_grad():
        for _ in range(reps):
            h = torch.randn(n_probe, layer.dim, device=dev)
            a = torch.randint(0, vocab, (n_probe,), device=dev)
            b = torch.randint(0, vocab, (n_probe,), device=dev)
            lhs = layer.transport.step(layer.transport.step(h, a), b)
            rhs = layer.transport.step(h, table[a, b])
            worst = max(worst, float((lhs - rhs).norm(dim=1).mean()) / layer.dim ** 0.5)
    return worst


def irrep_multiplicities(layer, vocab, char_table, class_of):
    """multiplicity(rho) = (1/|G|) sum_g chi(g) chi_rho(g).

    char_table: {name: tensor of chi_rho per CONJUGACY CLASS}
    class_of:   LongTensor mapping each group element to its conjugacy class index
    Integer values => a genuine representation. Which names are non-zero => which quotient.
    """
    chi = character(layer, vocab)
    out = {}
    for name, row in char_table.items():
        out[name] = float((chi * row[class_of]).sum() / vocab)
    return out


def coset_labels(table, subgroup):
    """Left cosets gH as integer labels, one per group element."""
    vocab = table.shape[0]
    lab = [-1] * vocab
    c = 0
    for g in range(vocab):
        if lab[g] >= 0:
            continue
        for h in subgroup:
            lab[int(table[g, h])] = c
        c += 1
    return torch.tensor(lab), c


def coset_consistency(pred, true, table, subgroups):
    """For each candidate kernel, the fraction of predictions landing in the right coset.

    Read the result as: the learned kernel is the SMALLEST subgroup scoring ~1.000, and the
    accuracy should then be ~1/|N|. Compare each score against its own chance level 1/n_cosets
    -- a coarser partition has a higher chance level, so the raw numbers are not comparable
    across rows.
    """
    out = {}
    for name, H in subgroups.items():
        lab, n_cos = coset_labels(table, H)
        lab = lab.to(pred.device)
        out[name] = {'score': float((lab[pred] == lab[true]).float().mean()),
                     'chance': 1.0 / n_cos,
                     'ceiling': 1.0 / len(H)}
    return out


def report(layer, vocab, table=None, char_table=None, class_of=None):
    """Everything computable from what you pass in. Table-free parts always run."""
    r = {'character_norm': character_norm(layer, vocab)}
    with torch.no_grad():
        beta = 2.0 * torch.sigmoid(layer.transport.raw) if not \
            layer.transport.input_dependent else None
    if beta is not None:
        r['beta_mean'] = float(beta.mean())
        r['beta_frac_reflective'] = float((beta > 1.5).float().mean())
    if table is not None:
        r['group_law_error'] = group_law_error(layer, table)
    if char_table is not None and class_of is not None:
        r['multiplicities'] = irrep_multiplicities(layer, vocab, char_table, class_of)
    return r
