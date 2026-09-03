"""Tests for the FaBP solver: the linear algebra, the convergence guard, and
the guilt-by-association behaviour it exists to produce.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from orbweaver.scoring import fabp as F


def _toy_graph(seed=0, n=8):
    """A small, symmetric, positively-weighted graph - dense enough that a
    direct dense solve is trivial to compute for comparison."""
    rng = np.random.default_rng(seed)
    dense = rng.uniform(0.1, 1.0, size=(n, n))
    dense = (dense + dense.T) / 2
    np.fill_diagonal(dense, 0.0)
    # Sparsify a little so this looks like a real graph, not a clique.
    mask = rng.uniform(size=(n, n)) < 0.5
    mask = mask | mask.T
    dense = dense * mask
    A = sp.csr_matrix(dense)
    d_diag = np.asarray(A.sum(axis=1)).ravel()
    return A, d_diag, dense


def test_fabp_constants_match_theorem_1():
    h_h = 0.1
    a, c = F.fabp_constants(h_h)
    denom = 1 - 4 * h_h ** 2
    assert a == pytest.approx(4 * h_h ** 2 / denom)
    assert c == pytest.approx(2 * h_h / denom)


def test_fabp_constants_reject_h_h_outside_the_linearisations_domain():
    with pytest.raises(ValueError):
        F.fabp_constants(0.5)
    with pytest.raises(ValueError):
        F.fabp_constants(0.9)


def test_solver_matches_a_dense_direct_solve_on_a_small_graph():
    A, d_diag, dense = _toy_graph()
    n = d_diag.size
    bounds = F.convergence_bounds(d_diag)
    h_h = 0.5 * bounds["max"]  # safely inside the guaranteed region

    rng = np.random.default_rng(1)
    phi = rng.choice([-0.01, 0.0, 0.01], size=n)

    solved = F.solve_fabp(A, d_diag, phi, h_h)
    a, c = F.fabp_constants(h_h)
    M = np.eye(n) + a * np.diag(d_diag) - c * dense
    dense_solution = np.linalg.solve(M, phi)

    assert solved["beliefs"] == pytest.approx(dense_solution, abs=1e-4)


def test_a_violating_c_prime_raises():
    A, d_diag, _ = _toy_graph()
    # 0.49 is close to the "about-half" ceiling and, for any graph with real
    # degree, clears neither Lemma 5 nor Lemma 6.
    bounds = F.convergence_bounds(d_diag)
    assert 0.49 >= bounds["max"], "test assumption: 0.49 should violate both bounds"
    with pytest.raises(ValueError):
        F.assert_convergent(0.49, d_diag)
    with pytest.raises(ValueError):
        F.solve_fabp(A, d_diag, np.zeros(d_diag.size), 0.49)


def test_beliefs_are_deterministic():
    A, d_diag, _ = _toy_graph()
    bounds = F.convergence_bounds(d_diag)
    h_h = 0.5 * bounds["max"]
    rng = np.random.default_rng(2)
    phi = rng.choice([-0.01, 0.0, 0.01], size=d_diag.size)

    r1 = F.solve_fabp(A, d_diag, phi, h_h)
    r2 = F.solve_fabp(A, d_diag, phi, h_h)
    assert np.array_equal(r1["beliefs"], r2["beliefs"])
    assert r1["iterations"] == r2["iterations"]


def test_choose_h_h_is_capped_by_the_convergence_bound_when_it_binds():
    # A high-degree node forces both convergence bounds very small, while a
    # large lift asks for a much bigger h_h than either bound allows.
    d_diag = np.array([500.0, 2.0, 2.0, 1.0])
    choice = F.choose_h_h(d_diag, lift=10.0)
    bounds = F.convergence_bounds(d_diag)
    assert choice["desired_from_assortativity"] > bounds["max"]
    assert choice["capped_by_convergence"] is True
    assert choice["h_h"] < bounds["max"]  # strictly inside, not on the boundary
    assert choice["h_h"] == pytest.approx(bounds["max"], rel=1e-3)
    F.assert_convergent(choice["h_h"], d_diag)  # must not raise


def test_choose_h_h_is_not_capped_when_assortativity_is_already_small():
    d_diag = np.array([2.0, 2.0, 3.0, 1.0])
    choice = F.choose_h_h(d_diag, lift=1.01)
    expected_desired = 0.5 * (1.0 - 1.0 / 1.01)
    assert choice["capped_by_convergence"] is False
    assert choice["h_h"] == pytest.approx(expected_desired)


def test_held_out_accounts_contribute_no_prior():
    labels = np.array([1, 0, -1, 1, 0, -1], dtype=np.int8)
    visible = np.array([True, True, False, False, False, False])
    phi = F.build_prior(labels, visible)
    assert phi[0] == F.PRIOR_MAGNITUDE
    assert phi[1] == -F.PRIOR_MAGNITUDE
    assert (phi[2:] == 0.0).all()


def test_visible_mask_raises_if_a_held_out_account_is_visible():
    class FakeSplit:
        labels = np.array([1, 0, -1], dtype=np.int8)
        train = np.array([0])
        val = np.array([2])       # accidentally includes the "test" index
        test = np.array([2])

    with pytest.raises(RuntimeError):
        F.visible_mask(FakeSplit())


def test_measured_assortativity_ignores_edges_touching_an_invisible_account():
    labels = np.array([1, 1, 0, 0])
    src = np.array([0, 0, 2])
    dst = np.array([1, 3, 3])
    visible = np.array([True, True, True, False])  # account 3 held out
    out = F.measured_assortativity(labels, src, dst, visible)
    # Only edge (0,1), both fraud and both visible, counts.
    assert out["edges_visible"] == 1
    assert out["fraud_fraud_rate"] == 1.0


def test_a_planted_homophilous_toy_graph_is_solved_correctly():
    """Two clusters, wired so members of each cluster mostly connect to each
    other; one labelled account per cluster. FaBP should give the unlabelled
    members of the fraud cluster a higher belief than the unlabelled members
    of the normal cluster - guilt genuinely spreads along the edges that
    carry it, not just linear algebra that happens to be self-consistent.
    """
    rng = np.random.default_rng(3)
    k = 6  # accounts per cluster
    n = 2 * k
    fraud_cluster = np.arange(0, k)
    normal_cluster = np.arange(k, n)

    dense = np.zeros((n, n))
    for grp in (fraud_cluster, normal_cluster):
        for i in grp:
            for j in grp:
                if i < j:
                    dense[i, j] = dense[j, i] = 1.0
    # A little cross-cluster noise so the graph is not two disconnected
    # islands - homophily should still win.
    for _ in range(4):
        i, j = rng.integers(0, k), rng.integers(k, n)
        dense[i, j] = dense[j, i] = 0.2

    A = sp.csr_matrix(dense)
    d_diag = np.asarray(A.sum(axis=1)).ravel()
    bounds = F.convergence_bounds(d_diag)
    h_h = 0.9 * bounds["max"]

    labels = np.full(n, -1, dtype=np.int8)
    labels[0] = 1   # one confirmed fraud account in the fraud cluster
    labels[k] = 0   # one confirmed normal account in the normal cluster
    visible = labels != -1

    phi = F.build_prior(labels, visible, magnitude=0.05)
    solved = F.solve_fabp(A, d_diag, phi, h_h)
    beliefs = solved["beliefs"]

    unlabelled_fraud_side = beliefs[fraud_cluster[1:]]
    unlabelled_normal_side = beliefs[normal_cluster[1:]]
    assert unlabelled_fraud_side.min() > unlabelled_normal_side.max()
