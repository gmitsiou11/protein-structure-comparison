"""Tests for src/information_imbalance.py (numpy only: runs in the main and in the DII environment)."""

import numpy as np
import pandas as pd
import pytest

from src.information_imbalance import (
    imbalance_table,
    information_imbalance,
    load_metric_matrix,
    log_standardise,
    neighbour_ranks,
    protein_covariates,
    subsample,
)


def test_neighbour_ranks_exclude_self_and_break_ties():
    Z = np.array([[0.0], [1.0], [1.0], [5.0]])  # points 1 and 2 tie as neighbours of 0
    ranks = neighbour_ranks(Z, np.random.default_rng(0))
    assert (np.diag(ranks) == len(Z)).all()  # self is last
    for i in range(len(Z)):
        assert sorted(ranks[i]) == list(range(1, len(Z) + 1))  # a permutation
    assert {ranks[0, 1], ranks[0, 2]} == {1, 2}  # the tie gets ranks 1 and 2


def test_identical_spaces_carry_all_information():
    Z = np.random.default_rng(1).normal(size=(300, 2))
    assert information_imbalance(Z, Z) == pytest.approx(2 / 300)  # rank 1 every time


def test_independent_spaces_carry_no_information():
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(800, 2)), rng.normal(size=(800, 2))
    assert information_imbalance(A, B) == pytest.approx(1.0, abs=0.1)


def test_imbalance_is_asymmetric_for_a_subspace():
    rng = np.random.default_rng(3)
    full = rng.normal(size=(600, 3))
    part = full[:, :1]  # B is one coordinate of A
    to_part = information_imbalance(full, part)
    to_full = information_imbalance(part, full)
    assert to_part < 0.25  # A's neighbours are close in its own first coordinate
    assert to_full > 0.5  # one coordinate does not fix the other two
    assert to_part < to_full


def test_log_standardise_columns():
    X = np.array([[0.0, 1.0], [1.0, 10.0], [9.9999, 100.0]])
    Z = log_standardise(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-12)
    constant = log_standardise(np.ones((4, 1)))
    assert np.isfinite(constant).all()


def test_imbalance_table_shape_and_diagonal():
    Z = np.random.default_rng(4).normal(size=(200, 3))
    table = imbalance_table(Z, ["a", "b", "c"])
    assert list(table.index) == ["a", "b", "c", "all"]
    assert np.isnan(np.diag(table.values.astype(float))).all()
    assert table.loc["all", "a"] < table.loc["a", "all"]


def test_load_metric_matrix_filters_records():
    records = [
        {"status": "ok", "category": "exp_vs_af2", "protein_name": "P", "comparison": "NMR_vs_AF2",
         "w_rdist_norm": 0.1, "b_phipsi": 0.01, "t_alpha": 0.2, "rmsd": 1.0},
        {"status": "ok", "category": "nmr_intra_ensemble", "protein_name": "P", "comparison": "NMR_vs_NMR",
         "w_rdist_norm": 0.1, "b_phipsi": 0.01, "t_alpha": 0.0, "rmsd": 0.5},
        {"status": "ok", "category": "exp_vs_exp", "protein_name": "P", "comparison": "NMR_vs_Cryo-EM",
         "w_rdist_norm": 0.1, "b_phipsi": None, "t_alpha": 0.2, "rmsd": 1.0},
        {"status": "rejected", "category": "exp_vs_exp", "protein_name": "P", "comparison": "X-ray_vs_NMR"},
    ]
    X, info = load_metric_matrix(records)
    assert X.shape == (1, 4) and list(info["category"]) == ["exp_vs_af2"]
    X_all, _ = load_metric_matrix(records, include_nmr_ensemble=True)
    assert X_all.shape == (2, 4)


def test_subsample_is_reproducible():
    a, b = subsample(1000, 100, seed=7), subsample(1000, 100, seed=7)
    np.testing.assert_array_equal(a, b)
    assert len(np.unique(a)) == 100
    np.testing.assert_array_equal(subsample(50, 100, seed=0), np.arange(50))


def test_protein_covariates():
    table = pd.DataFrame(
        {
            "protein_name": ["P1", "P2"],
            "plddt_af2": [90.0, 70.0],
            "plddt_of3": [80.0, 60.0],
            "length": [100, 200],
            "cryoem_chains": [1, 10],
            "has_xray": ["True", "False"],
            **{f"{m}__exp": [0.1, 0.2] for m in ["w_rdist_norm", "b_phipsi", "t_alpha", "rmsd"]},
        }
    )
    structures = pd.DataFrame(
        {
            "protein_name": ["P1", "P1", "P2"],
            "release_date": ["2010-01-01", "2020-01-01", ""],
        }
    )
    cov = protein_covariates(table, structures).set_index("protein_name")
    assert cov.loc["P1", "has_xray"] == 1.0 and cov.loc["P2", "has_xray"] == 0.0
    assert cov.loc["P1", "frac_before_af2_cutoff"] == pytest.approx(0.5)
    assert np.isnan(cov.loc["P2", "frac_before_af2_cutoff"])  # no release date known
    assert cov.loc["P2", "exp_rmsd"] == pytest.approx(0.2)


def test_one_per_protein_picks_one_row_of_each_protein():
    from src.information_imbalance import one_per_protein

    info = pd.DataFrame({"protein_name": ["a", "a", "a", "b", "b", "c"], "category": ["x"] * 6, "comparison": ["y"] * 6})
    idx = one_per_protein(info, seed=0)
    assert len(idx) == 3
    assert sorted(info.loc[idx, "protein_name"]) == ["a", "b", "c"]
    assert list(idx) == sorted(idx)
    assert list(one_per_protein(info, seed=0)) == list(idx)  # reproducible
    assert any(list(one_per_protein(info, seed=s)) != list(idx) for s in range(1, 20))  # but seed-dependent
