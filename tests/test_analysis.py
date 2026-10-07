"""
Tests for the protein-level analysis in src/analysis.py.

The records are written by hand (no structures needed): each protein gets the
comparison records the pipeline would produce, with chosen metric values, so
every per-protein number can be checked exactly.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    _paired_dz,
    analyse,
    bootstrap_ci,
    experimental_structures,
    protein_table,
    summarise_proteins,
)

METRIC_KEYS = ("w_rdist_norm", "b_phipsi", "t_alpha", "rmsd")


def _record(protein, comparison, category, value, **extra):
    rec = {
        "protein_name": protein,
        "uniprot_id": f"ID_{protein}",
        "comparison": comparison,
        "category": category,
        "status": "ok",
    }
    rec.update({m: value for m in METRIC_KEYS})
    rec.update(extra)
    return rec


def _protein(
    name, exp, af2, of3, nmr_ens=(0.1, 0.2, 0.3), identity=1.0, extra=0, title=""
):
    """
    Records of one protein with NMR + Cryo-EM (+ X-ray when exp has 3 values).
    exp: experimental pair values; af2/of3: prediction-vs-experimental values,
    one per experimental structure (NMR first, then Cryo-EM, then X-ray).
    """
    methods = ["NMR", "Cryo-EM", "X-ray"][: len(af2)]
    exp_pairs = (
        ["NMR_vs_Cryo-EM"]
        if len(exp) == 1
        else ["X-ray_vs_NMR", "X-ray_vs_Cryo-EM", "NMR_vs_Cryo-EM"]
    )
    recs = [_record(name, c, "exp_vs_exp", v) for c, v in zip(exp_pairs, exp)]
    for method, value in zip(methods, af2):
        recs.append(
            _record(
                name,
                f"{method}_vs_AF2",
                "exp_vs_af2",
                value,
                n_residues_a=100 + (extra if method == "NMR" else 0),
                n_residues_b=100,
                n_matched=100,
                seq_identity=identity if method == "Cryo-EM" else 1.0,
                title_a=title if method == "Cryo-EM" else "",
                release_date_a="2016-01-01" if method == "NMR" else "2023-01-01",
                deposition_date_a="2015-06-01" if method == "NMR" else "2022-06-01",
                n_polymer_chains_a=12 if method == "Cryo-EM" else 1,
                plddt_mean_b=90.0,
            )
        )
    for method, value in zip(methods, of3):
        recs.append(
            _record(
                name,
                f"{method}_vs_OF3",
                "exp_vs_of3",
                value,
                plddt_mean_b=70.0,
                msa_depth_b=202,
            )
        )
    recs.append(_record(name, "AF2_vs_OF3", "af2_vs_of3", 0.05))
    for k, v in enumerate(nmr_ens, start=1):
        recs.append(
            _record(
                name,
                f"nmr_intra_model0_vs_model{k}",
                "nmr_intra_ensemble",
                v,
                is_intra_ensemble=True,
            )
        )
    return recs


@pytest.fixture
def records():
    return (
        # P1: AF2 closer than the experiments are to each other; OF3 farther
        _protein("P1", exp=[1.0], af2=[0.4, 0.6], of3=[2.0, 3.0])
        # P2: three experimental structures; AF2 and OF3 both farther
        + _protein("P2", exp=[0.5, 0.7, 0.6], af2=[1.0, 2.0, 3.0], of3=[1.0, 1.0, 1.0])
        # P3: flagged (cryo-EM is an engineered, fibril variant; NMR has a tag)
        + _protein(
            "P3",
            exp=[1.0],
            af2=[0.2, 0.2],
            of3=[0.2, 0.2],
            identity=0.78,
            extra=19,
            title="Cryo-EM structure of an amyloid FIBRIL",
        )
    )


def test_protein_table_medians(records):
    t = protein_table(records).set_index("protein_name")
    assert t.loc["P1", "w_rdist_norm__exp"] == 1.0
    assert t.loc["P1", "w_rdist_norm__af2"] == pytest.approx(0.5)  # median of 2
    assert t.loc["P2", "w_rdist_norm__exp"] == pytest.approx(0.6)  # median of 3
    assert t.loc["P2", "w_rdist_norm__af2"] == pytest.approx(2.0)
    assert t.loc["P1", "w_rdist_norm__af2_of3"] == pytest.approx(0.05)
    assert t.loc["P1", "w_rdist_norm__nmr_ens"] == pytest.approx(0.2)
    assert t.loc["P1", "w_rdist_norm__nmr_ens_max"] == pytest.approx(0.3)
    assert t.loc["P1", "w_rdist_norm__nmr_af2"] == pytest.approx(0.4)  # NMR vs AF2
    assert bool(t.loc["P2", "has_xray"]) and not bool(t.loc["P1", "has_xray"])
    assert t.loc["P2", "n_experimental"] == 3
    assert t.loc["P1", "plddt_af2"] == 90.0 and t.loc["P1", "plddt_of3"] == 70.0
    assert t.loc["P1", "msa_depth_of3"] == 202
    assert t.loc["P1", "cryoem_chains"] == 12


def test_flags(records):
    s = experimental_structures(records).set_index(["protein_name", "method"])
    assert s.loc[("P3", "Cryo-EM"), "flag_identity"]
    assert s.loc[("P3", "Cryo-EM"), "flag_state"]  # "FIBRIL", any case
    assert s.loc[("P3", "NMR"), "flag_extra"]  # 19 residues not matching the UniProt sequence
    assert not s.loc[("P1", "NMR"), "flagged"]
    # training cutoffs: NMR entry released 2016 (AF2) and deposited 2015 (OF3)
    assert s.loc[("P1", "NMR"), "before_af2_cutoff"]
    assert s.loc[("P1", "NMR"), "before_of3_cutoff"]
    assert not s.loc[("P1", "Cryo-EM"), "before_af2_cutoff"]
    t = protein_table(records).set_index("protein_name")
    assert t["flagged"].to_dict() == {"P1": False, "P2": False, "P3": True}


def test_summary_fractions_and_clean_subset(records):
    summary = summarise_proteins(protein_table(records), n_boot=500)
    row = summary.query(
        "subset == 'all' and metric == 'w_rdist_norm' and prediction == 'AF2'"
    ).iloc[0]
    # AF2 within experimental variability for P1 (0.5 <= 1.0) and P3, not P2
    assert row["n"] == 3
    assert row["within_exp"] == pytest.approx(2 / 3)
    assert row["median_diff"] == pytest.approx(-0.5)  # diffs: -0.5, 1.4, -0.8
    assert row["within_exp_lo"] <= row["within_exp"] <= row["within_exp_hi"]
    clean = summary.query(
        "subset == 'clean' and metric == 'w_rdist_norm' and prediction == 'AF2'"
    ).iloc[0]
    assert clean["n"] == 2  # P3 is flagged
    versus = summary.query(
        "subset == 'all' and metric == 'w_rdist_norm' and prediction == 'AF2 vs OF3'"
    ).iloc[0]
    assert versus["af2_closer"] == pytest.approx(1 / 3)  # only P1 (P2: OF3 closer)


def test_inside_nmr_ensemble(records):
    summary = summarise_proteins(protein_table(records), n_boot=200)
    row = summary.query(
        "subset == 'all' and metric == 'w_rdist_norm' and prediction == 'AF2'"
    ).iloc[0]
    # d(NMR, AF2): P1 0.4, P2 1.0, P3 0.2 against the ensemble maximum 0.3
    assert row["inside_nmr"] == pytest.approx(1 / 3)


def test_bootstrap_ci():
    est, lo, hi = bootstrap_ci([2.0] * 20, np.median, n_boot=200)
    assert est == lo == hi == 2.0
    est, lo, hi = bootstrap_ci([1.0], np.median)
    assert est == 1.0 and np.isnan(lo) and np.isnan(hi)
    values = np.random.default_rng(1).normal(size=100)
    assert bootstrap_ci(values, np.mean, n_boot=500) == bootstrap_ci(
        values, np.mean, n_boot=500
    )  # same seed, same interval


def test_analyse_output_is_valid_json(records):
    result = analyse(records)
    json.dumps(result, allow_nan=False)  # NaN would make the file invalid JSON
    assert "PROTEIN-LEVEL SUMMARY" in result["summary_text"]
    assert len(result["per_protein"]) == 3


def test_tag_aligned_as_mismatches_is_flagged():
    # a 13-residue tag replacing unmodelled residues: aligned, but not identical
    recs = _protein("P4", exp=[1.0], af2=[0.5, 0.5], of3=[0.5, 0.5])
    for r in recs:
        if r["comparison"] == "NMR_vs_AF2":
            r.update(n_residues_a=150, n_matched=150, n_mismatches=13)
    s = experimental_structures(recs).set_index("method")
    assert s.loc["NMR", "extra_residues"] == 13 and s.loc["NMR", "flag_extra"]


def test_title_words_are_whole_words():
    recs = _protein("P5", exp=[1.0], af2=[0.5, 0.5], of3=[0.5, 0.5])
    for r in recs:
        if r["comparison"] == "Cryo-EM_vs_AF2":
            r["title_a"] = "Structure of fibrillarin bound to a filamentous RNA"
    assert not experimental_structures(recs)["flag_state"].any()


def test_flags_fall_back_to_of3_without_af2():
    recs = [
        r
        for r in _protein("P6", exp=[1.0], af2=[0.5, 0.5], of3=[0.5, 0.5])
        if r["category"] != "exp_vs_af2"
    ]
    for r in recs:
        if r["comparison"] == "Cryo-EM_vs_OF3":
            r.update(n_residues_a=100, n_matched=100, seq_identity=0.5)
    t = protein_table(recs).set_index("protein_name")
    assert bool(t.loc["P6", "flag_identity"])
    assert t.loc["P6", "n_experimental"] == 2


def test_n_nmr_counts_only_proteins_with_an_ensemble(records):
    no_ensemble = [
        r
        for r in _protein("P7", exp=[1.0], af2=[0.5, 0.5], of3=[0.5, 0.5])
        if r["category"] != "nmr_intra_ensemble"
    ]
    summary = summarise_proteins(protein_table(records + no_ensemble), n_boot=200)
    row = summary.query(
        "subset == 'all' and metric == 'w_rdist_norm' and prediction == 'AF2'"
    ).iloc[0]
    assert row["n"] == 4 and row["n_nmr"] == 3


def test_dz_of_identical_differences_is_undefined():
    est, lo, hi = bootstrap_ci([0.3, 0.3, 0.3, 0.3], _paired_dz, n_boot=200)
    assert np.isnan(est) and np.isnan(lo) and np.isnan(hi)


# ── training overlap ────────────────────────────────────────────────────────


def test_training_overlap_counts_seen_proteins():
    from src.analysis import training_overlap

    structures = pd.DataFrame(
        {
            "protein_name": ["old", "old", "new", "undated"],
            "pdb_id": ["1AAA", "2BBB", "8CCC", "9DDD"],
            "release_date": ["2010-01-01", "2024-01-01", "2023-05-01", ""],
            "deposition_date": ["2009-12-01", "2023-11-01", "2022-12-01", ""],
            "before_af2_cutoff": [True, False, False, False],
            "before_of3_cutoff": [True, False, False, False],
        }
    )
    overlap = training_overlap(structures).set_index("protein_name")
    assert overlap.loc["old", "n_structures"] == 2 and overlap.loc["old", "n_before_af2"] == 1
    assert overlap.loc["old", "seen_af2"] and overlap.loc["old", "seen_of3"]
    assert not overlap.loc["new", "seen_af2"] and overlap.loc["new", "dated"]
    assert not overlap.loc["undated", "dated"]  # unknown dates must not count as "unseen"
