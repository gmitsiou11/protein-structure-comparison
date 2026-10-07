"""
Tests for src/nmr.py (which model of an NMR ensemble represents the entry)
and for its use by run_pipeline, on synthetic structures.

The ensemble has 8 models: seven close to each other and one far outlier that
is placed FIRST (model 0).  A rule that picks model 0 therefore picks the worst
possible representative; the medoid must not.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.metrics import summarise_structure
from src.nmr import choose_nmr_model, ensemble_medoid, kabsch_rmsd, pdb_representative_model
from src.parser import clear_structure_cache
from src.pipeline import run_pipeline
from tests.synthetic import build_backbone, protein_like_torsions, rotate, rotation_matrix, write_cif

N_RES = 70
OUTLIER_FIRST = 0
PDB_REPRESENTATIVE = 4  # 1-based conformer id written into the file -> index 3


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_structure_cache()
    summarise_structure.cache_clear()
    yield
    clear_structure_cache()
    summarise_structure.cache_clear()


def _ensemble(seed=0):
    phi, psi = protein_like_torsions(N_RES, seed)
    rng = np.random.default_rng(seed + 1)
    models = []
    for k in range(8):
        noise = 25.0 if k == OUTLIER_FIRST else 2.0
        models.append(build_backbone(phi + rng.normal(0, noise, N_RES), psi + rng.normal(0, noise, N_RES)))
    return phi, psi, models


@pytest.fixture
def nmr_file(tmp_path):
    _, _, models = _ensemble()
    return write_cif(tmp_path / "NMR1.cif", models, representative=PDB_REPRESENTATIVE)


def test_kabsch_rmsd_zero_for_rotated_copy():
    _, _, models = _ensemble()
    a = models[1]["CA"]
    b = a @ rotation_matrix(3).T + 5.0
    assert kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-6)


def test_pdb_representative_is_read_from_the_file(nmr_file):
    assert pdb_representative_model(nmr_file) == PDB_REPRESENTATIVE - 1


def test_pdb_representative_missing_returns_none(tmp_path):
    _, _, models = _ensemble()
    path = write_cif(tmp_path / "NMR2.cif", models, representative=None)
    assert pdb_representative_model(path) is None


def test_medoid_is_not_the_outlier(nmr_file):
    model, mean_rmsd = ensemble_medoid(nmr_file, "A")
    assert model != OUTLIER_FIRST
    assert mean_rmsd > 0


def test_medoid_is_deterministic(nmr_file):
    assert ensemble_medoid(nmr_file, "A") == ensemble_medoid(nmr_file, "A")


@pytest.mark.parametrize(
    "policy, expected_model, expected_rule",
    [
        ("first", 0, "first"),
        ("pdb", PDB_REPRESENTATIVE - 1, "pdb"),
    ],
)
def test_policies(nmr_file, policy, expected_model, expected_rule):
    info = choose_nmr_model(nmr_file, "A", policy=policy)
    assert info["model"] == expected_model
    assert info["rule"] == expected_rule
    assert info["n_models"] == 8


def test_medoid_policy_reports_alternatives(nmr_file):
    info = choose_nmr_model(nmr_file, "A", policy="medoid")
    assert info["model"] == info["medoid"] != OUTLIER_FIRST
    assert info["pdb_representative"] == PDB_REPRESENTATIVE - 1
    assert info["medoid_mean_rmsd"] > 0


def test_pdb_policy_falls_back_to_medoid(tmp_path):
    _, _, models = _ensemble()
    path = write_cif(tmp_path / "NMR3.cif", models, representative=None)
    info = choose_nmr_model(path, "A", policy="pdb")
    assert info["rule"] == "pdb->medoid"
    assert info["model"] == info["medoid"]


def test_single_model_entry(tmp_path):
    _, _, models = _ensemble()
    path = write_cif(tmp_path / "NMR4.cif", models[1:2])
    info = choose_nmr_model(path, "A", policy="medoid")
    assert info["model"] == 0 and info["rule"] == "single model"


def test_unknown_policy_is_an_error(nmr_file):
    with pytest.raises(ValueError):
        choose_nmr_model(nmr_file, "A", policy="best")


# ── end to end: run_pipeline on two synthetic proteins ──────────────────────


@pytest.fixture
def mini_project(tmp_path):
    """data/ with an NMR ensemble, a cryo-EM structure, AF2 and OF3 for two proteins."""
    data = tmp_path / "data"
    for sub in ("experimental", "alphafold2", "openfold3"):
        (data / sub).mkdir(parents=True)
    rows = []
    for k, (uniprot, pdb_nmr, pdb_em) in enumerate([("P00001", "1AAA", "2AAA"), ("P00002", "1BBB", "2BBB")]):
        phi, psi, models = _ensemble(seed=10 * k)
        write_cif(data / "experimental" / f"{pdb_nmr}.cif", models, representative=PDB_REPRESENTATIVE)
        truth = build_backbone(phi, psi)
        write_cif(data / "experimental" / f"{pdb_em}.cif", [rotate(truth, rotation_matrix(k), (3, 4, 5))])
        write_cif(data / "alphafold2" / f"AF2_{uniprot}.cif", [truth])
        write_cif(data / "openfold3" / f"OF3_{uniprot}.cif", [build_backbone(phi + 3, psi - 3)])
        rows.append(
            {
                "protein_name": f"Protein {k}",
                "uniprot_id": uniprot,
                "mature_start": 1,
                "mature_end": N_RES,
                "pdb_xray": "",
                "chain_xray": "",
                "pdb_nmr": pdb_nmr,
                "chain_nmr": "A",
                "pdb_cryoem": pdb_em,
                "chain_cryoem": "A",
            }
        )
    csv = tmp_path / "proteins.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return str(csv), str(data), tmp_path


def _run(project, policy):
    csv, data, root = project
    out = root / f"results_{policy}"
    result = run_pipeline(csv, data, str(out), verbose=False, nmr_model_policy=policy)
    return result, out


def test_pipeline_uses_the_medoid_by_default(mini_project):
    result, out = _run(mini_project, "medoid")
    models = pd.read_csv(out / "nmr_models.csv")
    assert len(models) == 2
    assert (models["model_used"] != OUTLIER_FIRST).all()
    assert (models["rule"] == "medoid").all()
    assert (models["pdb_representative"] == PDB_REPRESENTATIVE - 1).all()
    nmr_records = [r for r in result["comparisons"] if r.get("nmr_model_used") is not None]
    assert nmr_records and all(r["nmr_model_rule"] == "medoid" for r in nmr_records)
    assert result["rejected"] == []


def test_pipeline_first_policy_uses_model_zero(mini_project):
    result, out = _run(mini_project, "first")
    models = pd.read_csv(out / "nmr_models.csv")
    assert (models["model_used"] == 0).all()


def test_medoid_gives_smaller_nmr_distances_than_an_outlier_model_zero(mini_project):
    """The reason for the change: NMR-vs-cryo-EM distance with the outlier as representative is inflated."""
    medoid, _ = _run(mini_project, "medoid")
    zero, _ = _run(mini_project, "first")

    def median_rmsd(result):
        return float(
            np.median([r["rmsd"] for r in result["comparisons"] if r["comparison"] == "NMR_vs_Cryo-EM"])
        )

    assert median_rmsd(medoid) < median_rmsd(zero)


def test_pipeline_rejects_unknown_policy(mini_project):
    csv, data, root = mini_project
    with pytest.raises(ValueError):
        run_pipeline(csv, data, str(root / "r"), verbose=False, nmr_model_policy="nope")


# ── scripts/verify_run.py on a real (synthetic-data) run ────────────────────


def _verify(results_dir, *extra):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "verify_run", os.path.join(os.path.dirname(__file__), "..", "scripts", "verify_run.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main([str(results_dir), *extra])


def test_verify_run_passes_on_a_medoid_run(mini_project, capsys):
    _, out = _run(mini_project, "medoid")
    assert _verify(out, "--proteins", "2", "--expect-policy", "medoid") == 0


def test_verify_run_flags_the_wrong_policy(mini_project, capsys):
    _, out = _run(mini_project, "first")
    assert _verify(out, "--expect-policy", "medoid") == 1
    assert "do not belong to policy" in capsys.readouterr().out


def test_verify_run_flags_old_results_without_nmr_models(mini_project, capsys):
    _, out = _run(mini_project, "medoid")
    os.remove(out / "nmr_models.csv")
    assert _verify(out) == 1
    assert "nmr_models.csv missing" in capsys.readouterr().out
