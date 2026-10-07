"""
Tests for src/sensitivity.py and src/machaon_reference.py on synthetic structures.

The structures are built from known (phi, psi) torsions (tests/synthetic.py),
so what the metrics should do is known in advance.
"""

import numpy as np
import pytest

from src.machaon_reference import compare_with_pipeline, reference_features
from src.metrics import summarise_structure
from src.parser import clear_structure_cache, get_phi_psi_with_index, load_structure
from src.sensitivity import (
    hydrogen_sensitivity,
    orientation_report,
    pca_frame,
    random_rotation,
    rotation_test,
    t_alpha_points,
    truncation_experiment,
)
from tests.synthetic import build_backbone, protein_like_torsions, rotate, rotation_matrix, write_cif

N_RES = 90


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_structure_cache()
    summarise_structure.cache_clear()
    yield
    clear_structure_cache()
    summarise_structure.cache_clear()


@pytest.fixture(scope="module")
def torsions():
    return protein_like_torsions(N_RES, seed=3)


@pytest.fixture
def chain_path(tmp_path, torsions):
    return write_cif(tmp_path / "A.cif", [build_backbone(*torsions)])


@pytest.fixture
def other_path(tmp_path, torsions):
    phi, psi = torsions
    rng = np.random.default_rng(5)
    return write_cif(tmp_path / "B.cif", [build_backbone(phi + rng.normal(0, 8, N_RES), psi + rng.normal(0, 8, N_RES))])


# ── the parser returns degrees (Machaon's unit) ─────────────────────────────


def test_parser_phi_psi_are_degrees(chain_path, torsions):
    phi, psi = torsions
    angles, _ = get_phi_psi_with_index(load_structure(chain_path), 0, "A")
    wrapped = (angles - np.c_[phi[1:-1], psi[1:-1]] + 180) % 360 - 180
    assert np.abs(wrapped).max() < 1e-3


def test_b_phipsi_does_not_depend_on_the_angle_unit():
    """Bhattacharyya distance is unchanged when both angles are rescaled, so the angle unit does not matter."""
    from src.metrics import _bhattacharyya_distance

    rng = np.random.default_rng(0)
    a = rng.normal([-60, -45], [10, 12], (200, 2))
    b = rng.normal([-100, 110], [15, 20], (200, 2))

    def distance(scale):
        x, y = a * scale, b * scale
        cx, cy = np.cov(x.T), np.cov(y.T)
        return _bhattacharyya_distance(x.mean(0), cx, np.linalg.det(cx), y.mean(0), cy, np.linalg.det(cy))

    assert distance(1.0) == pytest.approx(distance(np.pi / 180), rel=1e-9)


# ── orientation of t-alpha ──────────────────────────────────────────────────


def test_pca_frame_is_rotation_invariant():
    rng = np.random.default_rng(1)
    points = rng.standard_normal((300, 3)) * [5.0, 2.0, 1.0]
    a = np.abs(pca_frame(points))
    b = np.abs(pca_frame(points @ random_rotation(rng).T + 7.0))
    assert np.allclose(a, b, atol=1e-8)  # same up to axis signs


def test_t_alpha_depends_on_orientation_in_machaons_frame_but_not_in_pca_frame(chain_path):
    table = rotation_test(
        load_structure_atoms(chain_path), n_rotations=12, seed=0
    )
    assert table["raw"].median() > 0.01  # Machaon's definition: the same chain, rotated, is "different"
    assert table["pca"].max() < 0.02  # principal-axes frame: (nearly) zero
    assert table["pca"].median() < table["raw"].median() / 3


def load_structure_atoms(path):
    from src.parser import get_all_atom_coords

    return get_all_atom_coords(load_structure(path), 0, "A")


def test_orientation_report_keys(chain_path):
    report = orientation_report(chain_path, "A", n_rotations=4)
    assert {"raw_median", "raw_max", "pca_median", "pca_max", "n_atoms"} <= report.keys()
    assert report["n_atoms"] == 4 * N_RES


def test_t_alpha_points_identical_is_zero(chain_path):
    atoms = load_structure_atoms(chain_path)
    assert t_alpha_points(atoms, atoms, "raw") == pytest.approx(0.0)
    assert t_alpha_points(atoms, atoms, "pca") == pytest.approx(0.0)


# ── truncation ──────────────────────────────────────────────────────────────


def test_truncation_zero_is_zero_and_distances_grow(chain_path):
    table = truncation_experiment(chain_path, "A", fractions=(0.0, 0.1, 0.3))
    assert list(table["fraction_removed"]) == [0.0, 0.1, 0.3]
    zero = table.iloc[0]
    assert zero["w_rdist_norm"] == pytest.approx(0.0, abs=1e-9)
    assert zero["b_phipsi"] == pytest.approx(0.0, abs=1e-9)
    assert zero["t_alpha"] == pytest.approx(0.0, abs=1e-9)
    assert table["n_residues"].is_monotonic_decreasing
    assert table["w_rdist_norm"].iloc[2] > table["w_rdist_norm"].iloc[1] > 0


def test_truncation_needs_no_residue_pairing(chain_path):
    """The truncated chain is shorter and nothing is aligned: the metrics are still defined."""
    table = truncation_experiment(chain_path, "A", fractions=(0.2,))
    assert table.loc[0, "n_residues"] == pytest.approx(N_RES * 0.8, abs=1)
    assert np.isfinite(table.loc[0, ["w_rdist_norm", "b_phipsi", "t_alpha"]].astype(float)).all()


# ── hydrogens ───────────────────────────────────────────────────────────────


def test_hydrogen_sensitivity_equal_when_no_hydrogens(chain_path, other_path):
    out = hydrogen_sensitivity(chain_path, "A", other_path, "A")
    assert out["n_atoms_a_heavy"] == out["n_atoms_a_all_atoms"]
    assert out["t_alpha_heavy"] == pytest.approx(out["t_alpha_all_atoms"])


# ── cross-check with the Machaon procedure ──────────────────────────────────


def test_pipeline_agrees_with_machaon_reference(chain_path, other_path):
    table = compare_with_pipeline(chain_path, "A", other_path, "A").set_index("metric")
    assert (table["abs_diff"].astype(float) < 1e-6).all(), table


def test_pipeline_agrees_with_reference_for_a_rotated_copy_too(tmp_path, torsions):
    """Different lengths and orientations: still the same numbers as the reference."""
    phi, psi = torsions
    full = build_backbone(phi, psi)
    short = build_backbone(phi[:70], psi[:70])
    a = write_cif(tmp_path / "full.cif", [full])
    b = write_cif(tmp_path / "short.cif", [rotate(short, rotation_matrix(4), (10, -3, 2))])
    table = compare_with_pipeline(a, "A", b, "A").set_index("metric")
    assert (table["abs_diff"].astype(float) < 1e-6).all(), table


def test_reference_features_ignore_hetero_residues(tmp_path, torsions):
    path = write_cif(tmp_path / "C.cif", [build_backbone(*torsions)])
    features = reference_features(path, "A")
    assert len(features["distances"]) == N_RES * (N_RES - 1) // 2


def test_count_chain_breaks(tmp_path, torsions):
    from src.machaon_reference import count_chain_breaks

    complete = write_cif(tmp_path / "ok.cif", [build_backbone(*torsions)])
    assert count_chain_breaks(complete, "A") == 0
    # remove residues 30-34 from the coordinates: one break
    import gemmi

    st = gemmi.read_structure(complete)
    chain = st[0]["A"]
    for _ in range(5):
        del chain[30]
    broken = str(tmp_path / "broken.cif")
    st.setup_entities()
    st.make_mmcif_document().write_file(broken)
    assert count_chain_breaks(broken, "A") == 1


# ── orientation floor over a sample of chains; t-alpha of records in the PCA frame ──

import pandas as pd  # noqa: E402

from src.sensitivity import (  # noqa: E402
    orientation_floor,
    sample_chains,
    structure_path,
    t_alpha_pca_table,
)


@pytest.fixture
def project(tmp_path, torsions):
    """data/ with an NMR ensemble (3 models), a cryo-EM entry, AF2 and OF3 for one protein."""
    phi, psi = torsions
    data = tmp_path / "data"
    for sub in ("experimental", "alphafold2", "openfold3"):
        (data / sub).mkdir(parents=True)
    truth = build_backbone(phi, psi)
    rng = np.random.default_rng(1)
    models = [build_backbone(phi + rng.normal(0, 3, N_RES), psi + rng.normal(0, 3, N_RES)) for _ in range(3)]
    write_cif(data / "experimental" / "1NMR.cif", models)
    write_cif(data / "experimental" / "2EMX.cif", [rotate(truth, rotation_matrix(4), (5, 1, 2))])
    write_cif(data / "alphafold2" / "AF2_P1.cif", [truth])
    write_cif(data / "openfold3" / "OF3_P1.cif", [build_backbone(phi + 4, psi - 4)])
    manifest = pd.DataFrame(
        [{"protein_name": "Prot", "uniprot_id": "P1", "pdb_xray": "", "chain_xray": "",
          "pdb_nmr": "1NMR", "chain_nmr": "A", "pdb_cryoem": "2EMX", "chain_cryoem": "A"}]
    )
    return manifest, str(data)


def _record(comparison, category, method_a, method_b, pdb_a="", pdb_b="", model_a=0, model_b=0,
            t_alpha=0.1):
    return {
        "protein_name": "Prot", "uniprot_id": "P1", "comparison": comparison, "category": category,
        "status": "ok", "method_a": method_a, "method_b": method_b, "pdb_id_a": pdb_a, "pdb_id_b": pdb_b,
        "chain_id_a": "A", "chain_id_b": "A", "model_idx_a": model_a, "model_idx_b": model_b,
        "t_alpha": t_alpha,
    }


def test_structure_path_resolves_every_kind(project):
    manifest, data = project
    row = manifest.set_index("protein_name").loc["Prot"]
    rec = _record("NMR_vs_AF2", "exp_vs_af2", "NMR", "AF2", pdb_a="1NMR")
    assert structure_path(rec, "a", row, data).endswith("experimental/1NMR.cif")
    assert structure_path(rec, "b", row, data).endswith("alphafold2/AF2_P1.cif")
    intra = _record("nmr_intra_model0_vs_model1", "nmr_intra_ensemble", "NMR", "NMR", model_b=1)
    assert structure_path(intra, "a", row, data).endswith("experimental/1NMR.cif")  # pdb_id from the manifest


def test_sample_chains_and_orientation_floor(project):
    manifest, data = project
    chains = sample_chains(manifest, data, n_per_method=5, seed=0)
    assert set(chains["method"]) == {"NMR", "Cryo-EM"}  # no X-ray in this manifest
    assert sample_chains(manifest, data, n_per_method=5, seed=0).equals(chains)  # reproducible
    floor = orientation_floor(chains, n_rotations=4)
    assert floor["error"].isna().all()
    assert (floor["pca_max"] < 1e-9).all()  # principal-axes frame: no orientation dependence
    assert (floor["raw_max"] >= 0).all()


def test_t_alpha_pca_table(project):
    manifest, data = project
    records = [
        _record("NMR_vs_Cryo-EM", "exp_vs_exp", "NMR", "Cryo-EM", pdb_a="1NMR", pdb_b="2EMX"),
        _record("Cryo-EM_vs_AF2", "exp_vs_af2", "Cryo-EM", "AF2", pdb_a="2EMX"),
        _record("nmr_intra_model0_vs_model2", "nmr_intra_ensemble", "NMR", "NMR", model_b=2),
        _record("X-ray_vs_AF2", "exp_vs_af2", "X-ray", "AF2", pdb_a="9XXX"),  # file not on disk
    ]
    table = t_alpha_pca_table(records, manifest, data)
    assert list(table["status"][:3]) == ["ok", "ok", "ok"]
    assert "not found" in table["status"].iloc[3]
    # cryo-EM chain = rotated copy of the AF2 model: identical in the principal-axes frame
    assert table["t_alpha_pca"].iloc[1] == pytest.approx(0.0, abs=1e-12)


def test_t_alpha_pca_levels():
    from src.sensitivity import t_alpha_pca_levels

    records = pd.DataFrame(
        {
            "protein_name": ["P", "P", "P", "P", "P"],
            "category": ["nmr_intra_ensemble", "nmr_intra_ensemble", "exp_vs_exp", "exp_vs_af2", "exp_vs_of3"],
            "t_alpha_pca": [0.01, 0.03, 0.10, 0.08, 0.20],
            "status": "ok",
        }
    )
    copies = pd.DataFrame(
        {
            "protein_name": ["P", "P", "P"],
            "copy_method": ["X-ray", "X-ray", "Cryo-EM"],
            "rmsd": [0.5, 0.001, 0.4],  # the second copy is identical by symmetry: left out
            "t_alpha_pca": [0.02, 0.0, 0.04],
            "status": "ok",
        }
    )
    levels = t_alpha_pca_levels(records, copies).set_index("level")["value"]
    assert levels["same entry"] == pytest.approx(0.03)  # median over the entries (0.02, 0.04)
    assert levels["NMR ensemble"] == pytest.approx(0.02)
    assert levels["between methods"] == pytest.approx(0.10)
    assert list(levels.index) == ["same entry", "NMR ensemble", "between methods", "AF2", "OF3"]
