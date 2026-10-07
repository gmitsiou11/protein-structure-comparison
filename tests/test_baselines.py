"""Tests for src/baselines.py: method-pair tables, the within-variability rules, entry copies."""

import gemmi
import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    BASELINES,
    PAIR_ORDER,
    SYMMETRY_RMSD,
    baseline_levels,
    closest_experimental_method,
    compare_entry_copies,
    context_table,
    cryoem_assembly_size,
    find_entry_copies,
    method_pair_table,
    pair_label,
    summarise_copies,
    within_by_baseline,
)
from src.metrics import summarise_structure
from src.parser import clear_structure_cache


def _rec(protein, comparison, category, rmsd, w=0.1, b=0.01, t=0.1, status="ok"):
    return {
        "protein_name": protein,
        "comparison": comparison,
        "category": category,
        "status": status,
        "rmsd": rmsd,
        "w_rdist_norm": w,
        "b_phipsi": b,
        "t_alpha": t,
    }


@pytest.fixture
def records():
    """Two proteins: P1 has X-ray, NMR, cryo-EM; P2 has NMR and cryo-EM only."""
    return [
        # P1: experiments
        _rec("P1", "X-ray_vs_NMR", "exp_vs_exp", 3.0),
        _rec("P1", "X-ray_vs_Cryo-EM", "exp_vs_exp", 1.0),
        _rec("P1", "NMR_vs_Cryo-EM", "exp_vs_exp", 4.0),
        # P1: AF2 is closest to X-ray
        _rec("P1", "X-ray_vs_AF2", "exp_vs_af2", 0.5),
        _rec("P1", "NMR_vs_AF2", "exp_vs_af2", 4.5),
        _rec("P1", "Cryo-EM_vs_AF2", "exp_vs_af2", 1.5),
        # P1: OF3
        _rec("P1", "X-ray_vs_OF3", "exp_vs_of3", 2.0),
        _rec("P1", "NMR_vs_OF3", "exp_vs_of3", 5.0),
        _rec("P1", "Cryo-EM_vs_OF3", "exp_vs_of3", 2.5),
        _rec("P1", "AF2_vs_OF3", "af2_vs_of3", 2.2),
        # P2: one experimental pair
        _rec("P2", "NMR_vs_Cryo-EM", "exp_vs_exp", 2.0),
        _rec("P2", "NMR_vs_AF2", "exp_vs_af2", 2.5),
        _rec("P2", "Cryo-EM_vs_AF2", "exp_vs_af2", 1.0),
        _rec("P2", "NMR_vs_OF3", "exp_vs_of3", 1.0),
        _rec("P2", "Cryo-EM_vs_OF3", "exp_vs_of3", 1.5),
        _rec("P2", "AF2_vs_OF3", "af2_vs_of3", 0.4),
        # excluded: NMR ensemble and a failed pair
        _rec("P1", "NMR_vs_NMR", "nmr_intra_ensemble", 9.0),
        _rec("P2", "X-ray_vs_AF2", "exp_vs_af2", 99.0, status="rejected"),
    ]


def test_pair_label_is_unordered():
    assert pair_label("NMR_vs_AF2") == "NMR–AF2"
    assert pair_label("AF2_vs_NMR") == "NMR–AF2"
    assert pair_label("Cryo-EM_vs_X-ray") == "X-ray–Cryo-EM"
    assert pair_label("AF2_vs_OF3") == "AF2–OF3"
    with pytest.raises(ValueError):
        pair_label("NMR_vs_AF3")


def test_method_pair_table(records):
    table = method_pair_table(records)
    assert list(table.index) == [p for p in PAIR_ORDER if p in table.index]
    assert table.loc["NMR–Cryo-EM", "rmsd"] == pytest.approx(3.0)  # median of 4.0 and 2.0
    assert table.loc["NMR–Cryo-EM", "n_proteins"] == 2
    assert table.loc["X-ray–AF2", "rmsd"] == pytest.approx(0.5)  # rejected 99.0 excluded
    assert table.loc["X-ray–AF2", "n_proteins"] == 1
    assert "NMR–NMR" not in table.index  # ensemble records excluded


def test_closest_experimental_method(records):
    counts = closest_experimental_method(records, "AF2", "rmsd")
    assert counts.to_dict() == {"X-ray": 1, "NMR": 0, "Cryo-EM": 0}  # P2 has no X-ray
    counts_of3 = closest_experimental_method(records, "OF3", "rmsd")
    assert counts_of3.sum() == 1


# ── entry copies ────────────────────────────────────────────────────────────

_RNG = np.random.default_rng(0)
_CA = np.cumsum(_RNG.normal(0, 2.2, (40, 3)), axis=0)  # a compact-ish random chain
_OFFSETS = {"N": (-0.5, 1.2, -0.6), "CA": (0, 0, 0), "C": (0.9, -0.9, 0.5), "O": (1.8, -0.4, 1.0)}


def _write_entry(path, chains: dict) -> str:
    """Write one model with several chains; chains = {name: Cα array}."""
    structure = gemmi.Structure()
    structure.name = "COPY"
    model = gemmi.Model("1")
    for name, ca in chains.items():
        chain = gemmi.Chain(name)
        for i, pos in enumerate(ca, start=1):
            residue = gemmi.Residue()
            residue.name = "ALA"
            residue.seqid = gemmi.SeqId(i, " ")
            residue.entity_type = gemmi.EntityType.Polymer
            for atom_name, offset in _OFFSETS.items():
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(atom_name[0])
                atom.pos = gemmi.Position(*(pos + np.array(offset)))
                atom.occ = 1.0
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(path))
    return str(path)


@pytest.fixture
def entry(tmp_path):
    clear_structure_cache()
    summarise_structure.cache_clear()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    chains = {
        "A": _CA,
        "B": _CA @ rotation.T + 25.0,  # same shape, moved: identical by construction
        "C": _CA + _RNG.normal(0, 0.8, _CA.shape),  # a genuinely different copy
    }
    (tmp_path / "experimental").mkdir()
    _write_entry(tmp_path / "experimental" / "9ZZZ.cif", chains)
    yield tmp_path
    clear_structure_cache()
    summarise_structure.cache_clear()


def test_find_entry_copies():
    manifest = pd.DataFrame(
        [
            {
                "protein_name": "Prot",
                "uniprot_id": "P99999",
                "pdb_xray": "",
                "chain_xray": "",
                "pdb_nmr": "1ABC",
                "chain_nmr": "A",
                "pdb_cryoem": "9ZZZ",
                "chain_cryoem": "A",
            }
        ]
    )
    segments = pd.DataFrame(
        {
            "pdb": ["9zzz"] * 4 + ["1abc"],
            "chain": ["A", "B", "C", "D", "A"],
            "uniprot": ["P99999", "P99999", "P99999", "Q11111", "P99999"],
        }
    )
    copies = find_entry_copies(manifest, segments, max_copies=5)
    assert list(copies["chain_copy"]) == ["B", "C"]  # D is another protein; NMR has no copy
    assert set(copies["method"]) == {"Cryo-EM"}
    assert set(copies["n_copies"]) == {3}
    assert len(find_entry_copies(manifest, segments, max_copies=1)) == 1


def test_compare_and_summarise_copies(entry):
    copies = pd.DataFrame(
        {
            "protein_name": ["Prot"] * 3,
            "uniprot_id": ["P99999"] * 3,
            "method": ["Cryo-EM"] * 3,
            "pdb_id": ["9ZZZ", "9ZZZ", "8YYY"],
            "chain_ref": ["A", "A", "A"],
            "chain_copy": ["B", "C", "B"],
            "n_copies": [3, 3, 2],
        }
    )
    records = compare_entry_copies(copies, data_dir=str(entry))
    assert [r["status"] for r in records] == ["ok", "ok", "missing_file"]
    identical, varied = records[0], records[1]
    assert identical["rmsd"] < SYMMETRY_RMSD
    assert identical["w_rdist_raw"] == pytest.approx(0.0, abs=1e-9)
    assert varied["rmsd"] > 0.3 and varied["w_rdist_raw"] > 0
    assert varied["chain_ref"] == "A" and varied["chain_copy"] == "C"

    summary = summarise_copies(records).set_index("protein_name")
    assert summary.loc["Prot", "n_pairs"] == 2
    assert summary.loc["Prot", "n_identical"] == 1
    assert summary.loc["Prot", "rmsd"] == pytest.approx(varied["rmsd"])  # identical pair left out


def test_baseline_levels():
    table = pd.DataFrame(
        {
            "protein_name": ["P1", "P2"],
            "rmsd__nmr_ens": [1.0, np.nan],
            "rmsd__exp": [3.0, 2.0],
            "rmsd__af2": [1.5, 1.75],
            "rmsd__of3": [2.5, 1.25],
        }
    )
    copy_summary = pd.DataFrame(
        {"protein_name": ["P1", "P1"], "copy_method": ["Cryo-EM", "X-ray"], "rmsd": [0.2, 0.4]}
    )
    levels = baseline_levels(table, copy_summary, "rmsd")
    counts = levels["level"].value_counts().to_dict()
    assert counts == {"between methods": 2, "AF2": 2, "OF3": 2, "same entry": 1, "NMR ensemble": 1}
    same = levels[levels["level"] == "same entry"]["value"].iloc[0]
    assert same == pytest.approx(0.3)  # median over the protein's entries


# ── baseline choice and context ─────────────────────────────────────────────


def test_within_by_baseline_depends_on_the_baseline(records):
    table = within_by_baseline(records, ["rmsd"], n_boot=200).set_index(["prediction", "baseline"])
    # P1: AF2 median 1.5; experimental pairs 3.0 / 1.0 / 4.0.  P2: AF2 median 1.75; one pair, 2.0.
    assert table.loc[("AF2", "pooled, median rule"), "fraction"] == pytest.approx(1.0)
    assert table.loc[("AF2", "X-ray–Cryo-EM only"), "n"] == 1  # only P1 has X-ray
    assert table.loc[("AF2", "X-ray–Cryo-EM only"), "fraction"] == pytest.approx(0.0)  # 1.5 > 1.0
    assert table.loc[("AF2", "NMR–Cryo-EM only"), "fraction"] == pytest.approx(1.0)
    assert table.loc[("AF2", "tightest experimental pair"), "fraction"] == pytest.approx(0.5)
    assert table.loc[("AF2", "pooled, strict rule"), "fraction"] == pytest.approx(0.0)
    # OF3: P1 median 2.5 <= 3.0, max 5.0 > 4.0; P2 median 1.25 <= 2.0, max 1.5 <= 2.0
    assert table.loc[("OF3", "pooled, median rule"), "fraction"] == pytest.approx(1.0)
    assert table.loc[("OF3", "pooled, strict rule"), "fraction"] == pytest.approx(0.5)


def test_within_by_baseline_has_every_baseline_and_ordered_interval(records):
    table = within_by_baseline(records, ["rmsd", "w_rdist_norm"], n_boot=200)
    assert set(table["baseline"]) == set(BASELINES)
    ok = table.dropna(subset=["fraction"])
    assert ((ok["lo"] <= ok["fraction"] + 1e-12) & (ok["fraction"] <= ok["hi"] + 1e-12)).all()


def _context_records():
    def rec(protein, comparison, method_a, method_b, chains_a, chains_b, rmsd):
        return {
            "protein_name": protein,
            "comparison": comparison,
            "category": "exp_vs_exp" if "AF2" not in comparison else "exp_vs_af2",
            "status": "ok",
            "method_a": method_a,
            "method_b": method_b,
            "n_polymer_chains_a": chains_a,
            "n_polymer_chains_b": chains_b,
            "rmsd": rmsd,
            "w_rdist_norm": 0.1,
            "b_phipsi": 0.01,
            "t_alpha": 0.1,
        }

    return [
        rec("small", "NMR_vs_Cryo-EM", "NMR", "Cryo-EM", 1, 3, 2.0),
        rec("small", "Cryo-EM_vs_AF2", "Cryo-EM", "AF2", 3, None, 1.0),
        rec("big", "NMR_vs_Cryo-EM", "NMR", "Cryo-EM", 1, 60, 6.0),
        rec("big", "Cryo-EM_vs_AF2", "Cryo-EM", "AF2", 60, None, 1.2),
    ]


def test_cryoem_assembly_size():
    sizes = cryoem_assembly_size(_context_records())
    assert sizes.to_dict() == {"small": 3, "big": 60}


def test_context_table_splits_by_assembly_size():
    table = context_table(_context_records(), ["rmsd"]).set_index(["assembly", "pair"])
    assert table.loc[("1–4 chains", "NMR–Cryo-EM"), "rmsd"] == pytest.approx(2.0)
    assert table.loc[("20+ chains", "NMR–Cryo-EM"), "rmsd"] == pytest.approx(6.0)
    assert table.loc[("20+ chains", "Cryo-EM–AF2"), "rmsd"] == pytest.approx(1.2)
    assert ("5–19 chains", "NMR–Cryo-EM") not in table.index  # empty bins are omitted


# ── coverage, reference levels, robustness ──────────────────────────────────

from src.baselines import (  # noqa: E402
    BASELINE_CONDITIONS,
    ROBUSTNESS_CONDITIONS,
    coverage_dependence,
    length_gap,
    pair_medians_by_coverage,
    protein_max_gap,
    protein_strata,
    reference_levels_table,
    robustness_table,
    robustness_wide,
)

_EXP = ("X-ray", "NMR", "Cryo-EM")


def _full(protein, comparison, n_a, n_b, rmsd, w, chains_em=3, release="2015-01-01", plddt=90.0):
    """A record with the fields protein_table / experimental_structures read."""
    a, _, b = comparison.partition("_vs_")
    category = "exp_vs_exp"
    if "AF2" in (a, b) and "OF3" in (a, b):
        category = "af2_vs_of3"
    elif "AF2" in (a, b):
        category = "exp_vs_af2"
    elif "OF3" in (a, b):
        category = "exp_vs_of3"
    rec = {
        "protein_name": protein,
        "uniprot_id": f"U_{protein}",
        "comparison": comparison,
        "category": category,
        "status": "ok",
        "n_residues_a": n_a,
        "n_residues_b": n_b,
        "n_matched": min(n_a, n_b),
        "n_mismatches": 0,
        "seq_identity": 1.0,
        "rmsd": rmsd,
        "w_rdist_norm": w,
        "b_phipsi": 0.01,
        "t_alpha": 0.1,
    }
    for side, method in (("a", a), ("b", b)):
        rec[f"method_{side}"] = method
        rec[f"pdb_id_{side}"] = f"{protein[:2]}{method[:1]}" if method in _EXP else ""
        rec[f"chain_id_{side}"] = "A"
        rec[f"n_polymer_chains_{side}"] = chains_em if method == "Cryo-EM" else 1
        rec[f"release_date_{side}"] = release if method in _EXP else ""
        rec[f"deposition_date_{side}"] = release if method in _EXP else ""
        rec[f"title_{side}"] = "a protein"
        rec[f"plddt_mean_{side}"] = plddt if method in ("AF2", "OF3") else None
    return rec


@pytest.fixture
def rich_records():
    """
    'even': all chains 100 residues, AF2 inside the experimental spread.
    'gappy': NMR chain 80 of 100 residues (length gap 0.2), large w-rdist where NMR takes part.
    'new':  all structures released after both training cutoffs, cryo-EM from a 30-chain entry.
    """
    recs = []
    for protein, n_nmr, release, chains in (("even", 100, "2015-01-01", 3), ("gappy", 80, "2015-01-01", 3),
                                            ("new", 100, "2023-01-01", 30)):
        w_nmr = 0.6 if n_nmr < 100 else 0.2
        recs += [
            _full(protein, "NMR_vs_Cryo-EM", n_nmr, 100, 3.0, w_nmr, chains, release),
            _full(protein, "NMR_vs_AF2", n_nmr, 100, 2.0, w_nmr * 0.9, chains, release),
            _full(protein, "Cryo-EM_vs_AF2", 100, 100, 1.0, 0.1, chains, release),
            _full(protein, "NMR_vs_OF3", n_nmr, 100, 4.0, w_nmr * 1.1, chains, release, plddt=60.0),
            _full(protein, "Cryo-EM_vs_OF3", 100, 100, 3.5, 0.3, chains, release, plddt=60.0),
            _full(protein, "AF2_vs_OF3", 100, 100, 2.0, 0.2, chains, release),
        ]
    return recs


def test_length_gap():
    assert length_gap(100, 100) == 0.0
    assert length_gap(80, 100) == pytest.approx(0.2)
    assert length_gap(100, 80) == pytest.approx(0.2)
    assert np.isnan(length_gap(0, 0))


def test_protein_max_gap(rich_records):
    gaps = protein_max_gap(rich_records)
    assert gaps["even"] == 0.0 and gaps["new"] == 0.0
    assert gaps["gappy"] == pytest.approx(0.2)


def test_coverage_dependence_sees_the_gap(rich_records):
    table = coverage_dependence(rich_records, ["w_rdist_norm", "rmsd"]).set_index(["category", "metric"])
    # w-rdist is larger exactly where the NMR chain is shorter; RMSD does not follow the gap
    assert table.loc[("exp_vs_af2", "w_rdist_norm"), "rho"] > 0.5
    assert np.isnan(table.loc[("af2_vs_of3", "w_rdist_norm"), "rho"])  # no gap variation at all


def test_pair_medians_by_coverage_drops_gappy_pairs(rich_records):
    table = pair_medians_by_coverage(rich_records, max_gaps=(None, 0.05), metrics=["w_rdist_norm"])
    table = table.set_index(["max_gap", "pair"])
    assert table.loc[("all", "NMR–Cryo-EM"), "n_proteins"] == 3
    assert table.loc[(0.05, "NMR–Cryo-EM"), "n_proteins"] == 2  # 'gappy' is left out
    assert table.loc[(0.05, "NMR–Cryo-EM"), "w_rdist_norm"] == pytest.approx(0.2)


def test_protein_strata(rich_records):
    strata = protein_strata(rich_records).set_index("protein_name")
    assert not strata.loc["new", "seen_af2"] and not strata.loc["new", "seen_of3"]
    assert strata.loc["even", "seen_af2"] and strata.loc["even", "dated"]
    assert strata.loc["new", "cryoem_chains"] == 30
    assert strata.loc["gappy", "max_length_gap"] == pytest.approx(0.2)


def test_robustness_table_conditions(rich_records):
    robust = robustness_table(rich_records, metrics=["w_rdist_norm", "rmsd"], n_boot=200)
    proteins = robust[robust["block"] == "proteins"]
    assert set(proteins["condition"]) == set(ROBUSTNESS_CONDITIONS)
    assert set(robust.loc[robust["block"] == "baseline", "condition"]) == set(BASELINE_CONDITIONS.values())
    get = proteins.set_index(["condition", "metric", "prediction"])
    assert get.loc[("all proteins", "rmsd", "AF2"), "n"] == 3
    assert get.loc[("length gap ≤ 5 % in every pair", "rmsd", "AF2"), "n"] == 2
    assert get.loc[("no structure public before the model's training cutoff", "rmsd", "AF2"), "n"] == 1
    assert get.loc[("cryo-EM entry with 20+ chains", "rmsd", "OF3"), "n"] == 1
    assert get.loc[("model confidence pLDDT ≥ 70", "rmsd", "OF3"), "n"] == 0  # OF3 pLDDT is 60 everywhere
    # AF2: median(2.0, 1.0) = 1.5 <= 3.0 for every protein
    assert get.loc[("all proteins", "rmsd", "AF2"), "fraction"] == pytest.approx(1.0)
    wide = robustness_wide(robust, ["rmsd"])
    assert wide.index[0] == ("proteins", "all proteins")
    assert wide.loc[("proteins", "all proteins"), ("rmsd", "AF2")].startswith("1.00")


def test_reference_levels_table():
    table = pd.DataFrame(
        {
            "protein_name": ["P1", "P2"],
            "rmsd__nmr_ens": [1.0, 3.0],
            "rmsd__exp": [3.0, 5.0],
            "rmsd__af2": [1.5, 2.5],
            "rmsd__of3": [2.5, 4.5],
        }
    )
    copies = pd.DataFrame({"protein_name": ["P1"], "copy_method": ["X-ray"], "rmsd": [0.2]})
    levels = reference_levels_table(table, copies, ["rmsd"])
    assert list(levels.index) == ["same entry", "NMR ensemble", "between methods", "AF2", "OF3"]
    assert levels.loc["NMR ensemble", "rmsd"] == pytest.approx(2.0)
    assert levels.loc["same entry", "n_rmsd"] == 1
