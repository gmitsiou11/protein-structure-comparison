"""
Unit tests for src/parser.py.

A tiny mmCIF is written to a temporary folder: chain A has three amino acids
(ALA, selenomethionine MSE stored as HETATM, GLY), plus a calcium ion and a
water that carry the same author chain ID "A" — exactly the situation in real
PDB files.  Only the three amino acids may reach the metrics.
"""

import numpy as np
import pytest

from src.parser import (
    load_structure,
    get_ca_coords_and_seq,
    get_phi_psi_with_index,
    get_all_atom_coords,
    _one_letter,
)

_HEADER = """data_TEST
_entry.id TEST
loop_
_entity.id
_entity.type
1 polymer
2 non-polymer
3 water
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
"""


def _atom_rows() -> list[str]:
    rows, serial = [], 0
    # (group, residue name, label_seq_id) for the three amino acids
    residues = [("ATOM", "ALA", 1), ("HETATM", "MSE", 2), ("ATOM", "GLY", 3)]
    for group, name, seq_id in residues:
        x = 3.8 * seq_id
        backbone = [
            ("N", "N", x - 1.2, 0.5),
            ("CA", "C", x, 0.0),
            ("C", "C", x + 1.2, 0.5),
            ("O", "O", x + 1.3, 1.7),
        ]
        for atom, element, ax, ay in backbone:
            serial += 1
            rows.append(
                f"{group} {serial} {element} {atom} . {name} A 1 {seq_id} ? "
                f"{ax:.3f} {ay:.3f} 0.000 1.00 20.00 {seq_id} {name} A {atom} 1"
            )
    # calcium ion: its atom is ALSO named "CA"
    rows.append(
        f"HETATM {serial + 1} Ca CA . CA B 2 . ? 20.000 20.000 20.000 1.00 20.00 101 CA A CA 1"
    )
    # water
    rows.append(
        f"HETATM {serial + 2} O O . HOH C 3 . ? 25.000 25.000 25.000 1.00 20.00 201 HOH A O 1"
    )
    return rows


@pytest.fixture
def structure(tmp_path):
    path = tmp_path / "test.cif"
    path.write_text(_HEADER + "\n".join(_atom_rows()) + "\n")
    return load_structure(str(path))


def test_sequence_contains_only_amino_acids(structure):
    coords, seq = get_ca_coords_and_seq(structure, 0, "A")
    assert seq == "AMG"  # MSE -> "M", ion and water ignored
    assert coords.shape == (3, 3)


def test_ion_and_water_not_in_atom_cloud(structure):
    atoms = get_all_atom_coords(structure, 0, "A")
    assert atoms.shape == (12, 3)  # 3 residues x (N, CA, C, O)
    assert not np.any(np.all(np.isclose(atoms, 20.0), axis=1))  # calcium
    assert not np.any(np.all(np.isclose(atoms, 25.0), axis=1))  # water


def test_phi_psi_only_where_both_neighbours_exist(structure):
    angles, ca_indices = get_phi_psi_with_index(structure, 0, "A")
    assert angles.shape == (1, 2)  # only the middle residue has phi AND psi
    assert ca_indices == [1]


@pytest.mark.parametrize(
    "name, letter", [("ALA", "A"), ("MSE", "M"), ("HOH", "X"), ("CA", "X")]
)
def test_one_letter(name, letter):
    assert _one_letter(name) == letter


def _altloc_cif(tmp_path) -> str:
    """ALA 1 with its CA in two conformers (B more occupied) and its CB in two equal
    conformers; position 2 modelled as SER then THR (two residue types)."""
    rows = [
        "ATOM 1 N N . ALA A 1 1 ? 0.000 0.000 0.000 1.00 20.00 1 ALA A N 1",
        "ATOM 2 C CA A ALA A 1 1 ? 1.000 0.000 0.000 0.40 20.00 1 ALA A CA 1",
        "ATOM 3 C CA B ALA A 1 1 ? 1.000 5.000 0.000 0.60 20.00 1 ALA A CA 1",
        "ATOM 4 C CB A ALA A 1 1 ? 2.000 0.000 0.000 0.50 20.00 1 ALA A CB 1",
        "ATOM 5 C CB B ALA A 1 1 ? 2.000 9.000 0.000 0.50 20.00 1 ALA A CB 1",
        "ATOM 6 N N A SER A 1 2 ? 4.000 0.000 0.000 0.50 20.00 2 SER A N 1",
        "ATOM 7 C CA A SER A 1 2 ? 5.000 0.000 0.000 0.50 20.00 2 SER A CA 1",
        "ATOM 8 N N B THR A 1 2 ? 4.000 1.000 0.000 0.50 20.00 2 THR A N 1",
        "ATOM 9 C CA B THR A 1 2 ? 5.000 1.000 0.000 0.50 20.00 2 THR A CA 1",
    ]
    path = tmp_path / "altloc.cif"
    path.write_text(_HEADER + "\n".join(rows) + "\n")
    return str(path)


def test_one_conformer_per_atom_as_biopython(tmp_path):
    structure = load_structure(_altloc_cif(tmp_path))
    ala = structure[0]["A"][0]
    assert [a.name for a in ala] == ["N", "CA", "CB"]
    assert ala["CA"][0].pos.y == 5.0  # higher occupancy wins
    assert ala["CB"][0].pos.y == 0.0  # tie: the first in the file


def test_two_residue_types_keep_the_last(tmp_path):
    structure = load_structure(_altloc_cif(tmp_path))
    coords, seq = get_ca_coords_and_seq(structure, 0, "A")
    assert seq == "AT"
    assert coords.shape == (2, 3)


def test_hybrid_entry_gets_its_recognised_method(tmp_path):
    from src.metadata import extract_metadata

    header = _HEADER.replace(
        "data_TEST\n_entry.id TEST\n",
        "data_TEST\n_entry.id TEST\nloop_\n_exptl.entry_id\n_exptl.method\n"
        "TEST 'SOLUTION SCATTERING'\nTEST 'SOLUTION NMR'\n",
    )
    path = tmp_path / "hybrid.cif"
    path.write_text(header + "\n".join(_atom_rows()) + "\n")
    meta = extract_metadata(str(path), chain_id="A")
    assert meta.method == "SOLUTION SCATTERING; SOLUTION NMR"
    assert meta.method_category == "NMR"
