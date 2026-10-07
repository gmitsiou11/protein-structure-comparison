"""
Tests for residue_range: selecting residues by number (used by the truncation
test, sensitivity.truncation_experiment).

A synthetic, compact chain of N alanines numbered 1..N is written with gemmi.
Selecting residues FIRST..LAST must give exactly what a chain containing only
those residues gives — except that the φ/ψ of the two boundary residues are
still computed with their neighbours in the whole chain, as Machaon's
residue_selection does.
"""

import gemmi
import numpy as np
import pytest

from src.parser import (
    load_structure,
    clear_structure_cache,
    get_ca_coords_and_seq,
    get_phi_psi_with_index,
    get_all_atom_coords,
)
from src.metrics import summarise_structure, _w_rdist

N = 60
FIRST, LAST = 11, 50  # the residues kept in these tests
_ATOMS = {  # offsets from Cα (Å); only a rough backbone is needed
    "N": (-0.5, 1.2, -0.6),
    "CA": (0.0, 0.0, 0.0),
    "C": (0.9, -0.9, 0.5),
    "O": (1.8, -0.4, 1.0),
    "CB": (-1.0, -0.8, 0.3),
}
# Cα of residue i = _CA[i]: a seeded, compact (globule-like) cloud, identical in every file
_CA = np.random.default_rng(0).standard_normal((N + 1, 3)) * 6.0


def _write_chain(path, numbers) -> str:
    """Write chain A with one ALA per residue number."""
    structure = gemmi.Structure()
    structure.name = "TEST"
    chain = gemmi.Chain("A")
    for i in numbers:
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(i, " ")
        residue.label_seq = i
        residue.entity_type = gemmi.EntityType.Polymer
        ca = _CA[i]
        for name, offset in _ATOMS.items():
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element(name[0])
            atom.pos = gemmi.Position(*(ca + np.array(offset)))
            atom.occ = 1.0
            residue.add_atom(atom)
        chain.add_residue(residue)
    model = gemmi.Model("1")
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(path))
    return str(path)


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_structure_cache()
    summarise_structure.cache_clear()
    yield
    clear_structure_cache()
    summarise_structure.cache_clear()


@pytest.fixture
def full_path(tmp_path):
    return _write_chain(tmp_path / "FULL.cif", range(1, N + 1))


@pytest.fixture
def part_path(tmp_path):
    return _write_chain(tmp_path / "PART.cif", range(FIRST, LAST + 1))


def test_range_selects_residues_by_number(full_path):
    structure = load_structure(full_path)
    coords, seq = get_ca_coords_and_seq(structure, 0, "A", (FIRST, LAST))
    n = LAST - FIRST + 1
    assert seq == "A" * n
    assert coords.shape == (n, 3)
    assert get_all_atom_coords(structure, 0, "A", (FIRST, LAST)).shape == (n * 5, 3)
    assert get_ca_coords_and_seq(structure, 0, "A")[0].shape == (N, 3)  # no range


def test_boundary_angles_use_whole_chain_neighbours(full_path):
    structure = load_structure(full_path)
    angles, idx = get_phi_psi_with_index(structure, 0, "A", (FIRST, LAST))
    all_angles, _ = get_phi_psi_with_index(structure, 0, "A")
    n = LAST - FIRST + 1
    assert angles.shape == (n, 2)  # residues FIRST and LAST keep phi and psi
    assert idx == list(range(n))  # indices into the selected Cα array
    # whole chain: residues 2..N-1 have both angles, so residue FIRST is row FIRST-2
    np.testing.assert_allclose(angles, all_angles[FIRST - 2 : LAST - 1])


def test_summary_of_range_equals_chain_of_those_residues(full_path, part_path):
    selected = summarise_structure(full_path, 0, "A", (FIRST, LAST))
    part = summarise_structure(part_path, 0, "A")
    assert selected.seq == part.seq
    assert _w_rdist(selected.dists, part.dists)["raw"] == 0.0
    assert selected.n_tri == part.n_tri
    # the chain that really ends there has no phi at FIRST and no psi at LAST
    assert selected.n_phipsi == part.n_phipsi + 2
