"""
Tests for alignment.align_pair: when RMSD is computed.

RMSD needs the paired residues to be the same residues.  A point mutant pairs
correctly (RMSD computed); a different protein, here an unrelated sequence on
the same backbone, does not (identity below RMSD_IDENTITY_MIN: no RMSD).
"""

from src.parser import clear_structure_cache
from src.alignment import RMSD_IDENTITY_MIN, align_pair
from tests.synthetic import build_backbone, protein_like_torsions, write_cif

N_RES = 60


def _pair(tmp_path, sequence_b):
    phi, psi = protein_like_torsions(N_RES, seed=1)
    backbone = build_backbone(phi, psi)
    clear_structure_cache()
    a = write_cif(tmp_path / "A.cif", [backbone], sequence="A" * N_RES)
    b = write_cif(tmp_path / "B.cif", [backbone], sequence=sequence_b)
    return align_pair(a, b)


def test_mutant_keeps_rmsd(tmp_path):
    mutant = "".join("G" if i % 4 == 0 else "A" for i in range(N_RES))  # 75 % identical
    result = _pair(tmp_path, mutant)
    assert result.seq_identity == 0.75 >= RMSD_IDENTITY_MIN
    assert result.rmsd_reliable and result.warnings == []


def test_different_protein_has_no_rmsd(tmp_path):
    result = _pair(tmp_path, "G" * N_RES)  # nothing identical
    assert result.valid and result.seq_identity == 0.0
    assert not result.rmsd_reliable
    assert any("not the same protein" in w for w in result.warnings)
