"""
Synthetic protein-like structures for tests (no files from the PDB needed).

``build_backbone(phi, psi)`` places N, CA, C, O atoms with ideal bond lengths
and angles from a list of (phi, psi) torsions (the NeRF construction), so a
structure has known backbone angles and a realistic Cα–Cα spacing.
``write_cif`` writes one or several models as an mmCIF that src/parser.py reads.
"""

from __future__ import annotations

from typing import Optional

import gemmi
import numpy as np

# ideal peptide geometry (Å, degrees)
_N_CA, _CA_C, _C_N, _C_O = 1.458, 1.525, 1.329, 1.231
_ANG_N_CA_C, _ANG_CA_C_N, _ANG_C_N_CA, _ANG_CA_C_O = 111.2, 116.2, 121.7, 120.8


def _place(a, b, c, bond, angle_deg, torsion_deg):
    """Atom D bonded to C: |CD| = bond, angle(B, C, D) = angle, torsion(A, B, C, D) = torsion."""
    angle, torsion = np.radians(angle_deg), np.radians(torsion_deg)
    bc = (c - b) / np.linalg.norm(c - b)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array(
        [-bond * np.cos(angle), bond * np.sin(angle) * np.cos(torsion), bond * np.sin(angle) * np.sin(torsion)]
    )
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def build_backbone(phi, psi) -> dict[str, np.ndarray]:
    """N, CA, C, O coordinates (each of shape (n, 3)) of a chain with the given torsions (degrees)."""
    n_res = len(phi)
    N = np.zeros((n_res, 3))
    CA = np.zeros((n_res, 3))
    C = np.zeros((n_res, 3))
    O = np.zeros((n_res, 3))
    N[0] = [0.0, 0.0, 0.0]
    CA[0] = [_N_CA, 0.0, 0.0]
    ang = np.radians(_ANG_N_CA_C)
    C[0] = CA[0] + _CA_C * np.array([-np.cos(ang), np.sin(ang), 0.0])
    for i in range(n_res):
        if i > 0:
            N[i] = _place(N[i - 1], CA[i - 1], C[i - 1], _C_N, _ANG_CA_C_N, psi[i - 1])
            CA[i] = _place(CA[i - 1], C[i - 1], N[i], _N_CA, _ANG_C_N_CA, 180.0)
            C[i] = _place(C[i - 1], N[i], CA[i], _CA_C, _ANG_N_CA_C, phi[i])
        O[i] = _place(N[i], CA[i], C[i], _C_O, _ANG_CA_C_O, psi[i] + 180.0)
    return {"N": N, "CA": CA, "C": C, "O": O}


def protein_like_torsions(n_res: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A helix, a strand and loops: (phi, psi) with realistic spread, so the (phi, psi) covariance is non-singular."""
    rng = np.random.default_rng(seed)
    phi, psi = [], []
    while len(phi) < n_res:
        kind = rng.choice(["helix", "strand", "loop"], p=[0.45, 0.3, 0.25])
        length = {"helix": 12, "strand": 6, "loop": 4}[kind]
        for _ in range(length):
            if kind == "helix":
                phi.append(rng.normal(-60, 6))
                psi.append(rng.normal(-45, 6))
            elif kind == "strand":
                phi.append(rng.normal(-120, 10))
                psi.append(rng.normal(130, 10))
            else:
                phi.append(rng.uniform(-170, -50))
                psi.append(rng.uniform(-60, 170))
    return np.array(phi[:n_res]), np.array(psi[:n_res])


def write_cif(
    path,
    models: list[dict[str, np.ndarray]],
    sequence: Optional[str] = None,
    chain_id: str = "A",
    representative: Optional[int] = None,
) -> str:
    """
    Write a chain as mmCIF, one model per element of `models` (build_backbone output).

    representative : 1-based conformer id written to _pdbx_nmr_representative
                     (what the PDB does for NMR ensembles); None = not written.
    """
    n_res = len(models[0]["CA"])
    sequence = sequence or "A" * n_res
    names = {"A": "ALA", "G": "GLY", "L": "LEU", "V": "VAL", "K": "LYS", "E": "GLU"}
    structure = gemmi.Structure()
    structure.name = "TEST"
    for k, backbone in enumerate(models, start=1):
        model = gemmi.Model(str(k))
        chain = gemmi.Chain(chain_id)
        for i in range(n_res):
            residue = gemmi.Residue()
            residue.name = names.get(sequence[i], "ALA")
            residue.seqid = gemmi.SeqId(i + 1, " ")
            residue.label_seq = i + 1
            residue.entity_type = gemmi.EntityType.Polymer
            for atom_name in ("N", "CA", "C", "O"):
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(atom_name[0])
                atom.pos = gemmi.Position(*backbone[atom_name][i])
                atom.occ = 1.0
                atom.b_iso = 20.0
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)
    structure.setup_entities()
    document = structure.make_mmcif_document()
    if representative is not None:
        document.sole_block().set_pair("_pdbx_nmr_representative.conformer_id", str(representative))
    document.write_file(str(path))
    return str(path)


def rotation_matrix(seed: int) -> np.ndarray:
    """A random proper rotation (QR of a Gaussian matrix, determinant fixed to +1)."""
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def rotate(backbone: dict[str, np.ndarray], rotation: np.ndarray, shift=(0.0, 0.0, 0.0)) -> dict[str, np.ndarray]:
    """Rigidly move a backbone (same internal geometry, different orientation)."""
    return {k: v @ rotation.T + np.asarray(shift) for k, v in backbone.items()}
