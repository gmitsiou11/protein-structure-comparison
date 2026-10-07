"""
Parse mmCIF structure files and extract Cα coordinates, backbone dihedral angles and atomic coordinates.

Every extractor takes an optional ``residue_range = (first, last)``: only the
residues whose number lies in that closed interval are returned.  It mirrors
Machaon's ``PDBHandler.residue_selection``: residues are selected by residue
number, and φ/ψ are still computed with their neighbours in the whole chain.
The truncation test of notebook 03 (sensitivity.truncation_experiment) uses it
to remove residues from the ends of a chain.

Alternate conformations are reduced to one, as Biopython (and so Machaon) does:
for each atom the alternate location with the highest occupancy (the first one
on ties), and for a residue position modelled as two residue types the last one.
"""

import gemmi
import numpy as np
from pathlib import Path

_STRUCTURE_CACHE: dict[str, gemmi.Structure] = {}


def keep_one_conformer(structure: gemmi.Structure) -> gemmi.Structure:
    """
    Reduce alternate conformations in place, with Biopython's rules.

    Atoms: of the alternate locations of one atom name, keep the one with the
    highest occupancy; on ties the first in the file (Bio.PDB DisorderedAtom).
    Residues: when consecutive residues share a residue number (two residue
    types at one position), keep the last (Bio.PDB DisorderedResidue).
    """
    for model in structure:
        for chain in model:
            for i in range(len(chain) - 2, -1, -1):
                if chain[i].seqid == chain[i + 1].seqid:
                    del chain[i]
            for residue in chain:
                best: dict[str, int] = {}
                drop: list[int] = []
                for k, atom in enumerate(residue):
                    if atom.altloc == "\0":
                        continue
                    j = best.get(atom.name)
                    if j is None:
                        best[atom.name] = k
                    elif atom.occ > residue[j].occ:
                        drop.append(j)
                        best[atom.name] = k
                    else:
                        drop.append(k)
                for k in sorted(drop, reverse=True):
                    del residue[k]
    return structure


def load_structure(mmcif_path: str) -> gemmi.Structure:
    """Load a mmCIF file with one conformer per atom (cached: each file is read once)."""
    if mmcif_path not in _STRUCTURE_CACHE:
        structure = keep_one_conformer(gemmi.read_structure(mmcif_path))
        structure.setup_entities()  # needed by chain.get_polymer()
        _STRUCTURE_CACHE[mmcif_path] = structure
    return _STRUCTURE_CACHE[mmcif_path]


def clear_structure_cache() -> None:
    """Clear the cache."""
    _STRUCTURE_CACHE.clear()


def _one_letter(residue_name: str) -> str:
    """
    One-letter code of an amino acid, upper case; "X" if unknown.

    gemmi returns lower case for modified residues (MSE -> "m") and a space
    for non-amino acids, so strip + upper-case and fall back to "X".
    """
    info = gemmi.find_tabulated_residue(residue_name)
    if info is None or not info.is_amino_acid():
        return "X"
    return (info.one_letter_code or "").strip().upper() or "X"


def _in_range(residue: gemmi.Residue, residue_range: tuple[int, int] | None) -> bool:
    """True if no range is given or the residue number lies in [first, last]."""
    if residue_range is None:
        return True
    first, last = residue_range
    return first <= residue.seqid.num <= last


def get_ca_coords_and_seq(
    structure: gemmi.Structure,
    model_idx: int = 0,
    chain_id: str = None,
    residue_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, str]:
    """
    Extract Cα coordinates and the one-letter amino acid sequence.

    Parameters
    ----------
    structure     : gemmi.Structure
    model_idx     : which model to use (relevant for NMR ensembles)
    chain_id      : chain to extract; if None, uses the first chain in the model
    residue_range : (first, last) residue numbers to keep; None = whole chain

    Returns
    -------
    coords : ndarray of shape (N, 3)
    seq    : one-letter amino acid string of length N (used for the sequence alignment)
    """
    coords, seq = [], []
    model = structure[model_idx]
    chains = [model[chain_id]] if chain_id else [model[0]]

    for chain in chains:
        for residue in chain.get_polymer():
            if not _in_range(residue, residue_range):
                continue
            for atom in residue:
                if atom.name == "CA":
                    pos = atom.pos
                    coords.append([pos.x, pos.y, pos.z])
                    seq.append(_one_letter(residue.name))
                    break

    return np.array(coords), "".join(seq)


def get_phi_psi_with_index(
    structure: gemmi.Structure,
    model_idx: int = 0,
    chain_id: str = None,
    residue_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, list]:
    """
    Compute backbone phi/psi angles and return them with the corresponding
    Cα position indices from get_ca_coords_and_seq.

    Parameters
    ----------
    structure     : gemmi.Structure
    model_idx     : model index (0-based)
    chain_id      : chain to process; if None, uses the first chain
    residue_range : (first, last) residue numbers to keep; None = whole chain.
                    Angles are computed with the neighbours in the whole chain
                    (as Machaon does), then only residues in the range are kept.

    Returns
    -------
    angles : ndarray of shape (M, 2) with [phi, psi] in degrees for residues
             where both angles are well-defined (terminal residues are excluded)
    ca_indices : list of length M; ca_indices[k] is the 0-based index of that
                 residue in the Cα array returned by get_ca_coords_and_seq
                 (called with the same residue_range)
    """
    model = structure[model_idx]
    chains = [model[chain_id]] if chain_id else [model[0]]

    angles, ca_indices = [], []
    ca_counter = 0

    for chain in chains:
        residues = list(chain.get_polymer())
        for i, residue in enumerate(residues):
            if not _in_range(residue, residue_range):
                continue
            if not any(atom.name == "CA" for atom in residue):
                continue

            local_idx = ca_counter
            ca_counter += 1

            prev_res = residues[i - 1] if i > 0 else None
            next_res = residues[i + 1] if i < len(residues) - 1 else None
            result = gemmi.calculate_phi_psi(prev_res, residue, next_res)
            # gemmi returns radians; Machaon (and every doc here) uses degrees
            phi, psi = np.degrees(result[0]), np.degrees(result[1])

            if not (np.isnan(phi) or np.isnan(psi)):
                angles.append([phi, psi])
                ca_indices.append(local_idx)

    if angles:
        return np.array(angles), ca_indices
    return np.empty((0, 2)), ca_indices


def get_all_atom_coords(
    structure: gemmi.Structure,
    model_idx: int = 0,
    chain_id: str = None,
    residue_range: tuple[int, int] | None = None,
    include_hydrogens: bool = False,
) -> np.ndarray:
    """
    Extract all heavy-atom (non-hydrogen, non-deuterium) coordinates from a chain.

    Used by the t-alpha metric, which builds an alpha-shape triangulated surface
    from the all-atom point cloud of each structure.  Hydrogen atoms are excluded
    because they are absent from most experimental mmCIF files and from AlphaFold /
    OpenFold predictions, so excluding them gives consistent atom counts across all
    structure types.

    Parameters
    ----------
    structure     : gemmi.Structure
    model_idx     : model index (0 for X-ray / Cryo-EM / AF; NMR ensemble index otherwise)
    chain_id      : chain to extract; if None, uses the first chain in the model
    residue_range : (first, last) residue numbers to keep; None = whole chain
    include_hydrogens : False (default, the pipeline's choice) drops H and D;
                    True keeps every atom of the polymer residues, which is what
                    Machaon's load_points does (used only for the sensitivity check).

    Returns
    -------
    ndarray of shape (N_atoms, 3), or shape (0, 3) if no atoms found.
    """
    coords: list[list[float]] = []
    model = structure[model_idx]
    chains = [model[chain_id]] if chain_id else [model[0]]
    _H = gemmi.Element("H")
    _D = gemmi.Element("D")
    for chain in chains:
        for residue in chain.get_polymer():
            if not _in_range(residue, residue_range):
                continue
            for atom in residue:
                if include_hydrogens or atom.element not in (_H, _D):
                    pos = atom.pos
                    coords.append([pos.x, pos.y, pos.z])
    return np.array(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)
