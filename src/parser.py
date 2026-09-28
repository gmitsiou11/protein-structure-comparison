"""
Parse mmCIF structure files and extract Cα coordinates, backbone dihedral angles and atomic coordinates.

"""

import gemmi
import numpy as np
from pathlib import Path

_STRUCTURE_CACHE: dict[str, gemmi.Structure] = {}


def load_structure(mmcif_path: str) -> gemmi.Structure:
    """Load a mmCIF file, using a module-level cache to avoid repeated disk reads."""
    if mmcif_path not in _STRUCTURE_CACHE:
        _STRUCTURE_CACHE[mmcif_path] = gemmi.read_structure(mmcif_path)
    return _STRUCTURE_CACHE[mmcif_path]


def clear_structure_cache() -> None:
    """Clear the cache."""
    _STRUCTURE_CACHE.clear()


def get_ca_coords_and_seq(
    structure: gemmi.Structure,
    model_idx: int = 0,
    chain_id: str = None,
) -> tuple[np.ndarray, str]:
    """
    Extract Cα coordinates and the one-letter amino acid sequence.

    Parameters
    ----------
    structure : gemmi.Structure
    model_idx : which model to use (relevant for NMR ensembles)
    chain_id  : chain to extract; if None, uses the first chain in the model

    Returns
    -------
    coords : ndarray of shape (N, 3)
    seq    : one-letter amino acid string of length N (one-letter string is important for next step validation)
    """
    coords, seq = [], []
    model = structure[model_idx]
    chains = [model[chain_id]] if chain_id else [model[0]]

    for chain in chains:
        for residue in chain:
            for atom in residue:
                if atom.name == "CA":
                    pos = atom.pos
                    coords.append([pos.x, pos.y, pos.z])
                    info = gemmi.find_tabulated_residue(residue.name)
                    seq.append(
                        info.one_letter_code if info and info.one_letter_code else "X"
                    )
                    break

    return np.array(coords), "".join(seq)


def get_phi_psi_with_index(
    structure: gemmi.Structure,
    model_idx: int = 0,
    chain_id: str = None,
) -> tuple[np.ndarray, list]:
    """
    Compute backbone phi/psi angles and return them with the corresponding
    Cα position indices from get_ca_coords_and_seq.

    Parameters
    ----------
    structure : gemmi.Structure
    model_idx : model index (0-based)
    chain_id  : chain to process; if None, uses the first chain

    Returns
    -------
    angles : ndarray of shape (M, 2) with [phi, psi] in degrees for residues
             where both angles are well-defined (terminal residues are excluded)
    ca_indices : list of length M; ca_indices[k] is the 0-based index of that
                 residue in the Cα array returned by get_ca_coords_and_seq
    """
    model = structure[model_idx]
    chains = [model[chain_id]] if chain_id else [model[0]]

    angles, ca_indices = [], []
    ca_counter = 0

    for chain in chains:
        residues = list(chain)
        for i, residue in enumerate(residues):
            if not any(atom.name == "CA" for atom in residue):
                continue

            local_idx = ca_counter
            ca_counter += 1

            prev_res = residues[i - 1] if i > 0 else None
            next_res = residues[i + 1] if i < len(residues) - 1 else None
            result = gemmi.calculate_phi_psi(prev_res, residue, next_res)
            phi, psi = result[0], result[1]

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
    ca_index_filter: "set[int] | None" = None,
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
    structure : gemmi.Structure
    model_idx : model index (0 for X-ray / Cryo-EM / AF; NMR ensemble index otherwise)
    chain_id  : chain to extract; if None, uses the first chain in the model
    ca_index_filter : optional set of 0-based Cα indices as returned by
        get_ca_coords_and_seq.  When provided, only atoms belonging to residues
        whose Cα index is in this set are returned.  This restricts t-alpha to
        the sequence-aligned residues, making its structural scope consistent with
        RMSD, w-rdist, and b-phipsi.  When None (default), all heavy atoms from
        all residues are returned (backward-compatible behaviour).

    Returns
    -------
    ndarray of shape (N_atoms, 3), or shape (0, 3) if no atoms found.

    Notes
    -----
    The Cα counter increments only for residues that contain a CA atom, mirroring
    the counting logic in get_ca_coords_and_seq so that ca_index_filter indices
    correspond correctly between the two functions.
    """
    coords: list[list[float]] = []
    model = structure[model_idx]
    chains = [model[chain_id]] if chain_id else [model[0]]
    _H = gemmi.Element("H")
    _D = gemmi.Element("D")
    ca_counter = 0
    for chain in chains:
        for residue in chain:
            has_ca = any(atom.name == "CA" for atom in residue)
            if has_ca:
                ca_idx: "int | None" = ca_counter
                ca_counter += 1
            else:
                ca_idx = None

            # When a filter is active, skip residues not in the aligned set
            # (and always skip residues without a CA atom, which have no index).
            if ca_index_filter is not None:
                if ca_idx is None or ca_idx not in ca_index_filter:
                    continue

            for atom in residue:
                if atom.element not in (_H, _D):
                    pos = atom.pos
                    coords.append([pos.x, pos.y, pos.z])
    return np.array(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)
