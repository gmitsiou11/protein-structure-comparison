"""
Which model of an NMR ensemble stands for the entry.

An NMR entry deposits an ensemble (typically 10–20 models, sometimes 100+), and
model 0 is an arbitrary member of it.  The experimental baseline of the thesis
is dominated by NMR-to-other-method distances, so the model is chosen by a rule
(``choose_nmr_model``):

  "medoid"   (default) the most central model: the one with the smallest mean
             Cα RMSD to all the others after superposition; a member of the
             ensemble, chosen by the data.
  "pdb"      the model the depositors designate as representative
             (mmCIF item _pdbx_nmr_representative.conformer_id); the medoid
             when the entry does not say.
  "first"    model 0, the model Machaon reads (it uses the first model of a file).

Only numpy and gemmi are needed (no open3d), so notebooks that do not import
the metrics can use this module.
"""

from __future__ import annotations

from typing import Optional

import gemmi
import numpy as np

from src.parser import load_structure, get_ca_coords_and_seq

POLICIES = ("medoid", "pdb", "first")
MAX_MODELS_FOR_MEDOID = 100  # larger ensembles are subsampled evenly (O(n²) superpositions)


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD of two (N, 3) point sets after the optimal rigid superposition (Kabsch)."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, s, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(np.mean(np.sum((a - b @ rotation.T) ** 2, axis=1))))


def pdb_representative_model(path: str) -> Optional[int]:
    """
    0-based index of the model that the entry designates as representative,
    or None if the file does not say.

    Reads _pdbx_nmr_representative.conformer_id (the conformer's model number),
    or the older _pdbx_nmr_ensemble.representative_conformer.
    """
    try:
        block = gemmi.cif.read(path).sole_block()
    except Exception:
        return None
    structure = load_structure(path)
    names = [str(model.num) for model in structure]
    for tag in (
        "_pdbx_nmr_representative.conformer_id",
        "_pdbx_nmr_ensemble.representative_conformer",
    ):
        value = block.find_value(tag)
        if value is None or value in ("?", "."):
            continue
        value = str(value).strip().strip("'\"")
        if value in names:  # model names are the model numbers ("1", "2", ...)
            return names.index(value)
        if value.isdigit() and 1 <= int(value) <= len(structure):
            return int(value) - 1
    return None


def ensemble_medoid(path: str, chain_id: str) -> tuple[int, float]:
    """
    (index of the medoid model, its mean RMSD to the other models).

    RMSD over the Cα atoms of `chain_id`.  Models whose Cα count differs from
    the most common count are ignored.  Ties go to the lowest model index.
    """
    structure = load_structure(path)
    n_models = len(structure)
    if n_models < 2:
        return 0, 0.0
    indices = np.arange(n_models)
    if n_models > MAX_MODELS_FOR_MEDOID:
        indices = np.unique(np.linspace(0, n_models - 1, MAX_MODELS_FOR_MEDOID).round().astype(int))
    coords = {int(i): get_ca_coords_and_seq(structure, int(i), chain_id)[0] for i in indices}
    sizes = [len(c) for c in coords.values()]
    modal = max(set(sizes), key=sizes.count)
    usable = [i for i, c in coords.items() if len(c) == modal]
    if len(usable) < 2:
        return int(usable[0]) if usable else 0, 0.0
    m = len(usable)
    rmsd = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1, m):
            rmsd[a, b] = rmsd[b, a] = kabsch_rmsd(coords[usable[a]], coords[usable[b]])
    mean_rmsd = rmsd.sum(axis=1) / (m - 1)
    best = int(np.argmin(mean_rmsd))  # first minimum: lowest index
    return int(usable[best]), float(mean_rmsd[best])


def choose_nmr_model(path: str, chain_id: str, policy: str = "medoid") -> dict:
    """
    Apply a selection policy to one NMR entry.

    Returns a dict with: model (0-based index to use), policy (requested),
    rule (the rule that actually decided: "medoid", "pdb", "first",
    "single model" or "pdb->medoid" when the entry names no representative),
    n_models, pdb_representative (index or None), medoid (index or None),
    medoid_mean_rmsd.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown NMR model policy {policy!r}; use one of {POLICIES}")
    n_models = len(load_structure(path))
    info = {
        "model": 0,
        "policy": policy,
        "rule": policy,
        "n_models": n_models,
        "pdb_representative": None,
        "medoid": None,
        "medoid_mean_rmsd": None,
    }
    if n_models == 1:
        info["rule"] = "single model"
        return info
    info["pdb_representative"] = pdb_representative_model(path)
    if policy == "first":
        return info
    if policy == "pdb" and info["pdb_representative"] is not None:
        info["model"] = info["pdb_representative"]
        return info
    medoid, mean_rmsd = ensemble_medoid(path, chain_id)
    info.update(model=medoid, medoid=medoid, medoid_mean_rmsd=round(mean_rmsd, 4))
    if policy == "pdb":
        info["rule"] = "pdb->medoid"
    return info
