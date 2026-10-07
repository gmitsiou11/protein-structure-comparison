"""
Sensitivity checks of the Machaon metrics: what they do when the input changes
in ways that should (or should not) matter.  None of this changes the pipeline;
it measures it, so that a distance between two structures of the same protein
can be read against what the metric cannot resolve.

1. Orientation of t-alpha.  Machaon scales each coordinate axis to [0, 1]
   before building the alpha shape, so the bounding box, and with it the
   triangle count, depends on how the structure is oriented in space.  The same
   chain, only rotated, gets t-alpha > 0 against itself.  ``rotation_test`` and
   ``orientation_floor`` measure it; the principal-axes variant (``frame="pca"``:
   rotate the atoms onto their principal axes before scaling) removes the
   dependence, and ``t_alpha_pca_table`` recomputes every comparison in that
   frame.  The pipeline keeps Machaon's own definition (frame="raw").

2. Truncation.  Cut a growing share of residues from the two ends of a chain
   and compare the truncated chain with the whole chain.  Because the metrics
   compare summaries, no residue pairing is needed; the question is how fast
   the distance grows (``truncation_experiment``).

3. Hydrogens.  The pipeline builds the alpha shape from heavy atoms; Machaon
   uses every atom in the file.  ``hydrogen_sensitivity`` computes t-alpha both
   ways on a pair of structures.

All functions take the paths and chain identifiers used elsewhere in the
repository and work on one chain of one model.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.metrics import (
    _log_n_triangles,
    _t_alpha,
    _w_rdist,
    _b_phipsi,
    summary_from_arrays,
    summarise_structure,
)
from src.parser import (
    get_all_atom_coords,
    get_ca_coords_and_seq,
    get_phi_psi_with_index,
    load_structure,
)

FRAMES = ("raw", "pca")


# ── 1. Orientation of t-alpha ───────────────────────────────────────────────


def pca_frame(points: np.ndarray) -> np.ndarray:
    """Points centred and rotated onto their principal axes (largest variance first)."""
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt.T


def log_triangles(points: np.ndarray, frame: str = "raw") -> Optional[float]:
    """log(number of alpha-shape triangles) of a point cloud; frame="pca" rotates it onto its principal axes first."""
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}")
    pts = pca_frame(points) if frame == "pca" else points
    return _log_n_triangles(pts)[0]


def t_alpha_points(points_a: np.ndarray, points_b: np.ndarray, frame: str = "raw") -> Optional[float]:
    """t-alpha (exp|Δ log n_triangles| − 1) between two atom clouds, in the chosen frame."""
    return _t_alpha(log_triangles(points_a, frame), log_triangles(points_b, frame))


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random proper rotation matrix."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def rotation_test(
    points: np.ndarray, n_rotations: int = 20, seed: int = 0, frames: Iterable[str] = FRAMES
) -> pd.DataFrame:
    """
    t-alpha between a structure and randomly rotated copies of itself.  The
    true value is 0; anything else is orientation dependence.

    Returns one row per rotation with a column per frame ("raw", "pca").
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_rotations):
        rotated = points @ random_rotation(rng).T
        rows.append({"rotation": k, **{f: t_alpha_points(points, rotated, f) for f in frames}})
    return pd.DataFrame(rows)


def orientation_report(
    path: str, chain_id: str, model_idx: int = 0, n_rotations: int = 20, seed: int = 0
) -> dict:
    """Median and maximum t-alpha of a chain against its rotated copies, raw and PCA frame."""
    atoms = get_all_atom_coords(load_structure(path), model_idx, chain_id)
    table = rotation_test(atoms, n_rotations, seed)
    return {
        "path": path,
        "chain": chain_id,
        "n_atoms": len(atoms),
        "raw_median": float(table["raw"].median()),
        "raw_max": float(table["raw"].max()),
        "pca_median": float(table["pca"].median()),
        "pca_max": float(table["pca"].max()),
    }


# ── 2. Truncation ───────────────────────────────────────────────────────────


def _residue_numbers(path: str, model_idx: int, chain_id: str) -> list[int]:
    """Residue numbers of the polymer residues that have a Cα, in chain order."""
    chain = load_structure(path)[model_idx][chain_id]
    return [r.seqid.num for r in chain.get_polymer() if any(a.name == "CA" for a in r)]


def truncation_experiment(
    path: str,
    chain_id: str,
    model_idx: int = 0,
    fractions: Iterable[float] = (0.0, 0.05, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """
    Compare the whole chain with the same chain after removing a fraction of its
    residues, half from each end (the removed residues are simply absent: no
    alignment, no residue pairing).

    Returns one row per fraction: n_residues kept, w_rdist_norm, b_phipsi,
    t_alpha, t_alpha_pca.  At fraction 0 every distance is 0.
    """
    numbers = _residue_numbers(path, model_idx, chain_id)
    n = len(numbers)
    whole = summarise_structure(path, model_idx, chain_id)
    whole_atoms = get_all_atom_coords(load_structure(path), model_idx, chain_id)
    rows = []
    for fraction in fractions:
        cut = int(round(n * fraction))
        head = cut // 2
        tail = cut - head
        kept = numbers[head : n - tail]
        if len(kept) < 10:
            continue
        rng = (kept[0], kept[-1])
        part = summarise_structure(path, model_idx, chain_id, rng)
        part_atoms = get_all_atom_coords(load_structure(path), model_idx, chain_id, rng)
        w = _w_rdist(whole.dists, part.dists)
        rows.append(
            {
                "fraction_removed": fraction,
                "n_residues": len(kept),
                "w_rdist_norm": w["norm"],
                "b_phipsi": _b_phipsi(whole, part),
                "t_alpha": _t_alpha(whole.log_tri, part.log_tri),
                "t_alpha_pca": t_alpha_points(whole_atoms, part_atoms, "pca"),
            }
        )
    return pd.DataFrame(rows)


# ── 3. Hydrogens ────────────────────────────────────────────────────────────


def hydrogen_sensitivity(
    path_a: str,
    chain_a: str,
    path_b: str,
    chain_b: str,
    model_a: int = 0,
    model_b: int = 0,
) -> dict:
    """
    t-alpha between two chains from heavy atoms only (this pipeline) and from
    every atom in the file (Machaon).  The two agree when neither file has
    hydrogens; NMR entries usually do.
    """
    out = {}
    for label, include in (("heavy", False), ("all_atoms", True)):
        a = get_all_atom_coords(load_structure(path_a), model_a, chain_a, include_hydrogens=include)
        b = get_all_atom_coords(load_structure(path_b), model_b, chain_b, include_hydrogens=include)
        out[f"n_atoms_a_{label}"] = len(a)
        out[f"n_atoms_b_{label}"] = len(b)
        out[f"t_alpha_{label}"] = t_alpha_points(a, b, "raw")
    return out


# ── 4. Orientation floor over many chains ───────────────────────────────────

_MANIFEST_COLUMNS = {
    "X-ray": ("pdb_xray", "chain_xray"),
    "NMR": ("pdb_nmr", "chain_nmr"),
    "Cryo-EM": ("pdb_cryoem", "chain_cryoem"),
}


def sample_chains(
    manifest: pd.DataFrame, data_dir: str = "data", n_per_method: int = 15, seed: int = 0
) -> pd.DataFrame:
    """
    A reproducible random sample of the dataset's experimental chains, up to
    n_per_method per method, only those whose file is on disk.
    Columns: protein_name, method, pdb_id, chain, model, path.
    """
    import os

    rng = np.random.default_rng(seed)
    rows = []
    for method, (pdb_col, chain_col) in _MANIFEST_COLUMNS.items():
        candidates = []
        for _, row in manifest.iterrows():
            pdb_id = str(row.get(pdb_col, "") or "").strip().upper()
            chain = str(row.get(chain_col, "") or "").strip()
            path = os.path.join(data_dir, "experimental", f"{pdb_id}.cif")
            if pdb_id and chain and os.path.exists(path):
                candidates.append((row["protein_name"], method, pdb_id, chain, 0, path))
        if not candidates:
            continue
        take = rng.choice(len(candidates), size=min(n_per_method, len(candidates)), replace=False)
        rows += [candidates[i] for i in sorted(take)]
    return pd.DataFrame(rows, columns=["protein_name", "method", "pdb_id", "chain", "model", "path"])


def orientation_floor(chains: pd.DataFrame, n_rotations: int = 20, seed: int = 0) -> pd.DataFrame:
    """
    orientation_report() for every chain of sample_chains(): t-alpha of the
    chain against rotated copies of itself, in Machaon's frame ("raw") and in
    the principal-axes frame ("pca").  The raw values are the noise floor of
    t-alpha: a t-alpha between two structures that is not larger than this is
    not evidence of a different shape.  Chains that fail carry the error.
    """
    rows = []
    for _, c in chains.iterrows():
        base = {"protein_name": c["protein_name"], "method": c["method"], "pdb_id": c["pdb_id"], "chain": c["chain"]}
        try:
            report = orientation_report(c["path"], c["chain"], int(c["model"]), n_rotations, seed)
            rows.append({**base, **{k: v for k, v in report.items() if k not in ("path", "chain")}, "error": None})
        except Exception as exc:  # unreadable chain: keep the row, report why
            rows.append({**base, "error": str(exc)})
    return pd.DataFrame(rows)


# ── 5. t-alpha of every comparison in the principal-axes frame ──────────────


def structure_path(record: dict, side: str, manifest_row: Optional[pd.Series], data_dir: str = "data") -> Optional[str]:
    """
    File of one side ("a" or "b") of a comparison record.  Experimental
    structures: the record's pdb_id, or the manifest entry of that method when
    the record has none (NMR ensemble records); predictions: AF2_/OF3_<UniProt>.
    """
    import os

    method = record.get(f"method_{side}")
    uniprot = record.get("uniprot_id")
    if method == "AF2":
        return os.path.join(data_dir, "alphafold2", f"AF2_{uniprot}.cif")
    if method == "OF3":
        return os.path.join(data_dir, "openfold3", f"OF3_{uniprot}.cif")
    pdb_id = str(record.get(f"pdb_id_{side}") or "").strip()
    if not pdb_id and manifest_row is not None and method in _MANIFEST_COLUMNS:
        pdb_id = str(manifest_row.get(_MANIFEST_COLUMNS[method][0], "") or "").strip()
    if not pdb_id:
        return None
    return os.path.join(data_dir, "experimental", f"{pdb_id.upper()}.cif")


def t_alpha_pca_table(
    records: list[dict], manifest: pd.DataFrame, data_dir: str = "data"
) -> pd.DataFrame:
    """
    t-alpha recomputed for every comparison record with each structure rotated
    onto its principal axes before Machaon's per-axis scaling (frame "pca"),
    next to the record's own t-alpha (Machaon's frame).  Each structure (file,
    model, chain) is summarised once.

    Columns: protein_name, comparison, category, rmsd, copy_method (copy
    records only), t_alpha, t_alpha_pca, status ("ok", or why the value is
    missing).
    """
    import os

    by_name = manifest.set_index("protein_name") if len(manifest) else manifest
    cache: dict = {}

    def log_tri(path, model, chain):
        key = (path, model, chain)
        if key not in cache:
            atoms = get_all_atom_coords(load_structure(path), model, chain)
            cache[key] = log_triangles(atoms, "pca") if len(atoms) >= 4 else None
        return cache[key]

    rows = []
    for r in records:
        if r.get("status", "ok") != "ok":
            continue
        row_info = {
            "protein_name": r.get("protein_name"),
            "comparison": r.get("comparison"),
            "category": r.get("category"),
            "rmsd": r.get("rmsd"),
            "copy_method": r.get("copy_method"),
            "t_alpha": r.get("t_alpha"),
            "t_alpha_pca": None,
        }
        manifest_row = by_name.loc[r["protein_name"]] if r.get("protein_name") in by_name.index else None
        try:
            values = []
            for side in ("a", "b"):
                path = structure_path(r, side, manifest_row, data_dir)
                if path is None or not os.path.exists(path):
                    raise FileNotFoundError(f"{path} not found")
                chain = r.get(f"chain_id_{side}") or r.get("chain_id") or "A"
                model = int(r.get(f"model_idx_{side}") or 0)
                values.append(log_tri(path, model, chain))
            row_info["t_alpha_pca"] = _t_alpha(*values)
            row_info["status"] = "ok" if row_info["t_alpha_pca"] is not None else "no triangles"
        except Exception as exc:  # missing file or unreadable chain
            row_info["status"] = str(exc)
        rows.append(row_info)
    return pd.DataFrame(rows)


def t_alpha_pca_levels(pca_records: pd.DataFrame, pca_copies: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    The reference levels of baselines.baseline_levels for t-alpha in the
    principal-axes frame: long table (level, protein_name, value), one value per
    protein and level (median over its pairs).

    pca_records : t_alpha_pca_table() of the pipeline records (with the NMR
                  ensemble records): levels "NMR ensemble", "between methods",
                  "AF2", "OF3".
    pca_copies  : t_alpha_pca_table() of the entry-copy records: level "same
                  entry"; copies identical by imposed symmetry (RMSD below
                  baselines.SYMMETRY_RMSD) are left out, and a protein with
                  copies in several entries gets the median over its entries.
    """
    from src.baselines import SYMMETRY_RMSD

    level_of = {
        "nmr_intra_ensemble": "NMR ensemble",
        "exp_vs_exp": "between methods",
        "exp_vs_af2": "AF2",
        "exp_vs_of3": "OF3",
    }
    parts = []
    if pca_copies is not None and len(pca_copies):
        c = pca_copies[pca_copies["status"] == "ok"].copy()
        c = c[pd.to_numeric(c["rmsd"], errors="coerce") >= SYMMETRY_RMSD]
        per_entry = c.groupby(["protein_name", "copy_method"])["t_alpha_pca"].median()
        same = per_entry.groupby("protein_name").median()
        parts.append(pd.DataFrame({"level": "same entry", "protein_name": same.index, "value": same.values}))
    r = pca_records[(pca_records["status"] == "ok") & pca_records["category"].isin(level_of)].copy()
    r["level"] = r["category"].map(level_of)
    per = r.groupby(["level", "protein_name"])["t_alpha_pca"].median().reset_index()
    parts.append(per.rename(columns={"t_alpha_pca": "value"}))
    out = pd.concat(parts, ignore_index=True)
    order = ["same entry", "NMR ensemble", "between methods", "AF2", "OF3"]
    out["level"] = pd.Categorical(out["level"], categories=order, ordered=True)
    return out.sort_values(["level", "protein_name"]).reset_index(drop=True)
