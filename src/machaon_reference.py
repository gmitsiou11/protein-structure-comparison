"""
A straight port of the three Machaon feature extractions, for cross-checking.

Machaon implements its three metrics in three places of its repository:
b-phipsi (bhattacharyyadistance.py), w-rdist (scanner.py) and t-alpha
(pdbhandler.py).  src/metrics.py is a port of that code, restructured (one
summary per structure, gemmi instead of Bio.PDB).  This module goes the other
way: it reproduces Machaon's own procedure as literally as possible, with
Biopython, so that ``compare_with_pipeline`` can show numerically that the
pipeline computes the same thing.

What is copied from Machaon (pdbhandler.py / scanner.py):
  angles     Polypeptide(chain).get_phi_psi_list(); radians -> degrees; the
             pairs where both angles exist; mean, 2x2 covariance, determinant;
             a chain with det <= 0 has no b-phipsi.
  distances  Euclidean distance between the Cα atoms of every pair of residues
             i < j of the chain.
  triangles  every atom of the chain's residues, each coordinate axis scaled
             to [0, 1] (sklearn MinMaxScaler), Open3D alpha shape with
             alpha = 0.085, log of the number of triangles.
  metrics    b-phipsi: Bhattacharyya distance of the two Gaussians;
             w-rdist: log10(Wasserstein-1 + 1); t-alpha: exp(|Δ log n|) − 1.

The one deliberate difference: only standard amino-acid residues of the chain
are used (hetero residues such as a calcium ion named "CA", or water, are
skipped).  Machaon reads PDB files from its own curated dataset; on a raw
mmCIF chain they would otherwise enter the distances and the alpha shape.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, is_aa
from Bio.PDB.Polypeptide import Polypeptide
from scipy.linalg import det, inv
from scipy.stats import wasserstein_distance

ALPHA = 0.085


def _polymer_residues(path: str, chain_id: str, model_idx: int = 0) -> list:
    structure = MMCIFParser(QUIET=True).get_structure("ref", path)
    model = list(structure)[model_idx]
    chain = model[chain_id]
    return [r for r in chain if r.id[0] == " " and is_aa(r, standard=False)]


def count_chain_breaks(path: str, chain_id: str, model_idx: int = 0, max_peptide_bond: float = 2.0) -> int:
    """
    Number of places where consecutive residues of the chain are not bonded
    (C–N distance above max_peptide_bond, i.e. residues missing from the model).

    Reported next to a cross-check because a chain with breaks is where two
    φ/ψ implementations are most likely to disagree (how each treats the
    residues on either side of a gap).  On a synthetic chain with a gap the
    reference and the pipeline agree; if they do not on a real chain, look here first.
    """
    residues = _polymer_residues(path, chain_id, model_idx)
    breaks = 0
    for prev, nxt in zip(residues, residues[1:]):
        if "C" in prev and "N" in nxt and np.linalg.norm(prev["C"].coord - nxt["N"].coord) > max_peptide_bond:
            breaks += 1
    return breaks


def reference_features(path: str, chain_id: str, model_idx: int = 0) -> dict:
    """Machaon's three features of one chain: angles, distances, log triangle count."""
    residues = _polymer_residues(path, chain_id, model_idx)

    # angles: pdbhandler.fetch_angles
    poly = Polypeptide(residues)
    pairs = []
    for phi, psi in poly.get_phi_psi_list():
        if phi is not None and psi is not None:
            pairs.append([math.degrees(phi), math.degrees(psi)])
    angles = None
    if len(pairs) > 2:
        data = np.array(pairs)
        cov = np.cov(data.T)
        determinant = det(cov)
        if determinant > 0:
            angles = {"mean": data.mean(axis=0), "cov": cov, "det": determinant}

    # distances: pdbhandler.calculate_residue_distances
    ca = [r["CA"].coord for r in residues if "CA" in r]
    distances = np.array(
        [np.linalg.norm(ca[i] - ca[j]) for i in range(len(ca)) for j in range(i + 1, len(ca))],
        dtype=float,
    )

    # triangles: pdbhandler.load_points + get_mesh_triangles
    log_tri, n_tri = None, 0
    points = np.array([atom.get_coord() for r in residues for atom in r], dtype=float)
    if len(points) > 3:
        import open3d as o3d

        span = points.max(axis=0) - points.min(axis=0)
        span[span == 0] = 1.0
        scaled = (points - points.min(axis=0)) / span  # sklearn MinMaxScaler
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(scaled)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, ALPHA)
        n_tri = len(mesh.triangles)
        log_tri = float(np.log(n_tri)) if n_tri > 0 else None

    return {"angles": angles, "distances": distances, "log_tri": log_tri, "n_tri": n_tri}


def reference_metrics(features_a: dict, features_b: dict) -> dict:
    """Machaon's metrics from two reference_features() results (None where Machaon gives no value)."""
    out = {"b_phipsi": None, "w_rdist_norm": None, "t_alpha": None}
    a, b = features_a["angles"], features_b["angles"]
    if a is not None and b is not None:  # BhattacharyyaDistance.multivariate_compare_group
        cov = (a["cov"] + b["cov"]) / 2
        diff = a["mean"] - b["mean"]
        out["b_phipsi"] = float(
            0.125 * diff @ inv(cov) @ diff + 0.5 * np.log(det(cov) / np.sqrt(a["det"] * b["det"]))
        )
    if len(features_a["distances"]) and len(features_b["distances"]):
        out["w_rdist_norm"] = math.log10(
            wasserstein_distance(features_a["distances"], features_b["distances"]) + 1
        )
    if features_a["log_tri"] is not None and features_b["log_tri"] is not None:
        out["t_alpha"] = float(np.exp(abs(features_a["log_tri"] - features_b["log_tri"])) - 1)
    return out


def compare_with_pipeline(
    path_a: str,
    chain_a: str,
    path_b: str,
    chain_b: str,
    model_a: int = 0,
    model_b: int = 0,
) -> pd.DataFrame:
    """
    The three Machaon metrics of a pair, from the reference procedure and from
    src/metrics.py, with their absolute difference.

    Rows: b_phipsi, w_rdist_norm, t_alpha (heavy atoms: the pipeline's choice),
    t_alpha (all atoms: Machaon's choice, same atoms as the reference).
    The first two and the last row should agree to rounding (|diff| < 1e-6);
    the heavy-atom t-alpha differs only if the files contain hydrogens.
    """
    from src.metrics import _b_phipsi, _t_alpha, _w_rdist, summarise_structure, summary_from_arrays
    from src.parser import (
        get_all_atom_coords,
        get_ca_coords_and_seq,
        get_phi_psi_with_index,
        load_structure,
    )

    ref = reference_metrics(
        reference_features(path_a, chain_a, model_a), reference_features(path_b, chain_b, model_b)
    )
    sa = summarise_structure(path_a, model_a, chain_a)
    sb = summarise_structure(path_b, model_b, chain_b)

    def with_hydrogens(path, chain, model):
        structure = load_structure(path)
        ca, seq = get_ca_coords_and_seq(structure, model, chain)
        angles, _ = get_phi_psi_with_index(structure, model, chain)
        atoms = get_all_atom_coords(structure, model, chain, include_hydrogens=True)
        return summary_from_arrays(ca, seq, angles, atoms)

    ha, hb = with_hydrogens(path_a, chain_a, model_a), with_hydrogens(path_b, chain_b, model_b)
    ours = {
        "b_phipsi": _b_phipsi(sa, sb),
        "w_rdist_norm": _w_rdist(sa.dists, sb.dists)["norm"],
        "t_alpha (heavy atoms, pipeline)": _t_alpha(sa.log_tri, sb.log_tri),
        "t_alpha (all atoms, Machaon)": _t_alpha(ha.log_tri, hb.log_tri),
    }
    reference = {
        "b_phipsi": ref["b_phipsi"],
        "w_rdist_norm": ref["w_rdist_norm"],
        "t_alpha (heavy atoms, pipeline)": ref["t_alpha"],
        "t_alpha (all atoms, Machaon)": ref["t_alpha"],
    }
    rows = []
    for name in ours:
        o, r = ours[name], reference[name]
        rows.append(
            {
                "metric": name,
                "pipeline": o,
                "reference": r,
                "abs_diff": abs(o - r) if o is not None and r is not None else None,
            }
        )
    return pd.DataFrame(rows)
