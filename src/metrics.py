"""
Machaon structural comparison metrics (b-phipsi, w-rdist, t-alpha), plus Kabsch RMSD.

Machaon's metrics need no residue correspondence.  Each structure is first
reduced to a StructureSummary computed from its whole chain alone (the (phi, psi)
Gaussian, the distribution of all Cα–Cα distances, the alpha-shape triangle
count); two structures are then compared only through their summaries
(Kakoulidis et al. 2023, Commun Biol 6:752, Methods, Eq. 1–3).

RMSD is the one correspondence-based number: it uses the residue pairs of a
PairAlignment and is computed only when that pairing is reliable.

Because each metric compares a small summary (t-alpha: one number per
structure; b-phipsi: a mean and a 2x2 covariance; w-rdist: a 1-D distribution),
a value of 0 means "same summary", not "same structure".  Two properties
matter when the two structures belong to the same protein:
  * the summaries cover the whole chain, so residues present in one chain and
    absent in the other change the value (see sensitivity.truncation_experiment);
  * t-alpha scales each coordinate axis to [0, 1] before the alpha shape, as
    Machaon does, so it depends on how the structure is oriented in space
    (see sensitivity.orientation_floor).  The pipeline keeps Machaon's
    definition; the principal-axes variant is a sensitivity check.

Output units
------------
  rmsd          : Cα RMSD after Kabsch superposition (Å); None without a reliable pairing
  t_alpha       : exp(|log(n_triangles_A) − log(n_triangles_B)|) − 1  (dimensionless)
  w_rdist_raw   : 1-Wasserstein distance between the Cα–Cα distance distributions (Å)
  w_rdist_norm  : log10(w_rdist_raw + 1)  — the Machaon metric (dimensionless)
  b_phipsi      : Bhattacharyya distance between the 2-D (phi, psi) Gaussians (dimensionless)
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist
from scipy.linalg import det, inv
import open3d as o3d

from src.parser import (
    load_structure,
    get_ca_coords_and_seq,
    get_phi_psi_with_index,
    get_all_atom_coords,
)

if TYPE_CHECKING:
    from src.alignment import PairAlignment

MIN_B_PHIPSI_SAMPLES: int = 6
ALPHA = 0.085


def _bhattacharyya_distance(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    det_a: float,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    det_b: float,
) -> Optional[float]:
    """
    Bhattacharyya distance between two Gaussians given by (mean, cov, det).

    Same computation as Machaon's BhattacharyyaDistance.multivariate_compare_group:
    returns None (Machaon: False) when either covariance is degenerate (det <= 0).
    """
    if det_a <= 0 or det_b <= 0:
        return None
    cov = (cov_a + cov_b) / 2
    diff = mean_a - mean_b
    mahalanobis = float(diff @ inv(cov) @ diff)
    det_term = det(cov) / np.sqrt(det_a * det_b)
    return 0.125 * mahalanobis + 0.5 * float(np.log(det_term))


def _kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """
    Cα RMSD between two equal-length coordinate sets after optimal rigid-body
    superposition (Kabsch algorithm).

    Both arrays must have shape (N, 3) with N >= 3. The SVD decomposition
    guarantees a proper rotation (det R = +1) via the reflection correction.

    Returns RMSD in Angstroms.
    """
    ca = coords_a - coords_a.mean(axis=0)
    cb = coords_b - coords_b.mean(axis=0)

    H = cb.T @ ca
    U, S, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = ca - cb @ R.T
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def _minmax_normalize(coords: np.ndarray) -> np.ndarray:
    """Scale each axis independently to [0, 1] (Machaon: MinMaxScaler; depends on orientation)."""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    rng = hi - lo
    rng[rng == 0] = 1.0
    return (coords - lo) / rng


def _log_n_triangles(coords: np.ndarray):
    """Return (log(n_triangles), n_triangles), or (None, 0) on failure."""
    if coords is None or len(coords) < 4:
        return None, 0
    pts = _minmax_normalize(coords)
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, ALPHA)
        n = len(mesh.triangles)
    except Exception:
        return None, 0
    return (float(np.log(n)), n) if n > 0 else (None, 0)


@dataclass(frozen=True, eq=False)
class StructureSummary:
    """Everything the Machaon metrics need from one structure, computed once."""

    seq: str
    ca: np.ndarray  # (N, 3) Cα coordinates, whole chain
    dists: np.ndarray  # (N*(N-1)/2,) all Cα–Cα distances
    n_phipsi: int  # residues with both phi and psi
    pp_mean: Optional[np.ndarray]  # (2,) mean (phi, psi), degrees
    pp_cov: Optional[np.ndarray]  # (2, 2) covariance of (phi, psi)
    pp_det: float  # det(pp_cov); <= 0 means b-phipsi undefined
    log_tri: Optional[float]  # log(number of alpha-shape triangles)
    n_tri: int  # number of alpha-shape triangles


def summary_from_arrays(
    ca: np.ndarray, seq: str, phi_psi: np.ndarray, atoms: np.ndarray
) -> StructureSummary:
    """Build a StructureSummary from already-parsed arrays of one structure."""
    ca = np.asarray(ca, dtype=float).reshape(-1, 3)
    phi_psi = np.asarray(phi_psi, dtype=float).reshape(-1, 2)
    atoms = np.asarray(atoms, dtype=float).reshape(-1, 3)

    pp_mean, pp_cov, pp_det = None, None, 0.0
    if len(phi_psi) >= MIN_B_PHIPSI_SAMPLES:
        pp_mean = phi_psi.mean(axis=0)
        pp_cov = np.cov(phi_psi.T)
        pp_det = float(det(pp_cov))

    log_tri, n_tri = _log_n_triangles(atoms)

    return StructureSummary(
        seq=seq,
        ca=ca,
        dists=pdist(ca) if len(ca) >= 2 else np.empty(0),
        n_phipsi=len(phi_psi),
        pp_mean=pp_mean,
        pp_cov=pp_cov,
        pp_det=pp_det,
        log_tri=log_tri,
        n_tri=n_tri,
    )


@functools.lru_cache(maxsize=None)
def summarise_structure(
    path: str,
    model_idx: int = 0,
    chain_id: Optional[str] = None,
    residue_range: Optional[tuple[int, int]] = None,
) -> StructureSummary:
    """
    Parse one structure and summarise it: the whole chain, or only the residues
    numbered residue_range = (first, last) (a tuple, so that it can be cached).

    Cached: a structure that takes part in several comparisons is parsed once.
    Call summarise_structure.cache_clear() if files change during a session.
    """
    structure = load_structure(path)
    ca, seq = get_ca_coords_and_seq(structure, model_idx, chain_id, residue_range)
    phi_psi, _ = get_phi_psi_with_index(structure, model_idx, chain_id, residue_range)
    atoms = get_all_atom_coords(structure, model_idx, chain_id, residue_range)
    return summary_from_arrays(ca, seq, phi_psi, atoms)


def _t_alpha(log_tri_a: Optional[float], log_tri_b: Optional[float]) -> Optional[float]:
    """t-alpha (Machaon Eq. 3) from the two log triangle counts; None if either is missing."""
    if log_tri_a is None or log_tri_b is None:
        return None
    return float(np.exp(abs(log_tri_a - log_tri_b)) - 1)


def _w_rdist(dists_a: np.ndarray, dists_b: np.ndarray) -> dict:
    """
    w-rdist (Machaon Eq. 2): 1-Wasserstein distance between the distributions
    of all Cα–Cα distances of A and of B, then log10(raw + 1).

    The two arrays are separate samples and may have different lengths:
    no residue of A is paired with a residue of B.
    """
    if len(dists_a) == 0 or len(dists_b) == 0:
        return {"raw": None, "norm": None}
    raw = float(wasserstein_distance(dists_a, dists_b))
    return {"raw": raw, "norm": math.log10(raw + 1.0)}


def _b_phipsi(sa: StructureSummary, sb: StructureSummary) -> Optional[float]:
    """b-phipsi (Machaon Eq. 1) on raw (phi, psi) in degrees; None if undefined."""
    if sa.pp_mean is None or sb.pp_mean is None:
        return None
    return _bhattacharyya_distance(
        sa.pp_mean, sa.pp_cov, sa.pp_det, sb.pp_mean, sb.pp_cov, sb.pp_det
    )


def compute_all_metrics(
    sa: StructureSummary,
    sb: StructureSummary,
    val: Optional["PairAlignment"] = None,
) -> dict:
    """
    Compare two structures through their summaries.

    b-phipsi, w-rdist and t-alpha use only sa and sb (no alignment).
    RMSD is added only when a PairAlignment with a reliable residue
    pairing is given (val.rmsd_reliable); otherwise it is None.

    Returns
    -------
    dict with keys: rmsd, t_alpha, t_alpha_n_tri_a, t_alpha_n_tri_b,
    w_rdist_raw, w_rdist_norm, b_phipsi, b_phipsi_n_a, b_phipsi_n_b.
    """
    rmsd = None
    if val is not None and val.rmsd_reliable:
        ca_a, ca_b = val.matched_coords()
        rmsd = _kabsch_rmsd(ca_a, ca_b)

    wrdist = _w_rdist(sa.dists, sb.dists)

    return {
        "rmsd": rmsd,
        "t_alpha": _t_alpha(sa.log_tri, sb.log_tri),
        "t_alpha_n_tri_a": sa.n_tri,
        "t_alpha_n_tri_b": sb.n_tri,
        "w_rdist_raw": wrdist["raw"],
        "w_rdist_norm": wrdist["norm"],
        "b_phipsi": _b_phipsi(sa, sb),
        "b_phipsi_n_a": sa.n_phipsi,
        "b_phipsi_n_b": sb.n_phipsi,
    }
