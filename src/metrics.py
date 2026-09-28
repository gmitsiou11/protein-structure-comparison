"""
Machaon-based structural comparison metrics, plus Kabsch RMSD.

The public entry point is compute_all_metrics(val), which takes a
ValidationResult from validation.py and returns all metric values computed
on the pre-parsed data stored in that result.  No file I/O or re-alignment
is performed here.

Output units
------------
  rmsd          : Cα RMSD after Kabsch superposition (Å)
  t_alpha : exp(|log(n_triangles_A) − log(n_triangles_B)|) − 1  (dimensionless)
  w_rdist_raw   : 1-Wasserstein distance between flat upper-triangle Cα distance
                  distributions of A and B (Å)
  w_rdist_norm  : log10(w_rdist_raw + 1)  — the Machaon metric (dimensionless)
  b_phipsi      : Bhattacharyya distance on circular-embedded phi/psi (dimensionless)
"""

from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.linalg import det, inv
import open3d as o3d

if TYPE_CHECKING:
    from src.validation import ValidationResult

MIN_B_PHIPSI_SAMPLES: int = 6
ALPHA = 0.085


def _bhattacharyya_distance(group_one: np.ndarray, group_two: np.ndarray) -> float:
    """
    Multivariate Bhattacharyya distance between two sets of samples,
    assuming Gaussian distributions.

    Parameters
    ----------
    group_one, group_two : ndarray of shape (N, 4)
        Circular-embedded angle vectors [cos phi, sin phi, cos psi, sin psi].

    Returns
    -------
    float
        Bhattacharyya distance >= 0 (0 = identical distributions).
    """
    m1, m2 = np.mean(group_one, axis=0), np.mean(group_two, axis=0)
    c1, c2 = np.cov(group_one.T), np.cov(group_two.T)

    _reg = 1e-6 * np.eye(group_one.shape[1])
    c1_reg = c1 + _reg
    c2_reg = c2 + _reg
    cav = (c1_reg + c2_reg) / 2

    diff = m1 - m2
    mahal = float(np.dot(diff, inv(cav) @ diff))

    det_term = det(cav) / np.sqrt(det(c1_reg) * det(c2_reg))
    log_det_ratio = np.log(det_term)

    return 0.125 * mahal + 0.5 * log_det_ratio


def _angles_to_circular(angles_deg: np.ndarray) -> np.ndarray:
    """
    Map (phi, psi) angle pairs in degrees to a 4D circular embedding.

    Backbone dihedrals are periodic: -180 and +180 degrees are the same point.
    Naive linear statistics give incorrect distances near the wrap-around boundary.
    Mapping theta -> (cos theta, sin theta) produces a representation where
    Euclidean distance is geometrically correct for circular quantities.

    Parameters
    ----------
    angles_deg : ndarray of shape (N, 2)
        Columns are [phi, psi] in degrees.

    Returns
    -------
    ndarray of shape (N, 4) : [cos phi, sin phi, cos psi, sin psi]
    """
    rad = np.deg2rad(angles_deg)
    return np.column_stack(
        [
            np.cos(rad[:, 0]),
            np.sin(rad[:, 0]),
            np.cos(rad[:, 1]),
            np.sin(rad[:, 1]),
        ]
    )


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


def _dist_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Full N x N Cα pairwise distance matrix.

    Parameters
    ----------
    coords : ndarray of shape (N, 3)

    Returns
    -------
    ndarray of shape (N, N); entry [i, j] is the Euclidean distance between
    coords[i] and coords[j]. Diagonal is 0.
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def _minmax_normalize(coords: np.ndarray) -> np.ndarray:
    """Scale each axis independently to [0, 1]."""
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


def _t_alpha(
    all_atom_coords_a: np.ndarray,
    all_atom_coords_b: np.ndarray,
) -> dict:
    """
    t-alpha metric from Machaon.

    For each structure:
    1. Normalise all heavy-atom coordinates to [0, 1] per axis.
    2. Build a 3D alpha-shape with alpha=0.085, matching Machaon's hardcoded parameter.
    3. Count the boundary triangles n and store log(n).

    Parameters
    ----------
    all_atom_coords_a, all_atom_coords_b : ndarray of shape (K, 3)
        All heavy-atom coordinates.

    Returns
    -------
    dict with keys:
      'raw'          — metric value (dimensionless), or None on failure
      'norm'         — same as raw (Machaon applies no separate normalisation)
      'n_triangles_a', 'n_triangles_b' — raw triangle counts for diagnostics
    """

    log_a, n_a = _log_n_triangles(all_atom_coords_a)
    log_b, n_b = _log_n_triangles(all_atom_coords_b)

    if log_a is None or log_b is None:
        return {"t_alpha": None, "n_triangles_a": n_a, "n_triangles_b": n_b}

    val = float(np.exp(abs(log_a - log_b)) - 1)
    return {"t_alpha": val, "n_triangles_a": n_a, "n_triangles_b": n_b}


def _w_rdist(mat_a: np.ndarray, mat_b: np.ndarray) -> dict:
    """
    w-rdist metric from Machaon.

    Flattens the upper-triangle Cα pairwise distance arrays of both structures
    into 1D distributions and computes the 1-Wasserstein (Earth Mover's Distance)
    between them.
    Uses scipy.stats.wasserstein_distance in place of Machaon's manual CDF
    implementation.

    Parameters
    ----------
    mat_a, mat_b : ndarray of shape (N, N)
        Full Cα pairwise distance matrices for the aligned residues.

    Returns
    -------
    dict with keys:
      'raw'  — raw 1-Wasserstein distance in Å (before log compression)
      'norm' — log10(raw + 1), the Machaon metric (dimensionless)
    """
    upper = np.triu_indices(len(mat_a), k=1)
    dist_a = mat_a[upper]
    dist_b = mat_b[upper]
    raw = float(wasserstein_distance(dist_a, dist_b))
    norm = float(math.log10(raw + 1.0))
    return {"raw": raw, "norm": norm}


def _b_phipsi(
    phi_psi_a: np.ndarray,
    phi_psi_b: np.ndarray,
) -> Optional[float]:
    """
    b-phipsi metric from Machaon.

    Angles are mapped to a 4D circular embedding before computing the distance,
    so the periodic boundary at +-180 degrees is handled correctly — deviation
    from the original which operates on raw degree values.

    M_a and M_b may differ because terminal residues and residues adjacent to
    prolines lack one of the two dihedral angles. The Bhattacharyya distance
    treats the two arrays as independent empirical distributions, so different
    lengths are acceptable. Requires at least MIN_B_PHIPSI_SAMPLES valid angle
    pairs in each structure; returns None otherwise.

    Parameters
    ----------
    phi_psi_a, phi_psi_b : ndarray of shape (M, 2)
        [phi, psi] in degrees, restricted to sequence-aligned positions only.

    Returns
    -------
    float or None
        Bhattacharyya distance, or None if either array is too small to
        estimate a non-singular 4x4 covariance matrix.
    """
    if phi_psi_a is None or phi_psi_b is None:
        return None
    if len(phi_psi_a) < MIN_B_PHIPSI_SAMPLES or len(phi_psi_b) < MIN_B_PHIPSI_SAMPLES:
        return None
    try:
        return float(
            _bhattacharyya_distance(
                _angles_to_circular(phi_psi_a),
                _angles_to_circular(phi_psi_b),
            )
        )
    except Exception:
        return None


def compute_all_metrics(val: "ValidationResult") -> dict:
    """
    Compute all structural comparison metrics from a pre-validated pair.

    Called by pipeline.py. All coordinate arrays and phi/psi angles are
    taken directly from the ValidationResult — no file I/O, no re-alignment.
    The Cα pairwise distance matrix is computed once and used by w-rdist.
    t-alpha uses the pre-loaded all-atom coordinate arrays (val.all_atom_coords_a/b).

    Parameters
    ----------
    val : ValidationResult with valid=True.

    Returns
    -------
    dict with keys:
      rmsd, t_alpha, t_alpha_n_triangles_a, t_alpha_n_triangles_b,
      w_rdist_raw, w_rdist_norm, b_phipsi, b_phipsi_n_a, b_phipsi_n_b
    All values are float, int, or None.

    Raises
    ------
    ValueError if val.valid is False, or if fewer than 3 residues are aligned.
    """
    if not val.valid:
        raise ValueError(
            f"compute_all_metrics() called on an invalid ValidationResult. "
            f"Rejection reason: {val.reason}"
        )

    ca_a, ca_b = val.matched_coords()

    if len(ca_a) < 3:
        raise ValueError(
            f"Fewer than 3 aligned residues ({len(ca_a)}); cannot compute metrics."
        )

    mat_a = _dist_matrix(ca_a)
    mat_b = _dist_matrix(ca_b)

    talpha = _t_alpha(val.all_atom_coords_a, val.all_atom_coords_b)
    wrdist = _w_rdist(mat_a, mat_b)
    bphipsi = _b_phipsi(val.phi_psi_a, val.phi_psi_b)

    n_phi_psi_a = len(val.phi_psi_a) if val.phi_psi_a is not None else 0
    n_phi_psi_b = len(val.phi_psi_b) if val.phi_psi_b is not None else 0

    return {
        "rmsd": _kabsch_rmsd(ca_a, ca_b),
        "t_alpha": talpha["t_alpha"],
        "t_alpha_n_tri_a": talpha.get("n_triangles_a"),
        "t_alpha_n_tri_b": talpha.get("n_triangles_b"),
        "w_rdist_raw": wrdist["raw"],
        "w_rdist_norm": wrdist["norm"],
        "b_phipsi": bphipsi,
        "b_phipsi_n_a": n_phi_psi_a,
        "b_phipsi_n_b": n_phi_psi_b,
    }
