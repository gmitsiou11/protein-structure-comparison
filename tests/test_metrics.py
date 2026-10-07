"""
Unit tests for src/metrics.py and core numerical routines.

Run with:
    pytest tests/test_metrics.py -v

No structure files are needed — all tests use synthetic coordinate arrays.
"""

import numpy as np
import pytest

from types import SimpleNamespace

from src.metrics import (
    _kabsch_rmsd,
    _t_alpha,
    _w_rdist,
    _b_phipsi,
    _bhattacharyya_distance,
    summary_from_arrays,
    compute_all_metrics,
    MIN_B_PHIPSI_SAMPLES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_coords(n: int, seed: int = 42) -> np.ndarray:
    """Random Cα-like coordinate array, shape (n, 3)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3))


def _all_atom_coords(n: int, seed: int = 0) -> np.ndarray:
    """
    Dense all-atom point cloud (n, 3) for t-alpha testing.  With n >= 50
    standard-normal points, the alpha shape (alpha = 0.085, after MinMax
    scaling to [0, 1]^3) reliably has a non-zero number of triangles.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3))


def _random_rotation(seed: int = 0) -> np.ndarray:
    """Return a random proper rotation matrix (det = +1) via QR decomposition."""
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    # Ensure det = +1 (proper rotation, not a reflection)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _helix_angles(n: int, noise: float = 2.0, seed: int = 0) -> np.ndarray:
    """Approximate alpha-helix (phi, psi) pairs in degrees, with small noise."""
    rng = np.random.default_rng(seed)
    return np.full((n, 2), [-60.0, -45.0]) + rng.standard_normal((n, 2)) * noise


def _strand_angles(n: int, noise: float = 3.0, seed: int = 99) -> np.ndarray:
    """Approximate beta-strand (phi, psi) pairs in degrees."""
    rng = np.random.default_rng(seed)
    return np.full((n, 2), [-120.0, 130.0]) + rng.standard_normal((n, 2)) * noise


def _summary(n_res: int = 60, seed: int = 0, angles=None, atoms=None):
    """A synthetic StructureSummary of n_res residues."""
    ca = _random_coords(n_res, seed) * 10
    if angles is None:
        angles = _helix_angles(n_res, seed=seed)
    if atoms is None:
        atoms = _all_atom_coords(100, seed=seed)
    return summary_from_arrays(ca, "A" * n_res, angles, atoms)


# ---------------------------------------------------------------------------
# Kabsch RMSD
# ---------------------------------------------------------------------------

class TestKabschRMSD:
    def test_identical_structures_rmsd_zero(self):
        """RMSD between a structure and itself must be 0."""
        coords = _random_coords(50)
        assert _kabsch_rmsd(coords, coords) == pytest.approx(0.0, abs=1e-10)

    def test_rmsd_invariant_under_rotation(self):
        """RMSD must be the same after rotating one structure by any proper rotation."""
        coords_a = _random_coords(50, seed=1)
        coords_b = _random_coords(50, seed=2)
        R = _random_rotation(seed=7)
        rmsd_original = _kabsch_rmsd(coords_a, coords_b)
        rmsd_rotated  = _kabsch_rmsd(coords_a, coords_b @ R.T)
        assert rmsd_original == pytest.approx(rmsd_rotated, abs=1e-10)

    def test_rmsd_invariant_under_translation(self):
        """RMSD must be invariant to translation (centring is applied internally)."""
        coords_a = _random_coords(50, seed=1)
        coords_b = _random_coords(50, seed=2)
        translation = np.array([10.0, -5.0, 3.0])
        rmsd_original   = _kabsch_rmsd(coords_a, coords_b)
        rmsd_translated = _kabsch_rmsd(coords_a, coords_b + translation)
        assert rmsd_original == pytest.approx(rmsd_translated, abs=1e-10)

    def test_rmsd_non_negative(self):
        """RMSD must always be >= 0."""
        coords_a = _random_coords(30)
        coords_b = _random_coords(30, seed=99)
        assert _kabsch_rmsd(coords_a, coords_b) >= 0.0

    def test_rmsd_symmetric(self):
        """RMSD(A, B) must equal RMSD(B, A)."""
        coords_a = _random_coords(40, seed=3)
        coords_b = _random_coords(40, seed=4)
        assert _kabsch_rmsd(coords_a, coords_b) == pytest.approx(
            _kabsch_rmsd(coords_b, coords_a), abs=1e-10
        )

    def test_rmsd_correct_reflection_handling(self):
        """Kabsch reflection correction: mirroring a chiral structure must give
        a strictly larger RMSD than rotating it.

        If reflection correction were absent (or wrong), the algorithm would find
        an improper rotation (det R = -1) that maps a mirror image back to the
        original with RMSD ≈ 0, indistinguishable from a proper rotation.
        With correct reflection handling, the mirror image is not superimposable
        and must give a larger RMSD than a proper rotation of the same structure.

        This is the meaningful test: it distinguishes a correct implementation
        from one that silently accepts reflections.
        """
        rng = np.random.default_rng(42)
        coords_a = rng.standard_normal((30, 3))

        # Mirror image: reflect in the x-axis.  A generic point cloud in R^3 is
        # chiral, so the mirror image cannot be superimposed by any proper rotation.
        coords_mirror = coords_a.copy()
        coords_mirror[:, 0] *= -1

        # Proper rotation: det = +1, so RMSD should be essentially zero.
        Q = _random_rotation(seed=7)
        coords_rotated = coords_a @ Q.T

        rmsd_mirror  = _kabsch_rmsd(coords_a, coords_mirror)
        rmsd_rotated = _kabsch_rmsd(coords_a, coords_rotated)

        # Rotated version is the same structure — Kabsch must recover RMSD ≈ 0.
        assert rmsd_rotated == pytest.approx(0.0, abs=1e-10), (
            f"RMSD after proper rotation should be ~0, got {rmsd_rotated:.2e}"
        )
        # Mirror image is not superimposable — RMSD must be strictly positive.
        assert rmsd_mirror > 1e-6, (
            f"RMSD of mirror image should be > 0, got {rmsd_mirror:.2e}. "
            "Reflection correction may be missing or broken."
        )
        # And the mirror RMSD must exceed the rotation RMSD — this is the key check.
        assert rmsd_mirror > rmsd_rotated, (
            "Mirror image RMSD should exceed proper-rotation RMSD. "
            "If they are equal, the reflection correction is not working."
        )

    def test_rmsd_known_value(self):
        """Simple 3-point case with a known analytical RMSD."""
        # Two structures: A is at origin; B is shifted by 1 Å in each coordinate
        coords_a = np.array([[0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]])
        # B is A shifted by (1, 1, 1) — after centring, this is the same as A
        coords_b = coords_a + np.array([1.0, 1.0, 1.0])
        # After Kabsch centring, the two structures are identical → RMSD = 0
        assert _kabsch_rmsd(coords_a, coords_b) == pytest.approx(0.0, abs=1e-10)

    def test_rmsd_minimum_size(self):
        """RMSD should work with n=3 (the minimum accepted by the pipeline)."""
        coords_a = np.eye(3)       # 3 orthogonal unit vectors
        coords_b = np.eye(3) * 2  # scaled version
        rmsd = _kabsch_rmsd(coords_a, coords_b)
        assert rmsd >= 0.0


# ---------------------------------------------------------------------------
# t-alpha — compares the two log triangle counts stored in the summaries
# ---------------------------------------------------------------------------

class TestTAlpha:
    def test_identical_structures_gives_zero(self):
        s = _summary(seed=0)
        assert s.n_tri > 0, "use a denser point cloud: no alpha-shape triangles"
        assert _t_alpha(s.log_tri, s.log_tri) == pytest.approx(0.0, abs=1e-12)

    def test_non_negative_and_symmetric(self):
        a, b = _summary(seed=1), _summary(seed=2)
        assert _t_alpha(a.log_tri, b.log_tri) >= 0.0
        assert _t_alpha(a.log_tri, b.log_tri) == pytest.approx(_t_alpha(b.log_tri, a.log_tri))

    def test_dimensionless_order_of_magnitude(self):
        a, b = _summary(seed=10), _summary(seed=11)
        assert _t_alpha(a.log_tri, b.log_tri) < 10.0

    def test_none_for_too_few_atoms(self):
        tiny = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        s = _summary(atoms=tiny)
        assert s.log_tri is None and s.n_tri == 0
        assert _t_alpha(s.log_tri, _summary().log_tri) is None


# ---------------------------------------------------------------------------
# w-rdist — two 1-D samples of Cα–Cα distances, any lengths
# ---------------------------------------------------------------------------

class TestWRdist:
    def test_identical_structures_gives_zero(self):
        s = _summary(seed=3)
        result = _w_rdist(s.dists, s.dists)
        assert result["raw"] == pytest.approx(0.0, abs=1e-10)
        assert result["norm"] == pytest.approx(0.0, abs=1e-10)

    def test_norm_is_log10_compressed_raw(self):
        import math
        a, b = _summary(seed=7), _summary(seed=8)
        result = _w_rdist(a.dists, b.dists)
        assert result["raw"] >= 0.0
        assert result["norm"] == pytest.approx(math.log10(result["raw"] + 1.0), abs=1e-12)

    def test_symmetric(self):
        a, b = _summary(seed=9), _summary(seed=10)
        assert _w_rdist(a.dists, b.dists)["raw"] == pytest.approx(_w_rdist(b.dists, a.dists)["raw"])

    def test_different_lengths_allowed(self):
        """No correspondence: a 100-residue and an 80-residue chain are compared directly."""
        a, b = _summary(n_res=100, seed=1), _summary(n_res=80, seed=2)
        assert len(a.dists) != len(b.dists)
        assert _w_rdist(a.dists, b.dists)["raw"] > 0.0


# ---------------------------------------------------------------------------
# b-phipsi — raw (phi, psi) in degrees, 2-D Gaussian (as in Machaon)
# ---------------------------------------------------------------------------

class TestBPhipsi:
    def test_identical_distributions_near_zero(self):
        s = _summary(angles=_helix_angles(30))
        assert _b_phipsi(s, s) == pytest.approx(0.0, abs=1e-9)

    def test_none_below_min_samples(self):
        ok = _summary(angles=_helix_angles(MIN_B_PHIPSI_SAMPLES + 5))
        small = _summary(angles=_helix_angles(MIN_B_PHIPSI_SAMPLES - 1))
        assert _b_phipsi(small, ok) is None
        assert _b_phipsi(ok, small) is None

    def test_exactly_min_samples_accepted(self):
        s = _summary(angles=_helix_angles(MIN_B_PHIPSI_SAMPLES))
        assert _b_phipsi(s, s) is not None

    def test_degenerate_covariance_gives_none(self):
        """All pairs identical -> det(cov) = 0 -> undefined (Machaon returns False)."""
        flat = _summary(angles=np.tile([-60.0, -45.0], (20, 1)))
        assert flat.pp_det <= 0
        assert _b_phipsi(flat, _summary()) is None

    def test_helix_vs_strand_larger_than_helix_vs_helix(self):
        helix1 = _summary(angles=_helix_angles(40, seed=1))
        helix2 = _summary(angles=_helix_angles(40, seed=2))
        strand = _summary(angles=_strand_angles(40))
        assert _b_phipsi(helix1, strand) > _b_phipsi(helix1, helix2) >= 0.0


class TestBhattacharyyaDistance:
    @staticmethod
    def _gaussian(samples):
        cov = np.cov(samples.T)
        return samples.mean(axis=0), cov, float(np.linalg.det(cov))

    def test_identical_distributions_zero(self):
        g = self._gaussian(np.random.default_rng(0).standard_normal((30, 2)))
        assert _bhattacharyya_distance(*g, *g) == pytest.approx(0.0, abs=1e-9)

    def test_increases_with_separation(self):
        base = np.random.default_rng(2).standard_normal((40, 2))
        g0, g_small, g_large = (self._gaussian(base + k) for k in (0.0, 0.5, 5.0))
        assert _bhattacharyya_distance(*g0, *g_large) > _bhattacharyya_distance(*g0, *g_small) > 0.0

    def test_degenerate_returns_none(self):
        g = self._gaussian(np.random.default_rng(1).standard_normal((30, 2)))
        mean, cov, _ = g
        assert _bhattacharyya_distance(mean, cov, 0.0, *g) is None


# ---------------------------------------------------------------------------
# The Machaon metrics need no residue correspondence
# ---------------------------------------------------------------------------

class TestAlignmentFree:
    def test_residue_order_does_not_matter(self):
        """Shuffling the residues of a structure leaves all three metrics at 0,
        while an RMSD that pairs residue i with residue i becomes large."""
        rng = np.random.default_rng(0)
        ca = _random_coords(60, seed=4) * 10
        angles = _helix_angles(60, noise=10.0, seed=4)
        atoms = _all_atom_coords(100, seed=4)
        original = summary_from_arrays(ca, "A" * 60, angles, atoms)
        shuffled = summary_from_arrays(
            ca[rng.permutation(60)], "A" * 60,
            angles[rng.permutation(60)], atoms[rng.permutation(100)],
        )
        m = compute_all_metrics(original, shuffled)
        assert m["b_phipsi"] == pytest.approx(0.0, abs=1e-9)
        assert m["w_rdist_raw"] == pytest.approx(0.0, abs=1e-10)
        assert m["t_alpha"] == pytest.approx(0.0, abs=1e-12)
        assert _kabsch_rmsd(original.ca, shuffled.ca) > 1.0

    def test_different_lengths_no_alignment(self):
        m = compute_all_metrics(_summary(n_res=100, seed=1), _summary(n_res=80, seed=2))
        assert m["w_rdist_norm"] is not None
        assert m["b_phipsi"] is not None
        assert m["t_alpha"] is not None
        assert m["rmsd"] is None        # no PairAlignment -> no residue pairing -> no RMSD


class TestComputeAllMetrics:
    def test_rmsd_only_with_reliable_pairing(self):
        s = _summary(seed=5)
        pairs = lambda: (s.ca, s.ca)
        reliable = SimpleNamespace(rmsd_reliable=True, matched_coords=pairs)
        unreliable = SimpleNamespace(rmsd_reliable=False, matched_coords=pairs)
        assert compute_all_metrics(s, s, reliable)["rmsd"] == pytest.approx(0.0, abs=1e-9)
        assert compute_all_metrics(s, s, unreliable)["rmsd"] is None

    def test_output_keys_unchanged(self):
        s = _summary()
        assert set(compute_all_metrics(s, s)) == {
            "rmsd", "t_alpha", "t_alpha_n_tri_a", "t_alpha_n_tri_b",
            "w_rdist_raw", "w_rdist_norm", "b_phipsi", "b_phipsi_n_a", "b_phipsi_n_b",
        }
