"""
Unit tests for src/metrics.py and core numerical routines.

Run with:
    pytest tests/test_metrics.py -v

No structure files are needed — all tests use synthetic coordinate arrays.
"""

import numpy as np
import pytest

from src.metrics import (
    _kabsch_rmsd,
    _dist_matrix,
    _t_alpha,
    _w_rdist,
    _b_phipsi,
    _angles_to_circular,
    _bhattacharyya_distance,
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
    Dense all-atom coordinate array suitable for t-alpha testing.

    _t_alpha expects all heavy-atom coordinates (shape (K, 3)), not a distance
    matrix.  With n >= 50 standard-normal points, after MinMax normalisation to
    [0,1]^3 the alpha-shape (alpha=0.085) reliably produces a non-zero triangle
    count, so identity comparisons return 0 and non-identity comparisons return
    a positive value.

    Do NOT pass _dist_matrix(coords) to _t_alpha — that is an (n, n) matrix
    interpreted as n points in n-dimensional space, which causes Delaunay to
    fail (QhullError: not enough points for initial simplex in n-D) and all
    _t_alpha tests to return {raw: None}.
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
# Distance matrix
# ---------------------------------------------------------------------------

class TestDistMatrix:
    def test_diagonal_is_zero(self):
        """Pairwise distance from a point to itself must be 0."""
        coords = _random_coords(20)
        D = _dist_matrix(coords)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_symmetry(self):
        """Distance matrix must be symmetric."""
        coords = _random_coords(15)
        D = _dist_matrix(coords)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_non_negative(self):
        """All distances must be non-negative."""
        coords = _random_coords(10)
        D = _dist_matrix(coords)
        assert np.all(D >= 0.0)

    def test_triangle_inequality(self):
        """Basic triangle inequality sanity check for a few triplets."""
        coords = _random_coords(5)
        D = _dist_matrix(coords)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    assert D[i, j] <= D[i, k] + D[k, j] + 1e-10


# ---------------------------------------------------------------------------
# t-alpha
#
# IMPORTANT: _t_alpha(all_atom_coords_a, all_atom_coords_b) takes raw (K, 3)
# coordinate arrays — NOT (N, N) distance matrices.
#
# Passing a distance matrix (as produced by _dist_matrix) is wrong: an (N×N)
# matrix is interpreted as N points in N-dimensional space, which causes
# Delaunay triangulation to fail (QhullError: not enough points for an N-D
# simplex) and every test to silently return {"raw": None, "norm": None}.
#
# All tests below use _all_atom_coords() which generates a (n, 3) array.
# ---------------------------------------------------------------------------

class TestTAlpha:
    def test_identical_structures_gives_zero(self):
        """t-alpha of a structure against itself must be 0."""
        coords = _all_atom_coords(100)            # (100, 3) — not a distance matrix
        result = _t_alpha(coords, coords)
        assert result["raw"] is not None, (
            "t_alpha returned None for identical structures; "
            "the alpha-shape produced no triangles — use a denser point cloud."
        )
        assert result["raw"] == pytest.approx(0.0, abs=1e-12)
        assert result["norm"] == pytest.approx(0.0, abs=1e-12)

    def test_raw_non_negative(self):
        """t-alpha raw must be >= 0."""
        ca = _all_atom_coords(100, seed=1)        # (100, 3)
        cb = _all_atom_coords(100, seed=2)        # (100, 3) — different structure
        result = _t_alpha(ca, cb)
        assert result["raw"] is not None
        assert result["raw"] >= 0.0

    def test_norm_non_negative(self):
        """t-alpha normalised must be >= 0."""
        ca = _all_atom_coords(100, seed=3)
        cb = _all_atom_coords(100, seed=4)
        result = _t_alpha(ca, cb)
        assert result["norm"] is not None
        assert result["norm"] >= 0.0

    def test_norm_dimensionless_order_of_magnitude(self):
        """For two random structures with similar scale, normalised t-alpha should be < 10."""
        ca = _all_atom_coords(100, seed=10)
        cb = _all_atom_coords(100, seed=11)
        result = _t_alpha(ca, cb)
        assert result["norm"] is not None
        assert result["norm"] < 10.0

    def test_returns_none_for_too_few_points(self):
        """t-alpha must return None when fewer than 4 points are provided."""
        tiny = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 3 points
        normal = _all_atom_coords(100)
        result = _t_alpha(tiny, normal)
        assert result["raw"] is None

    def test_triangle_counts_are_non_negative(self):
        """n_triangles_a and n_triangles_b must be non-negative integers."""
        ca = _all_atom_coords(100, seed=5)
        cb = _all_atom_coords(100, seed=6)
        result = _t_alpha(ca, cb)
        assert result["n_triangles_a"] >= 0
        assert result["n_triangles_b"] >= 0


# ---------------------------------------------------------------------------
# w-rdist
# (correctly takes (N, N) distance matrices — _dist_matrix usage is right here)
# ---------------------------------------------------------------------------

class TestWRdist:
    def test_identical_structures_gives_zero(self):
        """w-rdist of a structure against itself must be 0."""
        coords = _random_coords(30)
        D = _dist_matrix(coords)
        result = _w_rdist(D, D)
        assert result["raw"] == pytest.approx(0.0, abs=1e-10)
        assert result["norm"] == pytest.approx(0.0, abs=1e-10)

    def test_raw_non_negative(self):
        """w-rdist raw must be >= 0."""
        ca = _random_coords(20, seed=5)
        cb = _random_coords(20, seed=6)
        result = _w_rdist(_dist_matrix(ca), _dist_matrix(cb))
        assert result["raw"] >= 0.0

    def test_norm_is_log10_compressed_raw(self):
        """w-rdist norm must equal log10(raw + 1)."""
        import math
        ca = _random_coords(25, seed=7)
        cb = _random_coords(25, seed=8)
        result = _w_rdist(_dist_matrix(ca), _dist_matrix(cb))
        expected_norm = math.log10(result["raw"] + 1.0)
        assert result["norm"] == pytest.approx(expected_norm, abs=1e-12)

    def test_symmetric(self):
        """w-rdist(A, B) must equal w-rdist(B, A) since Wasserstein is symmetric."""
        ca = _random_coords(20, seed=9)
        cb = _random_coords(20, seed=10)
        r1 = _w_rdist(_dist_matrix(ca), _dist_matrix(cb))
        r2 = _w_rdist(_dist_matrix(cb), _dist_matrix(ca))
        assert r1["raw"] == pytest.approx(r2["raw"], abs=1e-10)


# ---------------------------------------------------------------------------
# Circular embedding
# ---------------------------------------------------------------------------

class TestAnglesCircular:
    def test_output_shape(self):
        """Circular embedding should produce (N, 4) output from (N, 2) input."""
        angles = np.array([[0.0, 90.0], [180.0, -90.0], [45.0, 135.0]])
        emb = _angles_to_circular(angles)
        assert emb.shape == (3, 4)

    def test_known_values(self):
        """phi=0 → cos(0)=1, sin(0)=0; psi=90 → cos(90)=0, sin(90)=1."""
        angles = np.array([[0.0, 90.0]])
        emb = _angles_to_circular(angles)
        np.testing.assert_allclose(emb[0, 0], 1.0, atol=1e-10)   # cos(phi=0)
        np.testing.assert_allclose(emb[0, 1], 0.0, atol=1e-10)   # sin(phi=0)
        np.testing.assert_allclose(emb[0, 2], 0.0, atol=1e-10)   # cos(psi=90)
        np.testing.assert_allclose(emb[0, 3], 1.0, atol=1e-10)   # sin(psi=90)

    def test_180_minus180_are_same_point(self):
        """Circular embedding must map +180 and -180 to the same point."""
        plus180  = _angles_to_circular(np.array([[180.0, 180.0]]))
        minus180 = _angles_to_circular(np.array([[-180.0, -180.0]]))
        np.testing.assert_allclose(plus180, minus180, atol=1e-10)

    def test_unit_circle_norm(self):
        """Each (cos θ, sin θ) pair must lie on the unit circle."""
        angles = np.random.default_rng(0).uniform(-180, 180, (50, 2))
        emb = _angles_to_circular(angles)
        phi_norms = emb[:, 0]**2 + emb[:, 1]**2
        psi_norms = emb[:, 2]**2 + emb[:, 3]**2
        np.testing.assert_allclose(phi_norms, 1.0, atol=1e-10)
        np.testing.assert_allclose(psi_norms, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# b-phipsi
# ---------------------------------------------------------------------------

class TestBPhipsi:
    def _helix_angles(self, n: int, noise: float = 2.0, seed: int = 0) -> np.ndarray:
        """Generate approximate alpha-helix phi/psi angles with small noise."""
        rng = np.random.default_rng(seed)
        angles = np.full((n, 2), [-60.0, -45.0])
        angles += rng.standard_normal((n, 2)) * noise
        return angles

    def test_identical_distributions_near_zero(self):
        """b-phipsi of identical angle distributions must be ~0."""
        angles = self._helix_angles(30)
        result = _b_phipsi(angles, angles)
        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_returns_none_below_min_samples(self):
        """b-phipsi must return None when either array is too small."""
        angles_ok   = self._helix_angles(MIN_B_PHIPSI_SAMPLES + 5)
        angles_small = self._helix_angles(MIN_B_PHIPSI_SAMPLES - 1)
        assert _b_phipsi(angles_small, angles_ok)   is None
        assert _b_phipsi(angles_ok,   angles_small) is None

    def test_returns_none_for_none_input(self):
        """b-phipsi must return None if either input is None."""
        angles = self._helix_angles(20)
        assert _b_phipsi(None,   angles) is None
        assert _b_phipsi(angles, None)   is None

    def test_helix_vs_strand_is_larger_than_helix_vs_helix(self):
        """Two different secondary structure distributions should give a larger
        Bhattacharyya distance than two samples from the same distribution."""
        helix  = self._helix_angles(40, seed=1)
        helix2 = self._helix_angles(40, seed=2)     # same secondary structure
        # Beta-strand approximate angles
        rng = np.random.default_rng(99)
        strand = rng.standard_normal((40, 2)) * 3 + np.array([-120.0, 130.0])

        d_same      = _b_phipsi(helix, helix2)
        d_different = _b_phipsi(helix, strand)

        assert d_same is not None
        assert d_different is not None
        assert d_different > d_same

    def test_non_negative(self):
        """Bhattacharyya distance must always be >= 0."""
        a = self._helix_angles(20, seed=3)
        b = self._helix_angles(20, seed=4)
        result = _b_phipsi(a, b)
        assert result is not None
        assert result >= 0.0

    def test_exactly_min_samples_accepted(self):
        """Exactly MIN_B_PHIPSI_SAMPLES should be accepted (boundary case)."""
        angles = self._helix_angles(MIN_B_PHIPSI_SAMPLES)
        result = _b_phipsi(angles, angles)
        # Identical → ~0, or at least not None
        assert result is not None


# ---------------------------------------------------------------------------
# Bhattacharyya distance (internal)
# ---------------------------------------------------------------------------

class TestBhattacharyyaDistance:
    def test_identical_distributions_zero(self):
        """Bhattacharyya distance between a distribution and itself must be 0."""
        rng = np.random.default_rng(0)
        samples = rng.standard_normal((30, 4))
        d = _bhattacharyya_distance(samples, samples)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((25, 4))
        b = rng.standard_normal((25, 4)) + 1.0
        d = _bhattacharyya_distance(a, b)
        assert d >= 0.0

    def test_increases_with_separation(self):
        """Moving two Gaussians further apart should increase Bhattacharyya distance."""
        rng = np.random.default_rng(2)
        base = rng.standard_normal((40, 4))
        shifted_small = base + 0.5
        shifted_large = base + 5.0
        d_small = _bhattacharyya_distance(base, shifted_small)
        d_large = _bhattacharyya_distance(base, shifted_large)
        assert d_large > d_small
