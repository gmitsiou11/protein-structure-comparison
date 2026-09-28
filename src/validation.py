"""
Comparison eligibility validation for protein structure pairs.

Before any metric is computed, this module determines whether two structures
are scientifically valid to compare.  A comparison is valid only when the
structures represent the same protein region with sufficient sequence overlap
and identity.  Mismatched or low-quality comparisons are rejected early,
preventing misleading metric values.

Architectural role
------------------

  1. Structure parsing (Ca coordinates, sequences)
  2. Sequence alignment (Needleman-Wunsch, computed ONCE)
  3. Alignment index arrays (idx_a, idx_b) reused by all metric functions
  4. Phi/psi angle extraction filtered to aligned positions

The ``ValidationResult`` dataclass stores all pre-parsed data so that
``metrics.py`` functions can operate.

Key thresholds
--------------
  SEQ_IDENTITY_MIN  = 0.90  — 90 % of matched residues must have the same
                               amino acid.  Below this, global alignment may
                               be aligning non-homologous regions.
  COVERAGE_MIN      = 0.80  — At least 80 % of EACH structure's residues
                               must be matched.  This rejects fragment-vs-full
                               comparisons unless both structures sufficiently
                               cover the same protein region.
  MIN_MATCHED       = 30    — Hard floor.  All metrics become unreliable below
                               ~30 matched residues (especially b_phipsi whose
                               Gaussian assumption needs sufficient sample size).
  X_RESIDUE_WARN    = 0.10  — If >10 % of residues are unknown (\"X\"), emit a
                               warning.  Metrics still run but reliability
                               is reduced.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from Bio import Align

from src.parser import (
    load_structure,
    get_ca_coords_and_seq,
    get_phi_psi_with_index,
    get_all_atom_coords,
)

SEQ_IDENTITY_MIN: float = (
    0.90  # minimum fraction of matched positions with identical residue
)
COVERAGE_MIN: float = 0.80  # minimum coverage required for both structures
MIN_MATCHED: int = 30  # absolute minimum matched residues
X_RESIDUE_WARN: float = 0.10  # warn if more than this fraction are unknown


@dataclass
class ValidationResult:
    """
    Holds all eligibility information and pre-parsed data for a structure pair.

    Design principle
    ----------------
    This class serves as both a validation verdict AND a data container.
    When ``valid=True``, all array fields are populated so that metric
    functions in ``metrics.py`` can operate without re-reading files or
    re-running the sequence alignment.

    Fields — eligibility
    --------------------
    valid          : bool   — True iff the pair passes all hard criteria.
    reason         : str    — Human-readable explanation (used in output JSON).
    warnings       : list   — Non-fatal observations (e.g. high X fraction).

    Fields — residue counts
    -----------------------
    n_residues_a   : int    — Total Cα residues parsed from structure A.
    n_residues_b   : int    — Total Cα residues parsed from structure B.
    n_matched      : int    — Residues paired by the sequence alignment.
    n_mismatches   : int    — Matched positions where residue types differ.
    n_gaps         : int    — Total gap columns in the global alignment
                              (= n_a + n_b - 2*n_matched, combining both directions).

    Fields — coverage
    -----------------
    coverage_a     : float  — n_matched / n_residues_a  (fraction of A covered).
    coverage_b     : float  — n_matched / n_residues_b  (fraction of B covered).
    coverage       : float  — n_matched / min(n_a, n_b)  (shorter-sequence coverage;
                              retained for reporting/backward compatibility only).
                              Hard validation uses both coverage_a and coverage_b.

    Fields — sequence quality
    -------------------------
    seq_identity   : float  — Identical residues / n_matched.
    x_frac_a       : float  — Fraction of unknown residues (\"X\") in structure A.
    x_frac_b       : float  — Fraction of unknown residues (\"X\") in structure B.

    Fields — alignment indices (populated when valid=True)
    -------------------------------------------------------
    idx_a, idx_b   : np.ndarray — 1-D index arrays of length n_matched into
                                   the Cα coordinate arrays of A and B respectively.
                                   coords_a[idx_a] and coords_b[idx_b] give the
                                   aligned coordinate pairs.

    Fields — pre-parsed data (populated when valid=True)
    -----------------------------------------------------
    coords_a       : np.ndarray, shape (n_a, 3) — full Cα coords for structure A.
    coords_b       : np.ndarray, shape (n_b, 3) — full Cα coords for structure B.
    seq_a, seq_b   : str        — one-letter amino acid sequences.
    phi_psi_a      : np.ndarray, shape (M_a, 2) — [phi, psi] in degrees for
                                   residues in A that (i) have valid dihedral angles
                                   AND (ii) are at aligned positions.  M_a ≤ n_matched.
    phi_psi_b      : np.ndarray, shape (M_b, 2) — same for structure B.
                     Note: M_a and M_b may differ because terminal residues and
                     residues before/after prolines lack one of the two angles.
                     This is expected and handled by b_phipsi as independent samples.
    """

    valid: bool
    reason: str
    warnings: list[str] = field(default_factory=list)

    n_residues_a: int = 0
    n_residues_b: int = 0
    n_matched: int = 0
    n_mismatches: int = 0
    n_gaps: int = 0

    coverage_a: float = 0.0  # n_matched / n_residues_a
    coverage_b: float = 0.0  # n_matched / n_residues_b
    coverage: float = 0.0  # n_matched / min(n_a, n_b)
    seq_identity: float = 0.0
    x_frac_a: float = 0.0
    x_frac_b: float = 0.0

    idx_a: Optional[np.ndarray] = None
    idx_b: Optional[np.ndarray] = None

    coords_a: Optional[np.ndarray] = None  # shape (n_a, 3)
    coords_b: Optional[np.ndarray] = None  # shape (n_b, 3)
    seq_a: str = ""
    seq_b: str = ""
    phi_psi_a: Optional[np.ndarray] = None  # shape (M_a, 2), aligned positions only
    phi_psi_b: Optional[np.ndarray] = None  # shape (M_b, 2), aligned positions only
    all_atom_coords_a: Optional[np.ndarray] = (
        None  # shape (K_a, 3), all heavy atoms — used by t-alpha
    )
    all_atom_coords_b: Optional[np.ndarray] = (
        None  # shape (K_b, 3), all heavy atoms — used by t-alpha
    )

    def matched_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the sequence-aligned Cα coordinate pairs.

        Returns
        -------
        (ca_a_matched, ca_b_matched) — both of shape (n_matched, 3).

        Raises ValueError if called on an invalid ValidationResult.
        """
        if (
            self.coords_a is None
            or self.coords_b is None
            or self.idx_a is None
            or self.idx_b is None
        ):
            raise ValueError(
                "matched_coords() called on an invalid or unpopulated ValidationResult. "
                "Check .valid before calling."
            )
        return self.coords_a[self.idx_a], self.coords_b[self.idx_b]

    def to_record(self) -> dict:
        """
        Return a JSON-serialisable summary of all validation fields.

        Excludes large numpy arrays (coords, phi_psi).  Use this to embed
        validation metadata in every comparison record so output is self-documenting.
        """
        return {
            "valid": self.valid,
            "reason": self.reason,
            "warnings": self.warnings,
            "n_residues_a": self.n_residues_a,
            "n_residues_b": self.n_residues_b,
            "n_matched": self.n_matched,
            "n_mismatches": self.n_mismatches,
            "n_gaps": self.n_gaps,
            "coverage_a": round(self.coverage_a, 4),
            "coverage_b": round(self.coverage_b, 4),
            "coverage": round(self.coverage, 4),
            "seq_identity": round(self.seq_identity, 4),
            "x_frac_a": round(self.x_frac_a, 4),
            "x_frac_b": round(self.x_frac_b, 4),
        }


def validate_pair(
    path_a: str,
    path_b: str,
    model_idx_a: int = 0,
    model_idx_b: int = 0,
    chain_id: str = "A",
    chain_id_a: Optional[str] = None,
    chain_id_b: Optional[str] = None,
    seq_identity_min: float = SEQ_IDENTITY_MIN,
    coverage_min: float = COVERAGE_MIN,
    min_matched: int = MIN_MATCHED,
) -> ValidationResult:
    """
    Check whether structures A and B are valid to compare AND pre-parse all data
    needed by metric functions.

    This function is the single entry point for all structure parsing and alignment.
    When it returns a valid result, ``metrics.py`` functions use the stored coords,
    indices, and phi/psi arrays directly.

    Parameters
    ----
    path_a, path_b    : paths to mmCIF files.
    model_idx_a/b     : model indices (0 for X-ray / Cryo-EM / AF;
                        use proteins.csv nmr_model for NMR structures).
    chain_id          : legacy fallback chain (used when chain_id_a/b not given).
    chain_id_a        : chain to extract from structure A (overrides chain_id).
    chain_id_b        : chain to extract from structure B (overrides chain_id).
                        This enables comparing multi-chain experimental structures
                        (e.g. hemoglobin chain B) against AF2 predictions (chain A).
    seq_identity_min  : override default identity threshold.
    coverage_min      : override default per-structure coverage threshold.
    min_matched       : override default minimum matched residues.

    Returns
    -------
    ValidationResult with:
      - .valid indicating pass/fail
      - .reason giving the exact rejection reason (always populated)
      - all diagnostic coverage/identity fields
      - pre-parsed coords, alignment indices, and filtered phi/psi arrays
        (only when valid=True)
    """
    # Resolve per-structure chain IDs; fall back to legacy chain_id.
    cid_a: str = chain_id_a if chain_id_a is not None else chain_id
    cid_b: str = chain_id_b if chain_id_b is not None else chain_id

    warns: list[str] = []

    try:
        struct_a = load_structure(path_a)
        coords_a, seq_a = get_ca_coords_and_seq(struct_a, model_idx_a, cid_a)
    except Exception as exc:
        return ValidationResult(
            valid=False,
            reason=f"Failed to parse structure A ({path_a}): {exc}",
        )

    try:
        struct_b = load_structure(path_b)
        coords_b, seq_b = get_ca_coords_and_seq(struct_b, model_idx_b, cid_b)
    except Exception as exc:
        return ValidationResult(
            valid=False,
            reason=f"Failed to parse structure B ({path_b}): {exc}",
            n_residues_a=len(coords_a),
        )

    n_a, n_b = len(coords_a), len(coords_b)

    if n_a < 5:
        return ValidationResult(
            valid=False,
            reason=(
                f"Structure A ({path_a}) has only {n_a} Cα residues after parsing "
                f"chain {cid_a} model {model_idx_a}.  Minimum required: 5.  "
                "Check chain_id and model_idx in proteins.csv."
            ),
            n_residues_a=n_a,
            n_residues_b=n_b,
        )
    if n_b < 5:
        return ValidationResult(
            valid=False,
            reason=(
                f"Structure B ({path_b}) has only {n_b} Cα residues after parsing "
                f"chain {cid_b} model {model_idx_b}.  Minimum required: 5.  "
                "Check chain_id and model_idx in proteins.csv."
            ),
            n_residues_a=n_a,
            n_residues_b=n_b,
        )

    x_frac_a = seq_a.count("X") / max(n_a, 1)
    x_frac_b = seq_b.count("X") / max(n_b, 1)

    if x_frac_a > X_RESIDUE_WARN:
        warns.append(
            f'Structure A ({path_a}): {x_frac_a:.1%} unknown residues ("X"). '
            "Metric reliability is reduced when residue identity cannot be confirmed."
        )
    if x_frac_b > X_RESIDUE_WARN:
        warns.append(
            f'Structure B ({path_b}): {x_frac_b:.1%} unknown residues ("X"). '
            "Metric reliability is reduced when residue identity cannot be confirmed."
        )

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5

    try:
        alignment = aligner.align(seq_a, seq_b)[0]
    except Exception as exc:
        return ValidationResult(
            valid=False,
            reason=f"Sequence alignment failed: {exc}",
            n_residues_a=n_a,
            n_residues_b=n_b,
            x_frac_a=x_frac_a,
            x_frac_b=x_frac_b,
        )

    # Extract matched index pairs and alignment statistics
    idx_a_list: list[int] = []
    idx_b_list: list[int] = []
    identical = 0

    for (sa, ea), (sb, eb) in zip(alignment.aligned[0], alignment.aligned[1]):
        block_len = ea - sa  # both sides of a matched block have the same length
        for k in range(block_len):
            ia, ib = sa + k, sb + k
            idx_a_list.append(ia)
            idx_b_list.append(ib)
            if seq_a[ia] == seq_b[ib] and seq_a[ia] != "X":
                identical += 1

    n_matched = len(idx_a_list)
    idx_a = np.array(idx_a_list, dtype=int)
    idx_b = np.array(idx_b_list, dtype=int)

    # Mismatches = matched but not identical (and not X)
    n_mismatches = n_matched - identical

    n_gaps = (n_a - n_matched) + (n_b - n_matched)

    seq_identity = identical / n_matched if n_matched > 0 else 0.0

    if n_matched < min_matched:
        return ValidationResult(
            valid=False,
            reason=(
                f"Only {n_matched} residues matched by global alignment "
                f"(minimum: {min_matched}).  "
                "The structures may represent different proteins or severely truncated "
                "regions.  Verify that both structures correspond to the same UniProt entry."
            ),
            warnings=warns,
            n_residues_a=n_a,
            n_residues_b=n_b,
            n_matched=n_matched,
            n_mismatches=n_mismatches,
            n_gaps=n_gaps,
            seq_identity=seq_identity,
            x_frac_a=x_frac_a,
            x_frac_b=x_frac_b,
        )

    coverage_a = n_matched / n_a
    coverage_b = n_matched / n_b
    coverage = n_matched / min(
        n_a, n_b
    )  # retained for reporting/backward compatibility
    min_pair_coverage = min(coverage_a, coverage_b)

    if min_pair_coverage < coverage_min:
        return ValidationResult(
            valid=False,
            reason=(
                f"Pair coverage is below threshold {coverage_min:.0%}. "
                f"Matched {n_matched} residues "
                f"(A: {n_a} residues, coverage_a={coverage_a:.1%}; "
                f"B: {n_b} residues, coverage_b={coverage_b:.1%}).  "
                "Both structures must cover the same protein region sufficiently. "
                "Possible causes: truncation in one structure, domain mismatch, "
                "or different processing (mature chain vs preproprotein)."
            ),
            warnings=warns,
            n_residues_a=n_a,
            n_residues_b=n_b,
            n_matched=n_matched,
            n_mismatches=n_mismatches,
            n_gaps=n_gaps,
            coverage_a=coverage_a,
            coverage_b=coverage_b,
            coverage=coverage,
            seq_identity=seq_identity,
            x_frac_a=x_frac_a,
            x_frac_b=x_frac_b,
        )

    if seq_identity < seq_identity_min:
        return ValidationResult(
            valid=False,
            reason=(
                f"Sequence identity {seq_identity:.1%} is below threshold "
                f"{seq_identity_min:.0%} "
                f"({identical} identical / {n_matched} matched / {n_mismatches} mismatched).  "
                "The structures may represent different isoforms, species orthologs, "
                "or mutant variants rather than the same protein."
            ),
            warnings=warns,
            n_residues_a=n_a,
            n_residues_b=n_b,
            n_matched=n_matched,
            n_mismatches=n_mismatches,
            n_gaps=n_gaps,
            coverage_a=coverage_a,
            coverage_b=coverage_b,
            coverage=coverage,
            seq_identity=seq_identity,
            x_frac_a=x_frac_a,
            x_frac_b=x_frac_b,
        )

    size_ratio = min(n_a, n_b) / max(n_a, n_b)
    if size_ratio < 0.70:
        warns.append(
            f"Size asymmetry: {n_a} vs {n_b} residues (ratio {size_ratio:.2f}).  "
            "One structure may cover a longer region (e.g. signal peptide or propeptide "
            "present in the predicted but not experimental structure).  "
            "Metrics operate on matched residues only; the hard coverage check above "
            "requires sufficient coverage of both structures."
        )

    _INTERNAL_GAP_WARN = 10
    aligned_blocks_a = alignment.aligned[0]
    aligned_blocks_b = alignment.aligned[1]
    if len(aligned_blocks_a) > 1:
        internal_gaps_a = [
            aligned_blocks_a[i + 1][0] - aligned_blocks_a[i][1]
            for i in range(len(aligned_blocks_a) - 1)
        ]
        internal_gaps_b = [
            aligned_blocks_b[i + 1][0] - aligned_blocks_b[i][1]
            for i in range(len(aligned_blocks_b) - 1)
        ]
        max_internal_gap = max(max(internal_gaps_a), max(internal_gaps_b))
        n_blocks = len(aligned_blocks_a)
        if max_internal_gap > _INTERNAL_GAP_WARN:
            warns.append(
                f"Long internal gap: the global alignment contains {n_blocks} matched "
                f"blocks with a maximum internal gap of {max_internal_gap} consecutive "
                f"unmatched residues.  Metrics run on matched residues only, but w-rdist "
                "distance distributions may not represent comparable structural scopes "
                "if a domain or loop is present in one structure but absent in the other.  "
                "Consider excluding this pair from primary analysis or interpreting "
                "its metrics with caution."
            )

    matched_set_a = set(idx_a.tolist())
    matched_set_b = set(idx_b.tolist())

    phi_psi_a: np.ndarray = np.empty((0, 2))
    phi_psi_b: np.ndarray = np.empty((0, 2))

    try:
        angles_a_full, ca_idx_a = get_phi_psi_with_index(struct_a, model_idx_a, cid_a)
        if len(angles_a_full) > 0:
            mask_a = np.array([i in matched_set_a for i in ca_idx_a], dtype=bool)
            phi_psi_a = angles_a_full[mask_a]
    except Exception as exc:
        warns.append(
            f"Could not compute phi/psi for structure A ({path_a}): {exc}.  "
            "b_phipsi metric will not be available for this pair."
        )

    try:
        angles_b_full, ca_idx_b = get_phi_psi_with_index(struct_b, model_idx_b, cid_b)
        if len(angles_b_full) > 0:
            mask_b = np.array([i in matched_set_b for i in ca_idx_b], dtype=bool)
            phi_psi_b = angles_b_full[mask_b]
    except Exception as exc:
        warns.append(
            f"Could not compute phi/psi for structure B ({path_b}): {exc}.  "
            "b_phipsi metric will not be available for this pair."
        )

    all_atom_a: Optional[np.ndarray] = None
    all_atom_b: Optional[np.ndarray] = None
    aligned_set_a = set(idx_a.tolist())
    aligned_set_b = set(idx_b.tolist())
    try:
        all_atom_a = get_all_atom_coords(
            struct_a, model_idx_a, cid_a, ca_index_filter=aligned_set_a
        )
        if len(all_atom_a) == 0:
            all_atom_a = None
    except Exception as exc:
        warns.append(
            f"Could not load all-atom coordinates for structure A ({path_a}): {exc}.  "
            "t_alpha metric will not be available for this pair."
        )
    try:
        all_atom_b = get_all_atom_coords(
            struct_b, model_idx_b, cid_b, ca_index_filter=aligned_set_b
        )
        if len(all_atom_b) == 0:
            all_atom_b = None
    except Exception as exc:
        warns.append(
            f"Could not load all-atom coordinates for structure B ({path_b}): {exc}.  "
            "t_alpha metric will not be available for this pair."
        )

    return ValidationResult(
        valid=True,
        reason="OK",
        warnings=warns,
        # residue counts
        n_residues_a=n_a,
        n_residues_b=n_b,
        n_matched=n_matched,
        n_mismatches=n_mismatches,
        n_gaps=n_gaps,
        # coverage
        coverage_a=coverage_a,
        coverage_b=coverage_b,
        coverage=coverage,
        # sequence quality
        seq_identity=seq_identity,
        x_frac_a=x_frac_a,
        x_frac_b=x_frac_b,
        # alignment indices
        idx_a=idx_a,
        idx_b=idx_b,
        # pre-parsed arrays for metric reuse
        coords_a=coords_a,
        coords_b=coords_b,
        seq_a=seq_a,
        seq_b=seq_b,
        phi_psi_a=phi_psi_a,
        phi_psi_b=phi_psi_b,
        all_atom_coords_a=all_atom_a,
        all_atom_coords_b=all_atom_b,
    )


def validate_and_report(
    path_a: str,
    path_b: str,
    label: str = "",
    **kwargs,
) -> ValidationResult:
    """
    Convenience wrapper: validate and print a one-line summary to stdout.

    Parameters
    ----
    path_a, path_b : mmCIF file paths.
    label          : optional description (e.g. "Lysozyme xray_vs_nmr").
    **kwargs       : forwarded to validate_pair.

    Returns the ValidationResult unchanged.
    """
    result = validate_pair(path_a, path_b, **kwargs)
    tag = f"[{label}] " if label else ""

    if result.valid:
        print(
            f"  ✓ {tag}PASS — "
            f"{result.n_matched} matched residues, "
            f"coverage A/B {result.coverage_a:.1%}/{result.coverage_b:.1%}, "
            f"identity {result.seq_identity:.1%}, "
            f"mismatches {result.n_mismatches}, gaps {result.n_gaps}"
        )
    else:
        print(f"  ✗ {tag}FAIL — {result.reason}")

    for w in result.warnings:
        warnings.warn(f"{tag}{w}", stacklevel=2)

    return result
