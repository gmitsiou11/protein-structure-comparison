"""
Sequence relationship between two structures of a pair.

This module does not decide whether two structures may be compared: the
Machaon metrics (b-phipsi, w-rdist, t-alpha) are alignment-free and are
computed for every readable pair from each structure's whole chain
(see metrics.py).  Here the two chains are aligned once, globally, only to

  1. describe the pair (matched residues, coverage, sequence identity),
     reported with every comparison record, and
  2. provide the residue pairs used by RMSD — the one correspondence-based
     number — when that pairing is reliable (``rmsd_reliable``).

A pair is marked ``valid=False`` only when a file cannot be read or a chain
has fewer than 5 Cα residues.  Low coverage, low identity or few matched
residues produce warnings, not rejections.

Thresholds
----------
  RMSD_IDENTITY_MIN = 0.50 — below this the two chains are probably not the
                             same protein (e.g. a wrong chain identifier):
                             warning, no RMSD.  Point mutants and engineered
                             variants pair correctly and keep their RMSD.
  COVERAGE_MIN      = 0.80 — below this the two chains cover different parts
                             of the protein (warning).
  X_RESIDUE_WARN    = 0.10 — more than 10 % unknown residues ("X"): warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from Bio import Align

from src.parser import load_structure, get_ca_coords_and_seq

RMSD_IDENTITY_MIN: float = 0.50
COVERAGE_MIN: float = 0.80
X_RESIDUE_WARN: float = 0.10


@dataclass
class PairAlignment:
    """
    Description of a structure pair: readability, alignment statistics and,
    when ``valid=True``, the residue pairs used by RMSD.

    Fields — status
    ---------------
    valid          : bool   — True iff both chains could be read (>= 5 Cα each).
    reason         : str    — Human-readable explanation (used in output JSON).
    warnings       : list   — Non-fatal observations (low coverage, low identity,
                              high X fraction, ...).
    rmsd_reliable  : bool   — True iff seq_identity >= rmsd_identity_min (and at
                              least 3 residues are paired, the minimum for a
                              superposition): the pairs are the same residues.

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
    coverage       : float  — n_matched / min(n_a, n_b)  (coverage of the shorter chain).

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
    """

    valid: bool
    reason: str
    warnings: list[str] = field(default_factory=list)
    rmsd_reliable: bool = False

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

    def matched_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the sequence-aligned Cα coordinate pairs.

        Returns
        -------
        (ca_a_matched, ca_b_matched) — both of shape (n_matched, 3).

        Raises ValueError if called on an unreadable pair (valid=False).
        """
        if (
            self.coords_a is None
            or self.coords_b is None
            or self.idx_a is None
            or self.idx_b is None
        ):
            raise ValueError(
                "matched_coords() called on an unreadable or unpopulated PairAlignment. "
                "Check .valid before calling."
            )
        return self.coords_a[self.idx_a], self.coords_b[self.idx_b]

    def to_record(self) -> dict:
        """
        Return a JSON-serialisable summary of the alignment fields.

        Excludes the numpy arrays (coords, alignment indices).  Embedded in every
        comparison record, so the output is self-documenting.
        """
        return {
            "valid": self.valid,
            "reason": self.reason,
            "warnings": self.warnings,
            "rmsd_reliable": self.rmsd_reliable,
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


def align_pair(
    path_a: str,
    path_b: str,
    model_idx_a: int = 0,
    model_idx_b: int = 0,
    chain_id: str = "A",
    chain_id_a: Optional[str] = None,
    chain_id_b: Optional[str] = None,
    rmsd_identity_min: float = RMSD_IDENTITY_MIN,
    coverage_min: float = COVERAGE_MIN,
) -> PairAlignment:
    """
    Describe the sequence relationship of structures A and B.

    Both chains are read and aligned once (global alignment).  The result
    reports matched residues, coverage and identity for the output record and
    keeps the residue pairs that RMSD uses.  It never blocks the Machaon
    metrics: only an unreadable file or a chain with < 5 Cα gives valid=False.

    Parameters
    ----------
    path_a, path_b    : paths to mmCIF files.
    model_idx_a/b     : model indices (0 for X-ray / Cryo-EM / AF;
                        the chosen model for NMR structures, see nmr.py).
    chain_id          : chain read from both structures when chain_id_a/b are not given.
    chain_id_a        : chain to extract from structure A (overrides chain_id).
    chain_id_b        : chain to extract from structure B (overrides chain_id).
                        This enables comparing multi-chain experimental structures
                        (e.g. hemoglobin chain B) against AF2 predictions (chain A).
    rmsd_identity_min : identity below this -> warning, no RMSD.
    coverage_min      : per-structure coverage below this -> warning.

    Returns
    -------
    PairAlignment with:
      - .valid (False only if a chain cannot be read)
      - .reason ("OK" or why the pair could not be read)
      - .warnings, .rmsd_reliable and all coverage/identity fields
      - Cα coords and alignment indices (only when valid=True)
    """
    # Per-structure chain IDs; chain_id when not given.
    cid_a: str = chain_id_a if chain_id_a is not None else chain_id
    cid_b: str = chain_id_b if chain_id_b is not None else chain_id

    warns: list[str] = []

    try:
        struct_a = load_structure(path_a)
        coords_a, seq_a = get_ca_coords_and_seq(struct_a, model_idx_a, cid_a)
    except Exception as exc:
        return PairAlignment(
            valid=False,
            reason=f"Failed to parse structure A ({path_a}): {exc}",
        )

    try:
        struct_b = load_structure(path_b)
        coords_b, seq_b = get_ca_coords_and_seq(struct_b, model_idx_b, cid_b)
    except Exception as exc:
        return PairAlignment(
            valid=False,
            reason=f"Failed to parse structure B ({path_b}): {exc}",
            n_residues_a=len(coords_a),
        )

    n_a, n_b = len(coords_a), len(coords_b)

    if n_a < 5:
        return PairAlignment(
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
        return PairAlignment(
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
        return PairAlignment(
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

    coverage_a = n_matched / n_a
    coverage_b = n_matched / n_b
    coverage = n_matched / min(n_a, n_b)

    if min(coverage_a, coverage_b) < coverage_min:
        warns.append(
            f"Low coverage: {n_matched} matched residues "
            f"(A: {n_a} residues, coverage_a={coverage_a:.1%}; "
            f"B: {n_b} residues, coverage_b={coverage_b:.1%}).  "
            "The two chains cover different parts of the protein; "
            "whole-chain Machaon metrics include this difference."
        )
    if seq_identity < rmsd_identity_min:
        warns.append(
            f"Sequence identity {seq_identity:.1%} below {rmsd_identity_min:.0%} "
            f"({identical} identical / {n_matched} matched): the two chains are "
            "probably not the same protein.  RMSD not computed."
        )

    # a superposition needs at least three paired points
    rmsd_reliable = n_matched >= 3 and seq_identity >= rmsd_identity_min

    size_ratio = min(n_a, n_b) / max(n_a, n_b)
    if size_ratio < 0.70:
        warns.append(
            f"Size asymmetry: {n_a} vs {n_b} residues (ratio {size_ratio:.2f}).  "
            "One structure may cover a longer region (e.g. signal peptide or propeptide "
            "present in the predicted but not experimental structure).  "
            "RMSD uses matched residues only; the Machaon metrics use the whole "
            "chains, so they include this difference."
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
                f"unmatched residues: a domain or loop is present in one structure "
                "but absent in the other.  RMSD uses matched residues only; the "
                "whole-chain Machaon metrics include this difference."
            )

    return PairAlignment(
        valid=True,
        reason="OK",
        warnings=warns,
        rmsd_reliable=rmsd_reliable,
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
        # Cα coordinates (RMSD uses coords[idx])
        coords_a=coords_a,
        coords_b=coords_b,
        seq_a=seq_a,
        seq_b=seq_b,
    )


def align_and_report(
    path_a: str,
    path_b: str,
    label: str = "",
    **kwargs,
) -> PairAlignment:
    """
    Convenience wrapper: align and print a one-line summary to stdout.

    Parameters
    ----------
    path_a, path_b : mmCIF file paths.
    label          : optional description (e.g. "Lysozyme xray_vs_nmr").
    **kwargs       : forwarded to align_pair.

    Returns the PairAlignment unchanged.
    """
    result = align_pair(path_a, path_b, **kwargs)
    tag = f"[{label}] " if label else ""

    if result.valid:
        print(
            f"  [ok] {tag}{result.n_matched} matched residues, "
            f"coverage A/B {result.coverage_a:.1%}/{result.coverage_b:.1%}, "
            f"identity {result.seq_identity:.1%}, "
            f"RMSD {'yes' if result.rmsd_reliable else 'no'}"
        )
    else:
        print(f"  [unreadable] {tag}{result.reason}")

    # printed rather than raised as Python warnings: the text is already in the record,
    # and a warning header would print the absolute path of the calling file
    for w in result.warnings:
        print(f"  [warn] {tag}{w}")

    return result
