"""
Pipeline orchestrator, is the primary entry point for a full scientifically valid run.
It:
  1. Validates each structure pair (via ``validation.py``).
  2. Extracts structural metadata (method, resolution, pLDDT, etc.).
  3. Computes all metrics from pre-validated data — NO re-alignment.
  4. Assigns a comparison category (exp_vs_exp, exp_vs_af2, …).
  5. Emits a self-documenting JSON record for every comparison.
  6. Runs the statistical analysis layer over all valid records.
  7. Emits a rejection record (with reason) for every failed pair.


NMR ensemble handling
----------------------
NMR structures deposit multiple models (conformers) representing the
conformational ensemble sampled in solution.  This pipeline:

  1. Uses the representative model (``nmr_model`` from proteins.csv) for
     all NMR-vs-experimental and NMR-vs-AI comparisons.
  2. Separately computes INTRA-ensemble variability by comparing model 0
     against ALL other deposited models.
  3. Reports mean ± std across all model-0-vs-model-k comparisons as the
     "NMR flexibility baseline".

The NMR baseline is included as a reference point, not a benchmark to be beaten.

"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import traceback
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

import gemmi as _gemmi
import scipy as _scipy
import Bio as _Bio

from src.metrics import compute_all_metrics
from src.metadata import extract_metadata, plddt_on_matched_residues, StructureMetadata
from src.validation import validate_pair, validate_and_report, ValidationResult
from src.analysis import analyse, infer_category, summarise_rejections

_REQUIRED_CSV_COLUMNS = {
    "protein_name",
    "uniprot_id",
    "pdb_xray",
    "pdb_nmr",
    "pdb_cryoem",
    "chain_id",
    "nmr_model",
}


def _file_md5(path: str) -> str:
    """Compute the MD5 checksum of a file for reproducibility tracking."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_run_metadata(cif_paths: set[str]) -> dict:
    """
    Build a provenance record for the current pipeline run.

    Includes timestamp, library versions, and MD5 checksums of all input
    CIF files so that re-runs with silently updated structures can be detected.
    """
    return {
        "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "library_versions": {
            "gemmi": _gemmi.__version__,
            "numpy": np.__version__,
            "scipy": _scipy.__version__,
            "biopython": _Bio.__version__,
        },
        "input_checksums": {
            os.path.basename(p): _file_md5(p)
            for p in sorted(cif_paths)
            if os.path.exists(p)
        },
    }


def _exp_path(pdb_id: str, data_dir: str = "data") -> Optional[str]:
    if not pdb_id or str(pdb_id).strip().lower() in ("nan", ""):
        return None
    path = os.path.join(data_dir, "experimental", f"{pdb_id.upper()}.cif")
    return path if os.path.exists(path) else None


def _af2_path(uniprot_id: str, data_dir: str = "data") -> Optional[str]:
    path = os.path.join(data_dir, "alphafold2", f"AF2_{uniprot_id}.cif")
    return path if os.path.exists(path) else None


def _of3_path(uniprot_id: str, data_dir: str = "data") -> Optional[str]:
    path = os.path.join(data_dir, "openfold3", f"OF3_{uniprot_id}.cif")
    return path if os.path.exists(path) else None


def _expected_path(kind: str, identifier: str, data_dir: str = "data") -> Optional[str]:
    """Return the canonical expected mmCIF path without checking existence.

    This is used for explicit missing-input reporting.  The dataset manifest
    remains the single source of truth for which identifiers are expected.
    """
    if identifier is None or str(identifier).strip().lower() in ("nan", ""):
        return None

    identifier = str(identifier).strip()

    if kind == "experimental":
        return os.path.join(data_dir, "experimental", f"{identifier.upper()}.cif")

    if kind == "af2":
        return os.path.join(data_dir, "alphafold2", f"AF2_{identifier}.cif")

    if kind == "of3":
        return os.path.join(data_dir, "openfold3", f"OF3_{identifier}.cif")

    raise ValueError(f"Unknown structure kind: {kind}")


def compare_pair(
    protein_name: str,
    uniprot_id: str,
    path_a: str,
    path_b: str,
    method_cat_a: str,
    method_cat_b: str,
    chain_id: str = "A",
    chain_id_a: Optional[str] = None,
    chain_id_b: Optional[str] = None,
    model_idx_a: int = 0,
    model_idx_b: int = 0,
    pdb_id_a: str = "",
    pdb_id_b: str = "",
    notes: str = "",
    verbose: bool = True,
) -> dict:
    """
    Validate and compare two structures, returning a self-documenting record.

    This function ALWAYS returns a dict.  The ``"status"`` field indicates the
    outcome:
      - ``"ok"``            — comparison succeeded; all metric fields populated.
      - ``"rejected"``      — failed validation; ``"rejection_reason"`` explains why.
      - ``"metric_error"``  — passed validation but metric computation failed;
                              ``"rejection_reason"`` contains the traceback.

    Every returned dict includes validation metadata so results are
    self-documenting even for failed comparisons.

    The returned dict contains:
      - Identification (protein_name, uniprot_id, comparison tag, category)
      - Structural scope tag (comparison_scope = "chain_level")
      - is_intra_ensemble flag (False for regular comparisons)
      - Status and rejection reason
      - Full validation summary (n_matched, coverage_a, coverage_b, seq_identity,
        n_mismatches, n_gaps, x_frac_a, x_frac_b, warnings)
      - Metadata for both structures (method, resolution, pLDDT, plddt_confirmed)
      - All metric values (when status = "ok")
    """
    label = f"{protein_name} | {method_cat_a} vs {method_cat_b}"
    if verbose:
        print(f"\n  Comparing: {label}")

    category = infer_category(method_cat_a, method_cat_b)

    # Resolve per-structure chain IDs.
    cid_a: str = chain_id_a if chain_id_a is not None else chain_id
    cid_b: str = chain_id_b if chain_id_b is not None else chain_id

    base: dict = {
        "protein_name": protein_name,
        "uniprot_id": uniprot_id,
        "comparison": f"{method_cat_a}_vs_{method_cat_b}",
        "category": category,
        "comparison_scope": "chain_level",
        "chain_id": chain_id,
        "chain_id_a": cid_a,
        "chain_id_b": cid_b,
        "is_intra_ensemble": False,
        "status": "ok",
        "rejection_reason": None,
        "notes": notes,
    }

    val: ValidationResult = validate_and_report(
        path_a,
        path_b,
        label=label,
        model_idx_a=model_idx_a,
        model_idx_b=model_idx_b,
        chain_id_a=cid_a,
        chain_id_b=cid_b,
    )

    base.update(val.to_record())

    if not val.valid:
        base["status"] = "rejected"
        base["rejection_reason"] = val.reason
        try:
            meta_a = extract_metadata(
                path_a,
                uniprot_id=uniprot_id,
                pdb_id=pdb_id_a,
                chain_id=cid_a,
                model_idx=model_idx_a,
            )
            base["method_a"] = meta_a.method_category
            base["pdb_id_a"] = meta_a.pdb_id
            base["model_idx_a"] = model_idx_a
        except Exception:
            base["method_a"] = method_cat_a
            base["pdb_id_a"] = pdb_id_a
            base["model_idx_a"] = model_idx_a
        try:
            meta_b = extract_metadata(
                path_b,
                uniprot_id=uniprot_id,
                pdb_id=pdb_id_b,
                chain_id=cid_b,
                model_idx=model_idx_b,
            )
            base["method_b"] = meta_b.method_category
            base["pdb_id_b"] = meta_b.pdb_id
            base["model_idx_b"] = model_idx_b
        except Exception:
            base["method_b"] = method_cat_b
            base["pdb_id_b"] = pdb_id_b
            base["model_idx_b"] = model_idx_b
        return base

    meta_a: StructureMetadata = extract_metadata(
        path_a,
        uniprot_id=uniprot_id,
        pdb_id=pdb_id_a,
        chain_id=cid_a,
        model_idx=model_idx_a,
    )
    meta_b: StructureMetadata = extract_metadata(
        path_b,
        uniprot_id=uniprot_id,
        pdb_id=pdb_id_b,
        chain_id=cid_b,
        model_idx=model_idx_b,
    )

    plddt_matched_a = (
        plddt_on_matched_residues(meta_a, val.idx_a) if val.idx_a is not None else None
    )
    plddt_matched_b = (
        plddt_on_matched_residues(meta_b, val.idx_b) if val.idx_b is not None else None
    )

    try:
        metric_values = compute_all_metrics(val)
    except Exception as exc:
        if verbose:
            print(f"    [ERROR] Metric computation failed: {exc}")
            traceback.print_exc()
        base["status"] = "metric_error"
        base["rejection_reason"] = f"Metric computation failed: {exc}"
        _embed_metadata(
            base,
            meta_a,
            meta_b,
            model_idx_a,
            model_idx_b,
            plddt_matched_a,
            plddt_matched_b,
        )
        return base

    _embed_metadata(
        base, meta_a, meta_b, model_idx_a, model_idx_b, plddt_matched_a, plddt_matched_b
    )

    base.update(
        {
            "rmsd": round(metric_values["rmsd"], 6),
            "t_alpha": (
                round(metric_values["t_alpha"], 6)
                if metric_values["t_alpha"] is not None
                else None
            ),
            "t_alpha_n_tri_a": metric_values.get("t_alpha_n_tri_a"),
            "t_alpha_n_tri_b": metric_values.get("t_alpha_n_tri_b"),
            "w_rdist_raw": round(metric_values["w_rdist_raw"], 6),
            "w_rdist_norm": round(metric_values["w_rdist_norm"], 6),
            "b_phipsi": (
                round(metric_values["b_phipsi"], 6)
                if metric_values["b_phipsi"] is not None
                else None
            ),
            "b_phipsi_n_a": metric_values["b_phipsi_n_a"],
            "b_phipsi_n_b": metric_values["b_phipsi_n_b"],
        }
    )

    return base


def _embed_metadata(
    base: dict,
    meta_a: StructureMetadata,
    meta_b: StructureMetadata,
    model_idx_a: int,
    model_idx_b: int,
    plddt_matched_a: Optional[float],
    plddt_matched_b: Optional[float],
) -> None:
    """Embed metadata for both structures into the record dict (in-place)."""
    base.update(
        {
            "method_a": meta_a.method_category,
            "pdb_id_a": meta_a.pdb_id,
            "model_idx_a": model_idx_a,
            "resolution_a": meta_a.resolution,
            "n_models_a": meta_a.n_models,
            "is_af_a": meta_a.is_alphafold,
            "af_version_a": meta_a.af_version,
            "plddt_mean_a": meta_a.plddt_mean,
            "plddt_confirmed_a": meta_a.plddt_source_confirmed,
            "plddt_note_a": meta_a.plddt_note or None,
            "plddt_matched_a": (
                round(plddt_matched_a, 2) if plddt_matched_a is not None else None
            ),
            "method_b": meta_b.method_category,
            "pdb_id_b": meta_b.pdb_id,
            "model_idx_b": model_idx_b,
            "resolution_b": meta_b.resolution,
            "n_models_b": meta_b.n_models,
            "is_af_b": meta_b.is_alphafold,
            "af_version_b": meta_b.af_version,
            "plddt_mean_b": meta_b.plddt_mean,
            "plddt_confirmed_b": meta_b.plddt_source_confirmed,
            "plddt_note_b": meta_b.plddt_note or None,
            "plddt_matched_b": (
                round(plddt_matched_b, 2) if plddt_matched_b is not None else None
            ),
            "organism": meta_a.organism or meta_b.organism,
        }
    )


def compute_nmr_ensemble_variability(
    protein_name: str,
    uniprot_id: str,
    nmr_path: str,
    chain_id: str = "A",
    representative_model: int = 0,
    verbose: bool = True,
) -> list[dict]:
    """
    Quantify within-ensemble variability for an NMR structure.

    Compares the manifest-selected representative model against all other
    deposited models.
    """
    from src.parser import load_structure

    try:
        struct = load_structure(nmr_path)
        n_models = len(struct)
    except Exception as exc:
        if verbose:
            print(
                f"    [WARN] Could not load NMR structure for ensemble variability: {exc}"
            )
        return []

    if n_models < 2:
        if verbose:
            print(
                f"    [INFO] {protein_name}: NMR structure has only {n_models} model; "
                "intra-ensemble variability not computed."
            )
        return []

    if representative_model < 0 or representative_model >= n_models:
        raise ValueError(
            f"Invalid representative NMR model {representative_model} for "
            f"{protein_name}; structure has {n_models} models."
        )

    if verbose:
        print(
            f"\n  NMR intra-ensemble variability: {protein_name} "
            f"({n_models} models, comparing model {representative_model} "
            f"vs all other models)"
        )

    records = []

    for model_b in range(n_models):
        if model_b == representative_model:
            continue

        rec = compare_pair(
            protein_name=protein_name,
            uniprot_id=uniprot_id,
            path_a=nmr_path,
            path_b=nmr_path,
            method_cat_a="NMR",
            method_cat_b="NMR",
            chain_id=chain_id,
            model_idx_a=representative_model,
            model_idx_b=model_b,
            pdb_id_a="",
            pdb_id_b="",
            notes=(
                "Intra-ensemble comparison: quantifies NMR conformational flexibility baseline. "
                "Interpret separately from experimental-vs-AI comparisons."
            ),
            verbose=verbose,
        )

        rec["comparison"] = f"nmr_intra_model{representative_model}_vs_model{model_b}"
        rec["category"] = "nmr_intra_ensemble"
        rec["is_intra_ensemble"] = True
        rec["nmr_flexibility_note"] = (
            "NMR intra-ensemble variability reflects solution-state conformational flexibility "
            "— physically real molecular motions, not measurement error. "
            "This is FUNDAMENTALLY DIFFERENT from X-ray/Cryo-EM single-conformation "
            "uncertainty or AI prediction error. Use as a reference baseline only."
        )

        records.append(rec)

    return records


def aggregate_nmr_variability(
    intra_records: list[dict],
    protein_name: str,
) -> dict:
    """
    Aggregate intra-NMR comparison records into mean ± std statistics.

    This summary is the "NMR flexibility baseline" for the protein: the
    typical within-ensemble deviation across all metric dimensions.

    Returns
    -------
    Dict with:
      protein_name, nmr_baseline_available, n_model_comparisons,
      interpretation note, per-metric mean/std/min/max.
    """
    ok_records = [r for r in intra_records if r.get("status") == "ok"]
    all_records_n = len(intra_records)
    ok_n = len(ok_records)

    metric_names = [
        "rmsd",
        "t_alpha",
        "w_rdist_raw",
        "w_rdist_norm",
        "b_phipsi",
    ]

    summary: dict = {
        "protein_name": protein_name,
        "nmr_baseline_available": ok_n > 0,
        "n_model_pairs_attempted": all_records_n,
        "n_model_pairs_ok": ok_n,
        "interpretation": (
            "Mean ± std across all representative-model-vs-other-model comparisons within the NMR ensemble. "
            "This quantifies the conformational flexibility captured by the deposited models.  "
            "Values represent solution-state dynamics, not prediction error or measurement "
            "uncertainty.  Use as a reference point when interpreting AI-vs-experimental "
            "deviations: if AI metrics fall within the intra-NMR range for this protein, "
            "the AI prediction is compatible with one of the accessible conformations."
        ),
        "metrics": {},
    }

    for m in metric_names:
        vals = [
            r[m]
            for r in ok_records
            if r.get(m) is not None and not np.isnan(float(r.get(m, float("nan"))))
        ]
        if vals:
            arr = np.array(vals, dtype=float)
            summary["metrics"][m] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "n": len(arr),
            }

    return summary


_PIPELINE_METADATA = {
    "comparison_scope": (
        "Chain-level single polypeptide chain.  AlphaFold predictions are per-UniProt "
        "sequence, matching this scope.  Biological assembly-level conclusions "
        "(oligomeric states, inter-chain interactions) cannot be drawn from these results.  "
        "Each record carries comparison_scope = 'chain_level' as a short tag."
    ),
    "statistical_interpretation_note": (
        "Given only 1–3 exp-vs-exp comparison pairs per protein, classical hypothesis "
        "tests (t-test, ANOVA) are severely underpowered and are not reported.  "
        "Cohen's d and variance ratios are presented as DESCRIPTIVE effect sizes only.  "
        "The within-range check (AI value within exp_vs_exp ± 1σ) is an OPERATIONAL "
        "COMPARISON, not a statistical test.  Finding that AI metrics fall within "
        "experimental variability does NOT prove that AI replaces experiments — it is "
        "a necessary but not sufficient condition. The actual protein count (N) is "
        "reported in the analysis summary_text field."
    ),
    "plddt_note": (
        "AF2 pLDDT is extracted from B_iso_or_equiv (well-documented convention). "
        "OF3 pLDDT is extracted the same way but range-validated; if values fall outside "
        "[0, 100], confidence is set to None and the plddt_note field explains why. "
        "pLDDT is not meaningful for experimental structures (field stores B-factors there)."
    ),
    "nmr_intra_ensemble_note": (
        "NMR intra-ensemble records (is_intra_ensemble=True) are included in "
        "all_metrics_validated.json for completeness but are EXCLUDED from the main "
        "statistical analysis.  They are reported separately in nmr_variability.json.  "
        "Filter on category != 'nmr_intra_ensemble' or is_intra_ensemble == False "
        "to obtain only the primary comparison records."
    ),
}


def run_pipeline(
    proteins_csv: str = "proteins.csv",
    data_dir: str = "data",
    output_dir: str = "results",
    verbose: bool = True,
    include_nmr_ensemble: bool = True,
) -> dict:
    """
    Run the full validated comparison pipeline.

    Parameters
    ----
    proteins_csv          : path to the dataset manifest.
    data_dir              : root directory with experimental/, alphafold2/, openfold3/.
    output_dir            : where to write all output JSON files.
    verbose               : print progress to stdout.
    include_nmr_ensemble  : whether to compute intra-NMR ensemble variability.

    Returns
    -------
    Dict with:
      "comparisons"              : list of all records (ok + intra-ensemble)
      "rejected"                 : list of rejected/failed records (with reasons)
      "nmr_variability"          : per-protein NMR flexibility baselines
      "analysis"                 : statistical analysis output
      "per_protein_summary"      : per-protein interpretation summaries
      "pipeline_metadata"        : documentation of scope, exclusions, caveats
      "run_metadata"             : timestamp, library versions, input checksums
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(proteins_csv)

    missing_cols = _REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"proteins.csv is missing required columns: {missing_cols}.  "
            f"Found columns: {list(df.columns)}"
        )

    all_records: list[dict] = []
    rejected: list[dict] = []
    nmr_variability: dict[str, dict] = {}
    missing_inputs: list[dict] = []
    all_cif_paths: set[str] = set()

    for _, row in df.iterrows():
        protein_name = row["protein_name"]
        uniprot_id = row["uniprot_id"]
        chain_id = str(row.get("chain_id", "A")).strip()
        # chain_id_af2: chain to use for AF2/OF3 structures.  AF2/OF3 always
        # deposit as a single chain labelled "A", so the default is always "A".
        # Experimental structures in multi-subunit complexes may use a different
        # chain letter (e.g. "B" for hemoglobin beta), so chain_id stays as-is.
        _raw_af2_chain = row.get("chain_id_af2", None)
        if _raw_af2_chain is None or (
            isinstance(_raw_af2_chain, float) and pd.isna(_raw_af2_chain)
        ):
            chain_id_af2 = "A"
        else:
            chain_id_af2 = str(_raw_af2_chain).strip() or "A"

        def _chain_for(method: str) -> str:
            """Return the correct chain ID for the given method category."""
            return chain_id_af2 if method in ("AF2", "OF3") else chain_id

        nmr_model = (
            int(row.get("nmr_model", 0))
            if not pd.isna(row.get("nmr_model", float("nan")))
            else 0
        )

        xray_path = _exp_path(row.get("pdb_xray"), data_dir)
        nmr_path = _exp_path(row.get("pdb_nmr"), data_dir)
        cryoem_path = _exp_path(row.get("pdb_cryoem"), data_dir)
        af2_path = _af2_path(uniprot_id, data_dir)
        of3_path = _of3_path(uniprot_id, data_dir)

        expected_inputs = [
            ("experimental", "X-ray", row.get("pdb_xray")),
            ("experimental", "NMR", row.get("pdb_nmr")),
            ("experimental", "Cryo-EM", row.get("pdb_cryoem")),
            ("af2", "AF2", uniprot_id),
            ("of3", "OF3", uniprot_id),
        ]
        for kind, structure_type, identifier in expected_inputs:
            expected = _expected_path(kind, identifier, data_dir)
            if expected and not os.path.exists(expected):
                missing_inputs.append(
                    {
                        "protein_name": protein_name,
                        "uniprot_id": uniprot_id,
                        "pdb_id_or_uniprot": str(identifier),
                        "chain_id": chain_id,
                        "structure_type": structure_type,
                        "expected_path": expected,
                    }
                )

        notes_str = (
            str(row.get("notes", ""))
            if not pd.isna(row.get("notes", float("nan")))
            else ""
        )

        for p in (xray_path, nmr_path, cryoem_path, af2_path, of3_path):
            if p:
                all_cif_paths.add(p)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Protein: {protein_name}  ({uniprot_id})")
            print(f"  X-ray:   {xray_path}")
            print(f"  NMR:     {nmr_path}  (representative model: {nmr_model})")
            print(f"  Cryo-EM: {cryoem_path}")
            print(f"  AF2:     {af2_path}")
            print(f"  OF3:     {of3_path}")

        pairs = []

        if xray_path and nmr_path:
            pairs.append(
                (
                    xray_path,
                    nmr_path,
                    "X-ray",
                    "NMR",
                    0,
                    nmr_model,
                    row.get("pdb_xray", ""),
                    row.get("pdb_nmr", ""),
                )
            )
        if xray_path and cryoem_path:
            pairs.append(
                (
                    xray_path,
                    cryoem_path,
                    "X-ray",
                    "Cryo-EM",
                    0,
                    0,
                    row.get("pdb_xray", ""),
                    row.get("pdb_cryoem", ""),
                )
            )
        if nmr_path and cryoem_path:
            pairs.append(
                (
                    nmr_path,
                    cryoem_path,
                    "NMR",
                    "Cryo-EM",
                    nmr_model,
                    0,
                    row.get("pdb_nmr", ""),
                    row.get("pdb_cryoem", ""),
                )
            )
        if xray_path and af2_path:
            pairs.append(
                (xray_path, af2_path, "X-ray", "AF2", 0, 0, row.get("pdb_xray", ""), "")
            )
        if nmr_path and af2_path:
            pairs.append(
                (
                    nmr_path,
                    af2_path,
                    "NMR",
                    "AF2",
                    nmr_model,
                    0,
                    row.get("pdb_nmr", ""),
                    "",
                )
            )
        if cryoem_path and af2_path:
            pairs.append(
                (
                    cryoem_path,
                    af2_path,
                    "Cryo-EM",
                    "AF2",
                    0,
                    0,
                    row.get("pdb_cryoem", ""),
                    "",
                )
            )
        if xray_path and of3_path:
            pairs.append(
                (xray_path, of3_path, "X-ray", "OF3", 0, 0, row.get("pdb_xray", ""), "")
            )
        if nmr_path and of3_path:
            pairs.append(
                (
                    nmr_path,
                    of3_path,
                    "NMR",
                    "OF3",
                    nmr_model,
                    0,
                    row.get("pdb_nmr", ""),
                    "",
                )
            )
        if cryoem_path and of3_path:
            pairs.append(
                (
                    cryoem_path,
                    of3_path,
                    "Cryo-EM",
                    "OF3",
                    0,
                    0,
                    row.get("pdb_cryoem", ""),
                    "",
                )
            )
        if af2_path and of3_path:
            pairs.append((af2_path, of3_path, "AF2", "OF3", 0, 0, "", ""))

        for pa, pb, ma, mb, mia, mib, pdb_a, pdb_b in pairs:
            rec = compare_pair(
                protein_name=protein_name,
                uniprot_id=uniprot_id,
                path_a=pa,
                path_b=pb,
                method_cat_a=ma,
                method_cat_b=mb,
                chain_id=chain_id,
                chain_id_a=_chain_for(ma),
                chain_id_b=_chain_for(mb),
                model_idx_a=mia,
                model_idx_b=mib,
                pdb_id_a=str(pdb_a) if pdb_a else "",
                pdb_id_b=str(pdb_b) if pdb_b else "",
                notes=notes_str,
                verbose=verbose,
            )

            if rec["status"] == "ok":
                all_records.append(rec)
            else:
                rejected.append(rec)

        if include_nmr_ensemble and nmr_path:
            intra_records = compute_nmr_ensemble_variability(
                protein_name,
                uniprot_id,
                nmr_path,
                chain_id=chain_id,
                representative_model=nmr_model,
                verbose=verbose,
            )
            all_records.extend([r for r in intra_records if r.get("status") == "ok"])
            rejected.extend([r for r in intra_records if r.get("status") != "ok"])

            if intra_records:
                ok_intra = [r for r in intra_records if r.get("status") == "ok"]
                if ok_intra:
                    nmr_variability[protein_name] = aggregate_nmr_variability(
                        ok_intra, protein_name
                    )

        if protein_name not in nmr_variability:
            nmr_variability[protein_name] = {
                "nmr_baseline_available": False,
                "reason": (
                    "single-model NMR structure (no ensemble) or no NMR structure available"
                    if nmr_path
                    else "no NMR structure in dataset"
                ),
            }
        else:
            nmr_variability[protein_name]["nmr_baseline_available"] = True

    analysis_records = [
        r for r in all_records if r.get("category") != "nmr_intra_ensemble"
    ]
    analysis_result = analyse(analysis_records)

    if verbose:
        print("\n\n" + analysis_result["summary_text"])

    run_metadata = _build_run_metadata(all_cif_paths)

    missing_inputs_path = os.path.join(output_dir, "missing_inputs.csv")
    pd.DataFrame(missing_inputs).to_csv(missing_inputs_path, index=False)
    if verbose:
        print(
            f"Wrote {len(missing_inputs)} missing-input records to {missing_inputs_path}"
        )

    metrics_path = os.path.join(output_dir, "all_metrics_validated.json")
    with open(metrics_path, "w") as f:
        json.dump(all_records, f, indent=2, default=str)
    if verbose:
        print(f"\nWrote {len(all_records)} records to {metrics_path}")

    rejected_path = os.path.join(output_dir, "rejected_pairs.json")
    with open(rejected_path, "w") as f:
        json.dump(rejected, f, indent=2, default=str)
    if verbose:
        print(f"Wrote {len(rejected)} rejected pairs to {rejected_path}")

    rejection_summary = summarise_rejections(rejected)
    rejection_summary_path = os.path.join(output_dir, "rejection_summary.json")
    with open(rejection_summary_path, "w") as f:
        json.dump(
            {k: v for k, v in rejection_summary.items() if k != "printable_summary"},
            f,
            indent=2,
        )
    if verbose:
        print(f"\n{rejection_summary['printable_summary']}")

    analysis_path = os.path.join(output_dir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis_result, f, indent=2, default=str)
    if verbose:
        print(f"Wrote statistical analysis to {analysis_path}")

    nmr_path_out = os.path.join(output_dir, "nmr_variability.json")
    with open(nmr_path_out, "w") as f:
        json.dump(nmr_variability, f, indent=2, default=str)
    if verbose:
        print(f"Wrote NMR flexibility baselines to {nmr_path_out}")

    pipeline_output = {
        "comparisons": all_records,
        "rejected": rejected,
        "missing_inputs": missing_inputs,
        "nmr_variability": nmr_variability,
        "analysis": analysis_result,
        "per_protein_summary": analysis_result.get("per_protein_interpretation", {}),
        "pipeline_metadata": _PIPELINE_METADATA,
        "run_metadata": run_metadata,
    }

    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {k: v for k, v in pipeline_output.items() if k != "comparisons"},
            f,
            indent=2,
            default=str,
        )
    if verbose:
        print(f"Wrote pipeline summary to {summary_path}")

    return pipeline_output
