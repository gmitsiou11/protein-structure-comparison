"""
Pipeline orchestrator: the entry point for a full run.  For every protein it
  1. Describes each structure pair (alignment statistics, via ``alignment.py``).
  2. Extracts structural metadata (method, resolution, pLDDT, etc.).
  3. Summarises each structure once (whole chain) and computes the Machaon
     metrics from the two summaries — no alignment involved; RMSD only when
     the residue pairing is reliable.
  4. Assigns a comparison category (exp_vs_exp, exp_vs_af2, …).
  5. Emits a self-documenting JSON record for every comparison.
  6. Runs the statistical analysis layer over all computed records.
  7. Emits a rejection record (with reason) for every pair that cannot be read.


NMR ensemble handling
----------------------
NMR structures deposit multiple models (conformers) representing the
conformational ensemble sampled in solution.  This pipeline:

  1. Picks one model per NMR entry by a rule (``nmr_model_policy``, see
     src/nmr.py: the ensemble medoid by default, the model the PDB names as
     representative, or the first model) and uses it for all
     NMR-vs-experimental and NMR-vs-AI comparisons.
  2. Separately compares that representative model with every other model of
     its ensemble (category "nmr_intra_ensemble"): the spread of one NMR
     experiment, part flexibility in solution and part how loosely the
     restraints define the structure.  It is a reference level, not a
     benchmark to be beaten.
  3. Writes nmr_models.csv: the model used for each protein, the rule that
     chose it, and the alternatives (PDB-designated, medoid).

"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

import gemmi as _gemmi
import scipy as _scipy
import Bio as _Bio

from src.metrics import compute_all_metrics, summarise_structure
from src.metadata import extract_metadata, StructureMetadata
from src.alignment import align_pair, align_and_report, PairAlignment
from src.analysis import analyse, infer_category, summarise_rejections
from src.nmr import choose_nmr_model, POLICIES as NMR_POLICIES

_REQUIRED_CSV_COLUMNS = {
    "protein_name",
    "uniprot_id",
    "pdb_xray",
    "pdb_nmr",
    "pdb_cryoem",
    "chain_xray",
    "chain_nmr",
    "chain_cryoem",
}


def _file_md5(path: str) -> str:
    """Compute the MD5 checksum of a file for reproducibility tracking."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state() -> dict:
    """git_commit and git_dirty of the repository the code runs from (None outside git)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def git(*args):
        return subprocess.run(
            ["git", "-C", root, *args], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()

    try:
        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "src"))}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


def _build_run_metadata(cif_paths: set[str]) -> dict:
    """
    Build a provenance record for the current pipeline run.

    Includes timestamp, the git commit of the code (with a flag for uncommitted
    changes), library versions, and MD5 checksums of all input CIF files so that
    re-runs with silently updated structures can be detected.
    """
    return {
        "run_timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        **_git_state(),
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
    Compare two structures, returning a self-documenting record.

    This function always returns a dict.  The ``"status"`` field indicates the
    outcome:
      - ``"ok"``            — comparison succeeded; metric fields populated
                              (``rmsd`` is None when ``rmsd_reliable`` is False).
      - ``"rejected"``      — a structure could not be read; ``"rejection_reason"``
                              explains why.
      - ``"metric_error"``  — metric computation failed;
                              ``"rejection_reason"`` contains the traceback.

    Every returned dict includes the alignment summary, so results are
    self-documenting even for failed comparisons.

    The returned dict contains:
      - Identification (protein_name, uniprot_id, comparison tag, category)
      - Structural scope tag (comparison_scope = "chain_level")
      - is_intra_ensemble flag (False for regular comparisons)
      - Status and rejection reason
      - Alignment summary (n_matched, coverage_a, coverage_b, seq_identity,
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

    val: PairAlignment = align_and_report(
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

    try:
        summary_a = summarise_structure(path_a, model_idx_a, cid_a)
        summary_b = summarise_structure(path_b, model_idx_b, cid_b)
        metric_values = compute_all_metrics(summary_a, summary_b, val)
    except Exception as exc:
        if verbose:
            print(f"    [ERROR] Metric computation failed: {exc}")
            traceback.print_exc()
        base["status"] = "metric_error"
        base["rejection_reason"] = f"Metric computation failed: {exc}"
        _embed_metadata(base, meta_a, meta_b, model_idx_a, model_idx_b)
        return base

    _embed_metadata(base, meta_a, meta_b, model_idx_a, model_idx_b)

    base.update(
        {
            "rmsd": _round6(metric_values["rmsd"]),
            "t_alpha": _round6(metric_values["t_alpha"]),
            "t_alpha_n_tri_a": metric_values["t_alpha_n_tri_a"],
            "t_alpha_n_tri_b": metric_values["t_alpha_n_tri_b"],
            "w_rdist_raw": _round6(metric_values["w_rdist_raw"]),
            "w_rdist_norm": _round6(metric_values["w_rdist_norm"]),
            "b_phipsi": _round6(metric_values["b_phipsi"]),
            "b_phipsi_n_a": metric_values["b_phipsi_n_a"],
            "b_phipsi_n_b": metric_values["b_phipsi_n_b"],
        }
    )

    return base


def _round6(value: Optional[float]) -> Optional[float]:
    """Round a metric value to 6 decimals; keep None (metric not defined)."""
    return round(value, 6) if value is not None else None


def _embed_metadata(
    base: dict,
    meta_a: StructureMetadata,
    meta_b: StructureMetadata,
    model_idx_a: int,
    model_idx_b: int,
) -> None:
    """Embed metadata for both structures into the record dict (in-place)."""
    base.update(
        {
            "method_a": meta_a.method_category,
            "pdb_id_a": meta_a.pdb_id,
            "model_idx_a": model_idx_a,
            "resolution_a": meta_a.resolution,
            "deposition_date_a": meta_a.deposition_date,
            "release_date_a": meta_a.release_date,
            "title_a": meta_a.title or None,
            "n_polymer_chains_a": meta_a.n_polymer_chains,
            "n_models_a": meta_a.n_models,
            "is_af_a": meta_a.is_alphafold,
            "af_version_a": meta_a.af_version,
            "plddt_mean_a": meta_a.plddt_mean,
            "plddt_confirmed_a": meta_a.plddt_source_confirmed,
            "plddt_note_a": meta_a.plddt_note or None,
            "msa_depth_a": meta_a.msa_depth,
            "method_b": meta_b.method_category,
            "pdb_id_b": meta_b.pdb_id,
            "model_idx_b": model_idx_b,
            "resolution_b": meta_b.resolution,
            "deposition_date_b": meta_b.deposition_date,
            "release_date_b": meta_b.release_date,
            "title_b": meta_b.title or None,
            "n_polymer_chains_b": meta_b.n_polymer_chains,
            "n_models_b": meta_b.n_models,
            "is_af_b": meta_b.is_alphafold,
            "af_version_b": meta_b.af_version,
            "plddt_mean_b": meta_b.plddt_mean,
            "plddt_confirmed_b": meta_b.plddt_source_confirmed,
            "plddt_note_b": meta_b.plddt_note or None,
            "msa_depth_b": meta_b.msa_depth,
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
    The spread of one NMR ensemble: the representative model (chosen by
    src/nmr.py) compared with every other deposited model, one record each.
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
                "Intra-ensemble comparison: the spread of one NMR experiment. "
                "Interpret separately from the comparisons between structures."
            ),
            verbose=verbose,
        )

        rec["comparison"] = f"nmr_intra_model{representative_model}_vs_model{model_b}"
        rec["category"] = "nmr_intra_ensemble"
        rec["is_intra_ensemble"] = True

        records.append(rec)

    return records


_PIPELINE_METADATA = {
    "comparison_scope": (
        "Chain-level single polypeptide chain.  AlphaFold predictions are per-UniProt "
        "sequence, matching this scope.  Biological assembly-level conclusions "
        "(oligomeric states, inter-chain interactions) cannot be drawn from these results.  "
        "Each record carries comparison_scope = 'chain_level' as a short tag."
    ),
    "statistical_interpretation_note": (
        "The unit of analysis is the protein: the comparison pairs of one protein "
        "share structures and are not independent.  For each protein and metric the "
        "experimental variability is the median distance between its experimental "
        "structures, and a prediction is within experimental variability when its "
        "median distance to them is not larger.  Across proteins the analysis reports "
        "fractions, medians and paired effect sizes with 95 % bootstrap intervals "
        "(proteins resampled), for all proteins and for those without flagged "
        "experimental structures.  It is descriptive: no hypothesis test is run.  "
        "'Within experimental variability' is a necessary but not sufficient "
        "condition for claiming that a prediction can replace an experiment."
    ),
    "plddt_note": (
        "AF2 pLDDT is extracted from B_iso_or_equiv (well-documented convention). "
        "OF3 pLDDT is extracted the same way but range-validated; if values fall outside "
        "[0, 100], confidence is set to None and the plddt_note field explains why. "
        "pLDDT is not meaningful for experimental structures (field stores B-factors there)."
    ),
    "nmr_intra_ensemble_note": (
        "NMR intra-ensemble records (is_intra_ensemble=True) are included in "
        "comparisons.json.  They are the spread of one NMR experiment (part "
        "flexibility in solution, part restraint uncertainty) and are never pooled "
        "with the other comparisons: the protein-level analysis "
        "uses them as a second baseline (is the prediction no farther from the "
        "representative NMR model than the farthest other model of the ensemble?).  "
        "Filter on category != "
        "'nmr_intra_ensemble' to obtain only the primary comparison records."
    ),
}


def run_pipeline(
    proteins_csv: str = "proteins.csv",
    data_dir: str = "data",
    output_dir: str = "results",
    verbose: bool = True,
    include_nmr_ensemble: bool = True,
    nmr_model_policy: str = "medoid",
) -> dict:
    """
    Run the full comparison pipeline.

    Parameters
    ----------
    proteins_csv          : path to the dataset manifest.
    data_dir              : root directory with experimental/, alphafold2/, openfold3/.
    output_dir            : where to write all output JSON files.
    verbose               : print progress to stdout.
    include_nmr_ensemble  : whether to compute intra-NMR ensemble variability.
    nmr_model_policy      : which model of an NMR ensemble stands for the entry:
                            "medoid" (default), "pdb" or "first" (model 0).
                            See src/nmr.py.

    Returns
    -------
    Dict with:
      "comparisons"              : list of all records (ok + intra-ensemble)
      "rejected"                 : list of rejected/failed records (with reasons)
      "analysis"                 : statistical analysis output
      "per_protein_summary"      : one row per protein (analysis.protein_table)
      "pipeline_metadata"        : documentation of scope, exclusions, caveats
      "run_metadata"             : timestamp, library versions, input checksums
    """
    if nmr_model_policy not in NMR_POLICIES:
        raise ValueError(
            f"nmr_model_policy must be one of {NMR_POLICIES}, got {nmr_model_policy!r}"
        )
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(proteins_csv, dtype=str, keep_default_na=False)
    shared_names = sorted(set(df.loc[df["protein_name"].duplicated(), "protein_name"]))
    if shared_names:
        raise ValueError(
            f"protein_name must be unique in {proteins_csv} (it is used as a key); "
            f"duplicated: {shared_names}"
        )

    missing_cols = _REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"proteins.csv is missing required columns: {missing_cols}.  "
            f"Found columns: {list(df.columns)}"
        )

    all_records: list[dict] = []
    rejected: list[dict] = []
    missing_inputs: list[dict] = []
    nmr_model_rows: list[dict] = []
    all_cif_paths: set[str] = set()

    for _, row in df.iterrows():
        protein_name = row["protein_name"]
        uniprot_id = row["uniprot_id"]
        # One chain per method, chosen by src/dataset.py; predictions are always chain A.
        chains = {
            "X-ray": row["chain_xray"].strip(),
            "NMR": row["chain_nmr"].strip(),
            "Cryo-EM": row["chain_cryoem"].strip(),
            "AF2": "A",
            "OF3": "A",
        }

        def _chain_for(method: str) -> str:
            """Return the chain ID to read for the given method category."""
            return chains[method]

        xray_path = _exp_path(row.get("pdb_xray"), data_dir)
        nmr_path = _exp_path(row.get("pdb_nmr"), data_dir)
        cryoem_path = _exp_path(row.get("pdb_cryoem"), data_dir)

        # Which model of the NMR ensemble stands for the entry (src/nmr.py)
        nmr_model = 0
        nmr_choice = None
        if nmr_path and os.path.exists(nmr_path):
            try:
                nmr_choice = choose_nmr_model(
                    nmr_path, _chain_for("NMR"), policy=nmr_model_policy
                )
                nmr_model = nmr_choice["model"]
                nmr_model_rows.append(
                    {
                        "protein_name": protein_name,
                        "uniprot_id": uniprot_id,
                        "pdb_id": row.get("pdb_nmr"),
                        "chain": _chain_for("NMR"),
                        "model_used": nmr_model,
                        **{k: v for k, v in nmr_choice.items() if k != "model"},
                    }
                )
            except Exception as exc:  # unreadable file: compare_pair reports it as usual
                if verbose:
                    print(f"  [WARN] NMR model selection failed for {nmr_path}: {exc}")
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
                        "chain_id": _chain_for(structure_type),
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
                chain_id=_chain_for(ma),
                chain_id_a=_chain_for(ma),
                chain_id_b=_chain_for(mb),
                model_idx_a=mia,
                model_idx_b=mib,
                pdb_id_a=str(pdb_a) if pdb_a else "",
                pdb_id_b=str(pdb_b) if pdb_b else "",
                notes=notes_str,
                verbose=verbose,
            )

            if "NMR" in (ma, mb) and nmr_choice is not None:
                rec["nmr_model_used"] = nmr_model
                rec["nmr_model_rule"] = nmr_choice["rule"]

            if rec["status"] == "ok":
                all_records.append(rec)
            else:
                rejected.append(rec)

        if include_nmr_ensemble and nmr_path:
            intra_records = compute_nmr_ensemble_variability(
                protein_name,
                uniprot_id,
                nmr_path,
                chain_id=_chain_for("NMR"),
                representative_model=nmr_model,
                verbose=verbose,
            )
            all_records.extend([r for r in intra_records if r.get("status") == "ok"])
            rejected.extend([r for r in intra_records if r.get("status") != "ok"])

    analysis_result = analyse(all_records)  # protein level; uses the NMR ensembles too

    if verbose:
        print("\n\n" + analysis_result["summary_text"])

    run_metadata = _build_run_metadata(all_cif_paths)

    missing_inputs_path = os.path.join(output_dir, "missing_inputs.csv")
    pd.DataFrame(missing_inputs).to_csv(missing_inputs_path, index=False)
    if verbose:
        print(
            f"Wrote {len(missing_inputs)} missing-input records to {missing_inputs_path}"
        )

    pd.DataFrame(nmr_model_rows).to_csv(
        os.path.join(output_dir, "nmr_models.csv"), index=False
    )

    metrics_path = os.path.join(output_dir, "comparisons.json")
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

    pipeline_output = {
        "comparisons": all_records,
        "rejected": rejected,
        "missing_inputs": missing_inputs,
        "analysis": analysis_result,
        "per_protein_summary": analysis_result["per_protein"],
        "pipeline_metadata": _PIPELINE_METADATA,
        "run_metadata": {**run_metadata, "nmr_model_policy": nmr_model_policy},
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
