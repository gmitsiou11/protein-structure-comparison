"""
Extract structural, experimental, and AI-specific metadata from mmCIF files.

This module does NOT compute any comparison metrics — it only reads mmCIF
fields and returns structured objects.  Every comparison record produced by
``pipeline.py`` embeds this metadata so results are self-documenting.

AlphaFold confidence handling
------------------------------
**AF2 (AlphaFold DB)**: pLDDT is stored per-atom in ``_atom_site.B_iso_or_equiv``.
This is a well-documented convention confirmed in the AlphaFold DB documentation.
Values are in the range [0, 100].

**OF3 (OpenFold3 via NVIDIA NIM)**: The NIM endpoint returns mmCIF files that follow
the same B_iso_or_equiv convention for per-residue confidence scores.  This module
applies a range validation check: if the extracted values fall outside [0, 100] or
have a mean consistent with crystallographic B-factors, they are flagged as
potentially not pLDDT and confidence is set to None.


mmCIF field references
-----------------------
  _exptl.method                           — experimental method string
  _refine.ls_d_res_high                   — X-ray resolution (Å)
  _em_3d_reconstruction.resolution        — Cryo-EM resolution (Å)
  _pdbx_nmr_ensemble.conformers_submitted_total_number — NMR model count
  _entity_src_gen.pdbx_gene_src_scientific_name        — organism
  _atom_site.B_iso_or_equiv               — B-factor / pLDDT per atom
  _atom_site.pdbx_PDB_model_num           — NMR model index (PDBx extension field)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import re

import gemmi
import numpy as np

_PLDDT_MIN = 0.0
_PLDDT_MAX = 100.0
_PLDDT_MEAN_MIN = 10.0

_METHOD_MAP = {
    "X-RAY DIFFRACTION": "X-ray",
    "SOLUTION NMR": "NMR",
    "SOLID-STATE NMR": "NMR",
    "ELECTRON MICROSCOPY": "Cryo-EM",
    "ELECTRON CRYSTALLOGRAPHY": "Cryo-EM",
}


@dataclass
class StructureMetadata:
    """
    All metadata associated with a single structure file.

    Experimental structures
    -----------------------
      method           : raw mmCIF method string (e.g. \"SOLUTION NMR\")
      method_category  : normalised label -- "X-ray", "NMR", "Cryo-EM",
                         "AF2", or "OF3"
      resolution       : resolution in Å for X-ray and Cryo-EM; None for NMR
      n_models         : deposited model count (1 for X-ray/AF; >1 for NMR)
      organism         : scientific name from mmCIF

    AlphaFold-specific
    ------------------
      is_alphafold            : True if file is an AF2 or OF3 prediction
      af_version              : "AF2" | "OF3" | None
      plddt_mean              : mean pLDDT over Ca atoms in chain/model; None
                                if confidence extraction failed or unconfirmed
      plddt_per_residue       : per-Ca pLDDT list (for matched-region analysis)
      plddt_source_confirmed  : True  -- confirmed pLDDT (AF2 or OF3 NIM format)
                                False -- values extracted but range suspicious
                                None  -- not an AI prediction structure

    Common
    ------
      n_residues : Cα count for the parsed chain/model
      chain_id   : chain identifier
      model_idx  : model index (0-based)
    """

    # identification
    file_path: str = ""
    pdb_id: str = ""
    uniprot_id: str = ""

    # experimental
    method: str = ""
    method_category: str = ""
    resolution: Optional[float] = None
    n_models: int = 1
    organism: str = ""

    # AlphaFold-specific
    is_alphafold: bool = False
    af_version: Optional[str] = None
    plddt_mean: Optional[float] = None
    plddt_per_residue: Optional[list] = None
    plddt_source_confirmed: Optional[bool] = None  # see module docstring
    plddt_note: str = ""  # human-readable note when confidence is uncertain

    # structure scope
    n_residues: int = 0
    chain_id: str = "A"
    model_idx: int = 0

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (drops the large plddt_per_residue list)."""
        d = asdict(self)
        d.pop("plddt_per_residue", None)
        return d


_AF2_PATTERN = re.compile(r"AF[-_]?2", re.IGNORECASE)
_OF3_PATTERN = re.compile(r"OF[-_]?3|OpenFold[-_]?3", re.IGNORECASE)


def _detect_af_version(file_path: str, entry_id: str) -> Optional[str]:
    """
    Detect whether a file is an AF2 or OF3 (OpenFold3) prediction.

    Detection order (first match wins):
      1. File name pattern  -- "OF3_P00698.cif" -> "OF3"
      2. File name pattern  -- "AF2_P00698.cif" -> "AF2"
      3. mmCIF _entry.id   -- "AF-P00698-F1-model_v4" -> "AF2"
    """
    name = file_path.replace("\\", "/").split("/")[-1]

    if _OF3_PATTERN.search(name) or _OF3_PATTERN.search(entry_id):
        return "OF3"
    if _AF2_PATTERN.search(name) or "alphafold" in entry_id.lower():
        return "AF2"
    if entry_id.startswith("AF-") and "-F1-" in entry_id:
        return "AF2"
    return None


def _extract_uniprot_from_filename(file_path: str) -> str:
    """Extract UniProt ID from filename, e.g. 'AF2_P00698.cif' → 'P00698'."""
    name = file_path.replace("\\", "/").split("/")[-1]
    m = re.search(
        r"[A-Z][0-9][A-Z0-9]{3}[0-9]|[OPQ][0-9][A-Z0-9]{3}[0-9]", name, re.IGNORECASE
    )
    return m.group(0).upper() if m else ""


def _is_plddt_range_plausible(arr: np.ndarray) -> bool:
    """
    Return True if the extracted B_iso_or_equiv values are consistent with pLDDT.

    pLDDT scores are in [0, 100].  Crystallographic B-factors are typically
    in [1, 100] Å² for well-ordered structures but can exceed 100 Å² for
    disordered regions, and they have a different distributional signature.

    """
    if len(arr) == 0:
        return False
    return (
        float(np.min(arr)) >= _PLDDT_MIN
        and float(np.max(arr)) <= _PLDDT_MAX
        and float(np.mean(arr)) >= _PLDDT_MEAN_MIN
    )


def _extract_plddt(
    block: gemmi.cif.Block,
    chain_id: str = "A",
    model_idx: int = 0,
) -> Optional[np.ndarray]:
    """
    Extract per-Cα B_iso_or_equiv values from the _atom_site loop.

    In AlphaFold mmCIF files this column stores pLDDT per atom.
    For experimental files it stores crystallographic B-factors.

    Returns an array of shape (N_CA,) or None if extraction fails.
    """
    for item in block:
        if not item.loop:
            continue
        tags = item.loop.tags
        if "_atom_site.B_iso_or_equiv" not in tags:
            continue

        required = ["_atom_site.label_atom_id", "_atom_site.B_iso_or_equiv"]
        if not all(t in tags for t in required):
            return None

        atom_col = tags.index("_atom_site.label_atom_id")
        biso_col = tags.index("_atom_site.B_iso_or_equiv")
        chain_col = (
            tags.index("_atom_site.auth_asym_id")
            if "_atom_site.auth_asym_id" in tags
            else None
        )
        model_col = (
            tags.index("_atom_site.pdbx_PDB_model_num")
            if "_atom_site.pdbx_PDB_model_num" in tags
            else None
        )

        if model_col is None and model_idx > 0:
            return None

        stride = len(tags)
        values = item.loop.values
        n_rows = len(values) // stride
        target_model = str(model_idx + 1)  # mmCIF model numbers are 1-based

        plddt_vals = []
        for i in range(n_rows):
            base = i * stride
            if model_col is not None and values[base + model_col] != target_model:
                continue
            if (
                chain_col is not None
                and values[base + chain_col].strip("'") != chain_id
            ):
                continue
            atom = values[base + atom_col].strip("'")
            if atom == "CA":
                try:
                    plddt_vals.append(float(values[base + biso_col]))
                except ValueError:
                    pass

        return np.array(plddt_vals) if plddt_vals else None

    return None


def _normalise_method(raw: str) -> str:
    return _METHOD_MAP.get(raw.upper().strip(), raw.strip())


def _safe_float(block: gemmi.cif.Block, tag: str) -> Optional[float]:
    try:
        val = block.find_value(tag)
        if val and val not in (".", "?", "None"):
            return float(val)
    except Exception:
        pass
    return None


def _get_resolution(block: gemmi.cif.Block, method_cat: str) -> Optional[float]:
    """Return resolution in Å, using the method-appropriate mmCIF tag."""
    if method_cat == "X-ray":
        for tag in ["_refine.ls_d_res_high", "_reflns.d_resolution_high"]:
            r = _safe_float(block, tag)
            if r is not None:
                return r
    elif method_cat == "Cryo-EM":
        for tag in ["_em_3d_reconstruction.resolution", "_refine.ls_d_res_high"]:
            r = _safe_float(block, tag)
            if r is not None:
                return r
    return None  # NMR has no resolution in this sense


def _get_nmr_model_count(block: gemmi.cif.Block) -> int:
    for tag in [
        "_pdbx_nmr_ensemble.conformers_submitted_total_number",
        "_pdbx_nmr_ensemble.conformers_calculated_total_number",
    ]:
        val = _safe_float(block, tag)
        if val is not None:
            return int(val)
    return 1


def _get_organism(block: gemmi.cif.Block) -> str:
    for tag in [
        "_entity_src_gen.pdbx_gene_src_scientific_name",
        "_entity_src_nat.pdbx_organism_scientific",
        "_pdbx_entity_src_syn.organism_scientific",
    ]:
        try:
            val = block.find_value(tag)
            if val and val not in (".", "?"):
                return val.strip("'")
        except Exception:
            pass
    return ""


def _count_residues(structure: gemmi.Structure, model_idx: int, chain_id: str) -> int:
    from src.parser import get_ca_coords_and_seq

    try:
        coords, _ = get_ca_coords_and_seq(structure, model_idx, chain_id)
        return len(coords)
    except Exception:
        return 0


def extract_metadata(
    file_path: str,
    uniprot_id: str = "",
    pdb_id: str = "",
    chain_id: str = "A",
    model_idx: int = 0,
) -> StructureMetadata:
    """
    Parse a mmCIF file and return a populated ``StructureMetadata`` object.

    Parameters
    ----
    file_path   : path to the mmCIF file.
    uniprot_id  : UniProt accession (from proteins.csv; also attempted from filename).
    pdb_id      : PDB accession (experimental structures; empty for AF files).
    chain_id    : chain to extract metrics from.
    model_idx   : model index (0-based).

    Returns
    -------
    ``StructureMetadata`` — always valid, even if some fields are None/empty.

    Confidence handling
    -------------------
    For AF2: pLDDT is extracted and confirmed (plddt_source_confirmed=True).
    For OF3: pLDDT is extracted and range-validated.
      - If plausible -> plddt_source_confirmed=True
      - If suspicious (e.g. B-factors) -> plddt set to None, plddt_source_confirmed=False,
        plddt_note explains why.
    For experimental structures: pLDDT is not extracted; plddt_source_confirmed=None.
    """
    meta = StructureMetadata(
        file_path=file_path,
        uniprot_id=uniprot_id or _extract_uniprot_from_filename(file_path),
        pdb_id=pdb_id,
        chain_id=chain_id,
        model_idx=model_idx,
    )

    try:
        doc = gemmi.cif.read(file_path)
        block = doc.sole_block()
    except Exception as exc:
        meta.method = f"PARSE ERROR: {exc}"
        return meta

    entry_id = ""
    try:
        entry_id = block.find_value("_entry.id").strip("'") or ""
    except Exception:
        pass

    af_version = _detect_af_version(file_path, entry_id)
    meta.is_alphafold = af_version is not None
    meta.af_version = af_version

    if meta.is_alphafold:
        meta.method = f"AlphaFold {af_version}"
        meta.method_category = af_version
        meta.n_models = 1

        raw_biso = _extract_plddt(block, chain_id=chain_id, model_idx=model_idx)

        if raw_biso is not None and len(raw_biso) > 0:
            if af_version == "AF2":
                meta.plddt_mean = float(np.mean(raw_biso))
                meta.plddt_per_residue = raw_biso.tolist()
                meta.plddt_source_confirmed = True

            elif af_version == "OF3":
                if _is_plddt_range_plausible(raw_biso):
                    meta.plddt_mean = float(np.mean(raw_biso))
                    meta.plddt_per_residue = raw_biso.tolist()
                    meta.plddt_source_confirmed = True
                else:
                    meta.plddt_mean = None
                    meta.plddt_per_residue = None
                    meta.plddt_source_confirmed = False
                    meta.plddt_note = (
                        f"B_iso_or_equiv values (min={float(np.min(raw_biso)):.1f}, "
                        f"max={float(np.max(raw_biso)):.1f}, mean={float(np.mean(raw_biso)):.1f}) "
                        "do not conform to the expected pLDDT range [0, 100].  "
                        "This OF3 file may store B-factors rather than pLDDT in this field.  "
                        "Confidence is set to None and excluded from analysis."
                    )
        else:
            meta.plddt_source_confirmed = False if meta.is_alphafold else None
            meta.plddt_note = "B_iso_or_equiv column not found or empty."

    else:
        raw_method = ""
        try:
            raw_method = block.find_value("_exptl.method").strip("'")
        except Exception:
            pass
        meta.method = raw_method
        meta.method_category = _normalise_method(raw_method)
        meta.resolution = _get_resolution(block, meta.method_category)
        meta.organism = _get_organism(block)
        meta.plddt_source_confirmed = None  # not applicable for experimental structures

        if meta.method_category == "NMR":
            meta.n_models = _get_nmr_model_count(block)

    try:
        structure = gemmi.read_structure(file_path)
        meta.n_residues = _count_residues(structure, model_idx, chain_id)
    except Exception:
        pass

    return meta


def plddt_on_matched_residues(
    meta: StructureMetadata,
    matched_indices: np.ndarray,
) -> Optional[float]:
    """
    Return mean pLDDT restricted to the aligned (matched) residue positions.

    This is more informative than the global mean because alignment typically
    excludes disordered termini where pLDDT is lowest.  Low pLDDT in the
    matched region is particularly relevant: it suggests the AI prediction is
    uncertain precisely in the region being compared.

    Parameters
    ----
    meta            : StructureMetadata with plddt_per_residue populated.
    matched_indices : 1-D array of indices into the Cα array (val.idx_a or val.idx_b).

    Returns
    -------
    Mean pLDDT over matched residues, or None if pLDDT is unavailable or unconfirmed.
    """
    if not meta.is_alphafold:
        return None
    if meta.plddt_source_confirmed is False:
        return None  # don't report unconfirmed values
    if meta.plddt_per_residue is None:
        return None

    arr = np.array(meta.plddt_per_residue)
    valid_idx = matched_indices[matched_indices < len(arr)]
    if len(valid_idx) == 0:
        return None
    return float(np.mean(arr[valid_idx]))
