"""
Download and acquire structure files from RCSB PDB, AlphaFold DB, and OpenFold3 (NVIDIA NIM).

Acquisition layers
------------------
Experimental structures
  search_rcsb_by_uniprot()   Query the RCSB search API for all PDB entries linked
                              to a UniProt accession, grouped by experimental method.
  download_rcsb()            Download a single mmCIF by PDB ID.

AlphaFold2
  download_alphafold2()      Fetch the latest AF2 prediction from the AlphaFold DB
                             API using the UniProt ID.

OpenFold3 (NVIDIA NIM)
  fetch_uniprot_sequence()   Retrieve the canonical amino acid sequence from UniProt.
  predict_openfold3()        Submit a sequence to the NVIDIA NIM OpenFold3 endpoint
                             and save the returned mmCIF.

Utilities
---------
  load_proteins()            Read a two-column (name, UniProtID) text file into a
                             list of tuples. Retained for ad-hoc scripting; the
                             pipeline reads proteins.csv via pd.read_csv() directly.

"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import time
from typing import Optional

import gemmi
import requests

_RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
_RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"

_METHOD_LABELS = {
    "X-ray": "X-RAY DIFFRACTION",
    "NMR": "SOLUTION NMR",
    "Cryo-EM": "ELECTRON MICROSCOPY",
}

_NIM_OF3_URL = "https://health.api.nvidia.com/v1/biology/openfold/openfold3/predict"


def search_rcsb_by_uniprot(
    uniprot_id: str,
    methods: tuple[str, ...] = ("X-ray", "NMR", "Cryo-EM"),
    max_per_method: int = 3,
    resolution_cutoff: Optional[float] = 3.5,
    timeout: int = 30,
) -> dict[str, list[dict]]:
    """
    Uses the RCSB search API to find all PDB entries
    that reference the given UniProt ID, then the RCSB GraphQL Data API to
    retrieve method, resolution, and chain information for each hit.

    Parameters
    ----
    uniprot_id        : UniProt accession, e.g. "P0DP23".
    methods           : which experimental methods to return.  Must be keys of
                        _METHOD_LABELS ("X-ray", "NMR", "Cryo-EM").
    max_per_method    : maximum number of entries to return per method.
                        Results are sorted by resolution (best first) for X-ray
                        and Cryo-EM; NMR entries are sorted by deposition date.
    resolution_cutoff : reject X-ray and Cryo-EM structures worse than this (Å).
                        Set to None to disable.  NMR structures are not filtered.
    timeout           : HTTP request timeout in seconds.

    Returns
    -------
    Dict mapping method label to a list of dicts, each with:
      "pdb_id"      : four-character PDB ID (uppercase)
      "method"      : normalised method label ("X-ray", "NMR", "Cryo-EM")
      "resolution"  : float or None
      "chain_id"    : chain that carries the UniProt sequence (first match)
      "uniprot_id"  : the queried accession (echoed for traceability)

    Raises
    ------
    requests.HTTPError if the RCSB API returns a non-200 status.
    ValueError if an unknown method label is requested.
    """
    for m in methods:
        if m not in _METHOD_LABELS:
            raise ValueError(
                f"Unknown method {m!r}. Must be one of: {list(_METHOD_LABELS)}"
            )

    search_query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": (
                    "rcsb_polymer_entity_container_identifiers"
                    ".reference_sequence_identifiers.database_accession"
                ),
                "operator": "exact_match",
                "value": uniprot_id,
                "negation": False,
            },
        },
        "return_type": "entry",
        "request_options": {
            "results_verbosity": "compact",
            "return_all_hits": True,
        },
    }

    resp = requests.post(_RCSB_SEARCH_URL, json=search_query, timeout=timeout)
    resp.raise_for_status()

    if not resp.text.strip():
        print(f"  [warn] No RCSB entries found for UniProt ID {uniprot_id}")
        return {m: [] for m in methods}

    result_json = resp.json()
    raw = result_json.get("result_set", [])
    pdb_ids = [r["identifier"] if isinstance(r, dict) else r for r in raw]

    if not pdb_ids:
        print(f"  [warn] No RCSB entries found for UniProt ID {uniprot_id}")
        return {m: [] for m in methods}

    print(
        f"  [info] {uniprot_id}: {len(pdb_ids)} total RCSB hits — fetching metadata ..."
    )

    all_entries: list[dict] = []
    chunk_size = 100

    for i in range(0, len(pdb_ids), chunk_size):
        chunk = pdb_ids[i : i + chunk_size]
        ids_gql = str(chunk).replace("'", '"')
        graphql_query = f"""
        {{
          entries(entry_ids: {ids_gql}) {{
            rcsb_id
            exptl {{
              method
            }}
            refine {{
              ls_d_res_high
            }}
            rcsb_entry_info {{
              resolution_combined
            }}
            em_3d_reconstruction {{
              resolution
            }}
            polymer_entities {{
              rcsb_polymer_entity_container_identifiers {{
                auth_asym_ids
                reference_sequence_identifiers {{
                  database_accession
                  database_name
                }}
              }}
            }}
          }}
        }}
        """

        gql_resp = requests.post(
            _RCSB_GRAPHQL_URL,
            json={"query": graphql_query},
            timeout=timeout,
        )
        gql_resp.raise_for_status()
        data = gql_resp.json().get("data", {}).get("entries", [])
        all_entries.extend(data)
        time.sleep(0.1)

    grouped: dict[str, list[dict]] = {m: [] for m in methods}

    for entry in all_entries:
        pdb_id = entry.get("rcsb_id", "").upper()

        raw_methods = [e.get("method", "") for e in (entry.get("exptl") or [])]
        raw_method = raw_methods[0].upper().strip() if raw_methods else ""

        matched_label: Optional[str] = None
        for label, canonical in _METHOD_LABELS.items():
            if canonical in raw_method or raw_method in canonical:
                matched_label = label
                break

        if matched_label is None or matched_label not in methods:
            continue

        resolution: Optional[float] = None
        refine = entry.get("refine") or []
        if refine and refine[0].get("ls_d_res_high") is not None:
            try:
                resolution = float(refine[0]["ls_d_res_high"])
            except (TypeError, ValueError):
                pass
        if resolution is None:
            combined = (entry.get("rcsb_entry_info") or {}).get("resolution_combined")
            if combined is not None:
                try:
                    resolution = float(combined)
                except (TypeError, ValueError):
                    pass
        if resolution is None:
            em_recon = entry.get("em_3d_reconstruction") or []
            if em_recon and em_recon[0].get("resolution") is not None:
                try:
                    resolution = float(em_recon[0]["resolution"])
                except (TypeError, ValueError):
                    pass

        if matched_label != "NMR" and resolution_cutoff is not None:
            if resolution is None or resolution > resolution_cutoff:
                continue
        # for simple single-chain proteins, the chain is almost always labelled A by convention
        chain_id = "A"
        for entity in entry.get("polymer_entities") or []:
            ids_block = entity.get("rcsb_polymer_entity_container_identifiers", {})
            refs = ids_block.get("reference_sequence_identifiers") or []
            for ref in refs:
                if (
                    ref.get("database_name", "").upper() == "UNIPROT"
                    and ref.get("database_accession", "").upper() == uniprot_id.upper()
                ):
                    chains = ids_block.get("auth_asym_ids") or []
                    if chains:
                        chain_id = chains[0]
                    break

        grouped[matched_label].append(
            {
                "pdb_id": pdb_id,
                "method": matched_label,
                "resolution": resolution,
                "chain_id": chain_id,
                "uniprot_id": uniprot_id,
            }
        )

    for label in methods:
        bucket = grouped[label]
        if label in ("X-ray", "Cryo-EM"):
            bucket.sort(key=lambda x: (x["resolution"] is None, x["resolution"] or 999))
        grouped[label] = bucket[:max_per_method]

    for label, hits in grouped.items():
        print(f"  [info] {uniprot_id} {label}: {len(hits)} candidate(s) selected")

    return grouped


def download_rcsb(pdb_id: str, save_dir: str, timeout: int = 30) -> str:
    """
    Download a mmCIF file from RCSB PDB by PDB ID.

    Parameters
    ----
    pdb_id   : four-character PDB accession, e.g. "1UBQ".
    save_dir : directory to save the file into.
    timeout  : HTTP request timeout in seconds.

    Returns
    -------
    Full path to the saved .cif file.
    """
    os.makedirs(save_dir, exist_ok=True)
    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    out_path = os.path.join(save_dir, f"{pdb_id}.cif")

    if os.path.exists(out_path):
        print(f"  [skip] {pdb_id}.cif already exists")
        return out_path

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"  [ok]   Downloaded {pdb_id}.cif -> {out_path}")
    return out_path


def download_alphafold2(uniprot_id: str, save_dir: str, timeout: int = 30) -> str:
    """
    Download an AlphaFold2 predicted structure in mmCIF format from the AlphaFold DB.
    The API always returns the latest available prediction.

    Parameters
    ----
    uniprot_id : UniProt accession, e.g. "P0DP23".
    save_dir   : directory to save into (created if absent).
    timeout    : HTTP request timeout in seconds.

    Returns
    -------
    Full path to the saved AF2_<uniprot_id>.cif file.
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"AF2_{uniprot_id}.cif")

    if os.path.exists(out_path):
        print(f"  [skip] AF2_{uniprot_id}.cif already exists")
        return out_path

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    response = requests.get(api_url, timeout=timeout)
    response.raise_for_status()

    records = response.json()
    if not records:
        raise ValueError(
            f"No AlphaFold DB prediction found for UniProt ID {uniprot_id}"
        )

    cif_url = records[0].get("cifUrl")
    if not cif_url:
        raise ValueError(
            f"AlphaFold DB API response for {uniprot_id} did not contain a cifUrl"
        )

    cif_response = requests.get(cif_url, timeout=timeout)
    cif_response.raise_for_status()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cif_response.text)

    print(f"  [ok]   Downloaded AF2 for {uniprot_id} -> {out_path}")
    return out_path


def fetch_uniprot_sequence(uniprot_id: str, timeout: int = 15) -> str:
    """
    Fetch the canonical amino acid sequence for a UniProt accession. The
    returnedvsequence is the reviewed canonical isoform for Swiss-Prot
    entries.

    Parameters
    ----
    uniprot_id : UniProt accession, e.g. "P0DP23".
    timeout    : HTTP request timeout in seconds.

    Returns
    -------
    Single-letter amino acid sequence string (no header, no line breaks).

    Raises
    ------
    requests.HTTPError if the UniProt REST API returns a non-200 status.
    ValueError if the FASTA response is empty or malformed.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    lines = resp.text.strip().splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError(
            f"Unexpected FASTA format from UniProt for {uniprot_id}:\n{resp.text[:200]}"
        )

    sequence = "".join(lines[1:])
    if not sequence:
        raise ValueError(f"Empty sequence returned by UniProt for {uniprot_id}")

    return sequence


def predict_openfold3(
    uniprot_id: str,
    sequence: str,
    save_dir: str,
    nim_api_key: Optional[str] = None,
    diffusion_samples: int = 1,
    timeout: int = 300,
) -> str:
    """
    Submit an amino acid sequence to the NVIDIA NIM OpenFold3 endpoint and
    save the returned mmCIF prediction.

    OpenFold3 reference
    -------------------
    Model:   https://github.com/aqlaboratory/openfold-3
    NIM API: https://build.nvidia.com/openfold/openfold3
    Docs:    https://docs.nvidia.com/nim/bionemo/openfold3/latest/example-requests.html

    Authentication
    --------------
    Requires a NIM API key.  Pass it via the `nim_api_key` argument or set
    the environment variable NIM_API_KEY. Set the environment variable NIM_API_KEY before calling predict_openfold3().
    The recommended way is a .env file at the repo root:

        NIM_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx

    loaded at the top of the acquisition notebook with:

    from dotenv import load_dotenv; load_dotenv()

    --------------------------------------

    Parameters
    ----
    uniprot_id        : UniProt accession used to name the output file.
    sequence          : Single-letter amino acid sequence (from fetch_uniprot_sequence).
    save_dir          : Directory to save the CIF into (created if absent).
    nim_api_key       : NIM API key.  Falls back to os.environ["NIM_API_KEY"] if None.
    diffusion_samples : Independent structure samples to generate (1-5, default 1).
                        1 is fastest; the first sample is saved as the output file.
    timeout           : HTTP request timeout in seconds.  OF3 predictions for
                        proteins up to ~500 residues typically complete in 60-120 s.

    Returns
    -------
    Full path to the saved OF3_<uniprot_id>.cif file.

    Raises
    ------
    EnvironmentError  if no API key is available.
    requests.HTTPError if the NIM endpoint returns a non-200 status.
    ValueError if the response does not contain a valid mmCIF structure.
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"OF3_{uniprot_id}.cif")

    if os.path.exists(out_path):
        print(f"  [skip] OF3_{uniprot_id}.cif already exists")
        return out_path

    key = nim_api_key or os.environ.get("NIM_API_KEY")
    if not key:
        raise EnvironmentError(
            "No NIM API key found.  Set NIM_API_KEY in your environment or .env file, "
            "or pass it as the nim_api_key argument.\n"
            "Get a key at: https://build.nvidia.com/openfold/openfold3"
        )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    minimal_msa_a3m = f">query\n{sequence}"

    payload = {
        "inputs": [
            {
                "input_id": uniprot_id,
                "molecules": [
                    {
                        "type": "protein",
                        "sequence": sequence,
                        "msa": {
                            "main": {
                                "a3m": {
                                    "alignment": minimal_msa_a3m,
                                    "format": "a3m",
                                }
                            }
                        },
                    }
                ],
                "diffusion_samples": diffusion_samples,
                "output_format": "cif",
            }
        ]
    }

    print(
        f"  [info] Submitting OF3 prediction for {uniprot_id} "
        f"({len(sequence)} aa) to NVIDIA NIM ..."
    )

    resp = requests.post(_NIM_OF3_URL, json=payload, headers=headers, timeout=timeout)

    if resp.status_code == 401:
        raise requests.HTTPError(
            f"NIM authentication failed (401). Check your NIM_API_KEY.\n"
            f"Response: {resp.text[:300]}"
        )
    if resp.status_code == 422:
        raise requests.HTTPError(
            f"NIM validation error (422). The request schema is malformed.\n"
            f"Response: {resp.text[:500]}"
        )
    if resp.status_code == 429:
        raise requests.HTTPError(
            f"NIM rate limit exceeded (429). Wait and retry, or check your quota.\n"
            f"Response: {resp.text[:300]}"
        )
    resp.raise_for_status()

    result = resp.json()

    outputs = result.get("outputs") or []
    if not outputs:
        raise ValueError(
            f"NIM OF3 response for {uniprot_id} has no 'outputs' field.\n"
            f"Top-level keys: {list(result.keys())}\n"
            f"Raw (first 500 chars): {resp.text[:500]}"
        )

    structures_with_scores = outputs[0].get("structures_with_scores") or []
    if not structures_with_scores:
        raise ValueError(
            f"NIM OF3 response for {uniprot_id}: 'structures_with_scores' is empty.\n"
            f"Output-level keys: {list(outputs[0].keys())}"
        )

    cif_content: Optional[str] = structures_with_scores[0].get("structure")
    if not cif_content or not isinstance(cif_content, str):
        raise ValueError(
            f"NIM OF3 'structure' field missing or empty for {uniprot_id}.\n"
            f"structures_with_scores[0] keys: {list(structures_with_scores[0].keys())}"
        )

    try:
        gemmi.cif.read_string(cif_content)
    except Exception as exc:
        raise ValueError(
            f"NIM OF3 response for {uniprot_id} is not valid mmCIF: {exc}"
        ) from exc

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cif_content)

    print(f"  [ok]   OF3 prediction saved -> {out_path}")

    prov_path = out_path.replace(".cif", "_provenance.json")
    confidence_fields = {
        k: v for k, v in structures_with_scores[0].items() if k != "structure"
    }
    provenance = {
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "nim_endpoint_url": _NIM_OF3_URL,
        "uniprot_id": uniprot_id,
        "sequence_length": len(sequence),
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "diffusion_samples": diffusion_samples,
        "msa_type": "single_sequence_query_only",
        "msa_note": (
            "Only the query sequence was submitted as the MSA (no homologs). "
            "This is equivalent to MSA-free prediction. AF2 predictions from "
            "the AlphaFold DB use deep evolutionary MSAs; this difference "
            "must be stated as a limitation when comparing AF2 and OF3 results."
        ),
        "nim_response_headers": {
            k: v
            for k, v in resp.headers.items()
            if k.lower()
            in (
                "x-request-id",
                "x-nim-version",
                "x-model-version",
                "content-type",
                "date",
                "server",
            )
        },
        "confidence_scores": confidence_fields,
        "output_cif_path": out_path,
    }
    with open(prov_path, "w", encoding="utf-8") as pf:
        json.dump(provenance, pf, indent=2, default=str)
    print(f"  [ok]   OF3 provenance saved -> {prov_path}")

    return out_path


def verify_chain_uniprot_mapping(
    pdb_id: str,
    chain_id: str,
    expected_uniprot_id: str,
    timeout: int = 15,
) -> dict:
    """
    Confirm via the RCSB GraphQL API that a specific PDB chain is officially mapped to
    the expected UniProt accession, catching chain mismatches before the pipeline runs.


    Parameters
    ----
    pdb_id             : four-character PDB ID (case-insensitive).
    chain_id           : chain letter from proteins.csv (e.g. "A", "B").
    expected_uniprot_id: UniProt accession to verify against.
    timeout            : HTTP timeout in seconds.

    Returns
    -------
    dict with:
      "verified"        : bool — True if RCSB maps this chain to the expected UniProt.
      "mapped_uniprots" : list[str] — all UniProt IDs RCSB associates with this chain.
      "all_chains_for_uniprot" : list[str] — chains in this entry that carry the UniProt.
      "error"           : str or None — set if the API call failed.
    """
    pdb_id = pdb_id.upper().strip()
    expected = expected_uniprot_id.upper().strip()

    query = f"""
    {{
      polymer_entity_instances(instance_ids: ["{pdb_id}.{chain_id}"]) {{
        polymer_entity {{
          rcsb_polymer_entity_container_identifiers {{
            auth_asym_ids
            reference_sequence_identifiers {{
              database_accession
              database_name
            }}
          }}
        }}
      }}
    }}
    """

    try:
        resp = requests.post(
            _RCSB_GRAPHQL_URL,
            json={"query": query},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return {
            "verified": False,
            "mapped_uniprots": [],
            "all_chains_for_uniprot": [],
            "error": str(exc),
        }

    instances = (data.get("data") or {}).get("polymer_entity_instances") or []
    if not instances:
        return {
            "verified": False,
            "mapped_uniprots": [],
            "all_chains_for_uniprot": [],
            "error": f"No polymer_entity_instances found for {pdb_id}.{chain_id}",
        }

    entity_ids = (
        instances[0]
        .get("polymer_entity", {})
        .get("rcsb_polymer_entity_container_identifiers", {})
    )
    refs = entity_ids.get("reference_sequence_identifiers") or []
    mapped = [
        r["database_accession"].upper()
        for r in refs
        if r.get("database_name", "").upper() == "UNIPROT"
        and r.get("database_accession")
    ]
    all_chains = entity_ids.get("auth_asym_ids") or []

    return {
        "verified": expected in mapped,
        "mapped_uniprots": mapped,
        "all_chains_for_uniprot": all_chains,
        "error": None,
    }


def download_missing_inputs(
    missing_inputs_csv: str,
    data_dir: str = "data",
    nim_api_key: Optional[str] = None,
    skip_of3: bool = False,
) -> dict:
    """
    Attempt to download all structure files listed in missing_inputs.csv. (Due to network or api related errors)
    After running this, re-run the pipeline to include the newly downloaded files.

    Parameters
    ----
    missing_inputs_csv : path to missing_inputs.csv (default: results/missing_inputs.csv).
    data_dir           : root data directory (must contain experimental/, alphafold2/,
                         openfold3/ subdirectories or they will be created).
    nim_api_key        : NIM API key for OF3 downloads.  Falls back to NIM_API_KEY env var.
    skip_of3           : if True, skip OF3 entries (useful when NIM key is unavailable).

    Returns
    -------
    dict with:
      "attempted": int
      "succeeded": int
      "failed": list[dict]  — entries that could not be downloaded
      "skipped": list[dict] — entries skipped (e.g. OF3 with skip_of3=True)
    """
    import csv

    if not os.path.exists(missing_inputs_csv):
        print(f"[info] {missing_inputs_csv} not found — nothing to download.")
        return {"attempted": 0, "succeeded": 0, "failed": [], "skipped": []}

    with open(missing_inputs_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    attempted = succeeded = 0
    failed: list[dict] = []
    skipped: list[dict] = []

    for row in rows:
        structure_type = row.get("structure_type", "").strip()
        pdb_or_uniprot = row.get("pdb_id_or_uniprot", "").strip()
        uniprot_id = row.get("uniprot_id", "").strip()
        protein_name = row.get("protein_name", "").strip()

        if not pdb_or_uniprot or pdb_or_uniprot.lower() in ("nan", ""):
            skipped.append({**row, "skip_reason": "empty identifier"})
            continue

        try:
            if structure_type in ("X-ray", "NMR", "Cryo-EM"):
                attempted += 1
                download_rcsb(pdb_or_uniprot, os.path.join(data_dir, "experimental"))
                succeeded += 1

            elif structure_type == "AF2":
                attempted += 1
                download_alphafold2(
                    uniprot_id or pdb_or_uniprot, os.path.join(data_dir, "alphafold2")
                )
                succeeded += 1

            elif structure_type == "OF3":
                if skip_of3:
                    skipped.append({**row, "skip_reason": "skip_of3=True"})
                    continue
                attempted += 1
                seq = fetch_uniprot_sequence(uniprot_id or pdb_or_uniprot)
                predict_openfold3(
                    uniprot_id=uniprot_id or pdb_or_uniprot,
                    sequence=seq,
                    save_dir=os.path.join(data_dir, "openfold3"),
                    nim_api_key=nim_api_key,
                )
                succeeded += 1

            else:
                skipped.append(
                    {**row, "skip_reason": f"unknown structure_type '{structure_type}'"}
                )

        except Exception as exc:
            print(f"  [fail] {protein_name} {structure_type} {pdb_or_uniprot}: {exc}")
            failed.append({**row, "error": str(exc)})

    print(
        f"\n[download_missing_inputs] attempted={attempted}  "
        f"succeeded={succeeded}  failed={len(failed)}  skipped={len(skipped)}"
    )
    return {
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
    }


def load_proteins(file_path: str) -> list[tuple[str, str]]:
    """
    Load a protein list from a plain-text file.

    Each non-empty, non-comment line must have the format:
        protein_name,UniProtID

    Lines starting with '#' and blank lines are ignored.

    Parameters
    ----------
    file_path : path to the protein list file.

    Returns
    -------
    List of (protein_name, uniprot_id) tuples.
    """
    proteins = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, uniprot = line.split(",", 1)
            proteins.append((name.strip(), uniprot.strip()))
    return proteins
