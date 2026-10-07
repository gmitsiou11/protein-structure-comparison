"""
Download structure files from RCSB PDB and AlphaFold DB, and predict OpenFold3 models (NVIDIA NIM).

Experimental structures
  download_rcsb()            Download a single mmCIF by PDB ID.

AlphaFold2
  download_alphafold2()      Fetch the latest AF2 prediction from the AlphaFold DB
                             API using the UniProt ID.

OpenFold3 (NVIDIA NIM)
  fetch_uniprot_sequence()   Retrieve the canonical amino acid sequence from UniProt.
  fetch_msa()                Build the MSA of that sequence with the NVIDIA MSA Search
                             NIM (ColabFold MMseqs2 search: UniRef30 + ColabFold
                             environmental database) and cache it in data/msas/.
  predict_openfold3()        Submit the sequence and its MSA to the NVIDIA NIM
                             OpenFold3 endpoint and save the top-ranked mmCIF.

Recovery
  download_missing_inputs()  Fetch the files listed in results/missing_inputs.csv.
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

_NIM_HOST = "https://health.api.nvidia.com"
_NIM_OF3_URL = f"{_NIM_HOST}/v1/biology/openfold/openfold3/predict"
_NIM_MSA_URL = f"{_NIM_HOST}/v1/biology/colabfold/msa-search/predict"
_NIM_STATUS_URL = f"{_NIM_HOST}/v1/status"
_NIM_POLL_SECONDS = 300  # how long the server may hold a request before answering 202
_NIM_RETRY_STATUS = {429, 502, 503, 504}

# ColabFold's standard MSA: UniRef30 + the ColabFold environmental database
# (Mirdita et al. 2022, Nat Methods 19:679).  OpenFold3 NIM accepts <= 3 MSAs.
MSA_DATABASES = ("Uniref30_2302", "colabfold_envdb_202108")
_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# AlphaFold DB API fields kept as provenance of an AF2 model (those present in the answer)
_AFDB_PROVENANCE_KEYS = (
    "entryId",
    "latestVersion",
    "allVersions",
    "modelCreatedDate",
    "sequenceVersionDate",
    "uniprotStart",
    "uniprotEnd",
    "globalMetricValue",
    "cifUrl",
)


def download_rcsb(pdb_id: str, save_dir: str, timeout: int = 30) -> str:
    """
    Download a mmCIF file from RCSB PDB by PDB ID.

    Parameters
    ----------
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
    The API always returns the latest available prediction, so the model version
    and the download date are saved in AF2_<uniprot_id>_provenance.json.

    Parameters
    ----------
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

    record = records[0]
    provenance = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "api_url": api_url,
        "uniprot_id": uniprot_id,
        "cif_sha256": hashlib.sha256(cif_response.text.encode()).hexdigest(),
        **{k: record.get(k) for k in _AFDB_PROVENANCE_KEYS if k in record},
    }
    with open(out_path.replace(".cif", "_provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, default=str)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cif_response.text)

    print(f"  [ok]   Downloaded AF2 for {uniprot_id} -> {out_path}")
    return out_path


def fetch_uniprot_sequence(uniprot_id: str, timeout: int = 15) -> str:
    """
    Fetch the canonical amino acid sequence for a UniProt accession (for
    Swiss-Prot entries, the reviewed canonical isoform).

    Parameters
    ----------
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


def _nim_key(nim_api_key: Optional[str]) -> str:
    """The NIM API key: the argument, else the NIM_API_KEY environment variable."""
    key = nim_api_key or os.environ.get("NIM_API_KEY")
    if not key:
        raise EnvironmentError(
            "No NIM API key found.  Set NIM_API_KEY in your environment or .env file, "
            "or pass it as the nim_api_key argument.\n"
            "Get a key at: https://build.nvidia.com/openfold/openfold3"
        )
    return key


def _retry_wait(resp: requests.Response, attempt: int) -> float:
    """Seconds to wait before retrying: the Retry-After header, else 10, 20, 40, ... s."""
    try:
        return float(resp.headers.get("Retry-After", ""))
    except ValueError:
        return 10.0 * 2**attempt


def _post_nim(
    url: str, payload: dict, key: str, max_wait: int = 1800, max_retries: int = 5
) -> requests.Response:
    """
    POST a request to a hosted NVIDIA NIM endpoint and return the final response.

    429, 5xx  (rate limit, 502/503/504), a time-out or a dropped connection on
              submission: wait and submit again (the Retry-After header if
              given, else 10, 20, 40, ... s).
    202       the job is running: poll /v1/status/<nvcf-reqid> until it ends.
              The job is never submitted twice; temporary errors while polling
              are waited out and the same job is polled again.
    other     raise requests.HTTPError with the server's message.
    max_wait  bounds the whole call (submission + polling), in seconds.
    """
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "NVCF-POLL-SECONDS": str(_NIM_POLL_SECONDS),
    }
    http_timeout = _NIM_POLL_SECONDS + 30
    deadline = time.monotonic() + max_wait

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=http_timeout
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == max_retries:
                raise
            wait = 10.0 * 2**attempt
            print(
                f"  [wait] {type(exc).__name__} from NIM; "
                f"retry {attempt + 1}/{max_retries} in {wait:.0f} s"
            )
            time.sleep(wait)
            continue
        if resp.status_code not in _NIM_RETRY_STATUS or attempt == max_retries:
            break
        wait = _retry_wait(resp, attempt)
        print(
            f"  [wait] HTTP {resp.status_code} from NIM; "
            f"retry {attempt + 1}/{max_retries} in {wait:.0f} s"
        )
        time.sleep(wait)

    if resp.status_code == 202:
        request_id = resp.headers.get("nvcf-reqid")
        if not request_id:
            raise requests.HTTPError(
                f"NIM answered 202 (job running) without a request id: {resp.text[:300]}",
                response=resp,
            )
        poll_failures = 0
        while resp.status_code == 202 or (
            resp.status_code in _NIM_RETRY_STATUS and poll_failures < max_retries
        ):
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(
                    f"NIM job {request_id} still running after {max_wait} s"
                )
            if resp.status_code == 202:
                time.sleep(5)
            else:
                time.sleep(_retry_wait(resp, poll_failures))
                poll_failures += 1
            poll_headers = {
                **headers,
                "NVCF-POLL-SECONDS": str(int(min(_NIM_POLL_SECONDS, max(left, 1)))),
            }
            try:
                resp = requests.get(
                    f"{_NIM_STATUS_URL}/{request_id}",
                    headers=poll_headers,
                    timeout=http_timeout,
                )
            except (requests.Timeout, requests.ConnectionError):
                poll_failures += 1  # keep the last answer and poll the same job again
                if poll_failures > max_retries:
                    raise

    if resp.status_code != 200:
        hints = {
            401: "Check NIM_API_KEY.",
            422: "The request was rejected as invalid.",
            429: "Rate limit: still exceeded after all retries.",
        }
        raise requests.HTTPError(
            f"NIM request to {url} failed with HTTP {resp.status_code}. "
            f"{hints.get(resp.status_code, '')}\nResponse: {resp.text[:500]}",
            response=resp,
        )
    return resp


def _provenance_headers(resp: requests.Response) -> dict:
    """The response headers worth keeping for provenance (date, request and version ids)."""
    keep = {
        "date",
        "server",
        "nvcf-reqid",
        "x-request-id",
        "x-nim-version",
        "x-model-version",
    }
    return {k: v for k, v in resp.headers.items() if k.lower() in keep}


def msa_depth(a3m: str) -> int:
    """Number of sequences in an a3m alignment (the query included)."""
    return sum(1 for line in a3m.splitlines() if line.startswith(">"))


def _first_a3m_sequence(a3m: str) -> str:
    """The first (query) sequence of an a3m alignment, without line breaks."""
    lines = []
    for line in a3m.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            if lines:
                break
            continue
        lines.append(line.strip())
    return "".join(lines)


def fetch_msa(
    uniprot_id: str,
    sequence: str,
    save_dir: str = "data/msas",
    nim_api_key: Optional[str] = None,
    databases: tuple[str, ...] = MSA_DATABASES,
    e_value: float = 1e-4,
    max_msa_sequences: int = 500,
) -> dict[str, str]:
    """
    Build the MSA of one sequence with the NVIDIA MSA Search NIM.

    The NIM runs ColabFold's MMseqs2 search (search_type "colabfold") against
    the given databases.  Besides one alignment per database it also returns
    "colabfold", their merge; only the requested databases are returned here,
    so OpenFold3 never gets the same sequences twice.  Every returned a3m is
    saved as <save_dir>/<uniprot_id>/<name>.a3m together with msa_search.json
    (parameters, date, sequence hash, number of sequences per alignment).  If
    those files exist for the same sequence and the same search settings they
    are read back instead of searching again, so re-runs send no request.

    Parameters
    ----------
    uniprot_id        : UniProt accession (folder name).
    sequence          : the query sequence (the same one given to OpenFold3).
    save_dir          : root folder for the MSAs (data/ is not in git).
    nim_api_key       : NIM API key; falls back to os.environ["NIM_API_KEY"].
    databases         : sequence databases to search.
    e_value           : E-value threshold for hits (NIM default 1e-4).
    max_msa_sequences : maximum sequences kept per alignment (NIM default 500).

    Returns
    -------
    {database: a3m text} for the requested databases, ready for
    predict_openfold3(msa=...).
    """
    wanted = {db.lower() for db in databases}
    out_dir = os.path.join(save_dir, uniprot_id)
    info_path = os.path.join(out_dir, "msa_search.json")
    seq_hash = hashlib.sha256(sequence.encode()).hexdigest()
    params = {
        "sequence": sequence,
        "databases": list(databases),
        "search_type": "colabfold",
        "e_value": e_value,
        "max_msa_sequences": max_msa_sequences,
        "output_alignment_formats": ["a3m"],
    }
    settings = {k: v for k, v in params.items() if k != "sequence"}

    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        files = {db: os.path.join(out_dir, f"{db}.a3m") for db in info["databases"]}
        if (
            info.get("sequence_sha256") == seq_hash
            and info.get("parameters") == settings
            and all(os.path.exists(path) for path in files.values())
        ):
            msa = {}
            for db, path in files.items():
                if db.lower() in wanted:
                    with open(path, encoding="utf-8") as f:
                        msa[db] = f.read()
            if msa:
                print(f"  [skip] MSA for {uniprot_id} already exists")
                return msa

    unusual = sorted(set(sequence) - _STANDARD_AA)
    if unusual:
        raise ValueError(
            f"{uniprot_id}: sequence contains {unusual}; the MSA Search NIM accepts "
            "only the 20 standard amino acids.  Handle this protein by hand."
        )

    print(f"  [info] MSA search for {uniprot_id} ({len(sequence)} aa) ...")
    resp = _post_nim(_NIM_MSA_URL, params, _nim_key(nim_api_key))
    result = resp.json()

    returned = {}
    for db, formats in (result.get("alignments") or {}).items():
        text = ((formats or {}).get("a3m") or {}).get("alignment") or ""
        if text.strip():
            returned[db] = text
    msa = {db: text for db, text in returned.items() if db.lower() in wanted}
    if not msa:
        raise ValueError(
            f"MSA Search returned none of {list(databases)} for {uniprot_id}; "
            f"alignments returned: {list(returned)}"
        )

    os.makedirs(out_dir, exist_ok=True)
    for db, text in returned.items():
        with open(os.path.join(out_dir, f"{db}.a3m"), "w", encoding="utf-8") as f:
            f.write(text)
    info = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "endpoint": _NIM_MSA_URL,
        "uniprot_id": uniprot_id,
        "sequence_length": len(sequence),
        "sequence_sha256": seq_hash,
        "parameters": settings,
        "databases": list(returned),
        "n_sequences": {db: msa_depth(text) for db, text in returned.items()},
        "nim_response_headers": _provenance_headers(resp),
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    used = {db: info["n_sequences"][db] for db in msa}
    print(f"  [ok]   MSA saved -> {out_dir}  used: {used}")
    return msa


def _check_existing_of3(out_path: str) -> None:
    """Refuse to reuse an OF3 file that was not predicted with this pipeline's MSA."""
    prov_path = out_path.replace(".cif", "_provenance.json")
    if not os.path.exists(prov_path):
        raise RuntimeError(
            f"{out_path} exists without {os.path.basename(prov_path)}, so it is not "
            "known how it was predicted.  Move it out of the folder and run again."
        )
    with open(prov_path, encoding="utf-8") as f:
        msa_type = json.load(f).get("msa_type", "")
    if "colabfold" not in msa_type:
        raise RuntimeError(
            f"{out_path} was predicted without an MSA.  Delete it (and its "
            "_provenance.json) and run again."
        )


def predict_openfold3(
    uniprot_id: str,
    sequence: str,
    save_dir: str,
    nim_api_key: Optional[str] = None,
    *,
    msa: dict[str, str],
    diffusion_samples: int = 5,
    max_wait: int = 1800,
) -> str:
    """
    Predict the structure of one sequence with OpenFold3 (NVIDIA NIM), using its MSA.

    OpenFold3 reference
    -------------------
    Model:   https://github.com/aqlaboratory/openfold-3
    NIM API: https://build.nvidia.com/openfold/openfold3
    Docs:    https://docs.nvidia.com/nim/bionemo/openfold3/latest/example-requests.html

    Authentication: NIM_API_KEY in the environment (.env file at the repo root,
    loaded with `from dotenv import load_dotenv; load_dotenv()`), or nim_api_key.

    The model returns `diffusion_samples` structures, each with a
    confidence_score.  The top-ranked one (highest confidence_score) is saved
    as OF3_<uniprot_id>.cif; all samples go to <save_dir>/samples/ and all
    scores to OF3_<uniprot_id>_provenance.json.

    Parameters
    ----------
    uniprot_id        : UniProt accession used to name the output file.
    sequence          : the UniProt canonical sequence (fetch_uniprot_sequence).
    save_dir          : directory to save the CIF into (created if absent).
    nim_api_key       : NIM API key; falls back to os.environ["NIM_API_KEY"].
    msa               : {database: a3m text} from fetch_msa (required, keyword only).
                        The first sequence of each a3m must equal `sequence`.
    diffusion_samples : structures generated per request (1-5, default 5).
    max_wait          : seconds to wait for a running job before giving up.

    Returns
    -------
    Full path to the saved OF3_<uniprot_id>.cif file.
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"OF3_{uniprot_id}.cif")
    prov_path = out_path.replace(".cif", "_provenance.json")

    if os.path.exists(out_path):
        _check_existing_of3(out_path)
        print(f"  [skip] OF3_{uniprot_id}.cif already exists")
        return out_path

    if not msa:
        raise ValueError(
            f"{uniprot_id}: no MSA given.  OpenFold3 without an MSA is not a valid "
            "prediction for this study; build one with fetch_msa()."
        )
    if len(msa) > 3:
        raise ValueError("OpenFold3 NIM accepts at most 3 MSA databases per protein")
    for db, a3m in msa.items():
        if _first_a3m_sequence(a3m) != sequence:
            raise ValueError(
                f"{uniprot_id}: the first sequence of the {db} MSA is not the query "
                "sequence (the NIM requires them to be identical)."
            )

    payload = {
        "inputs": [
            {
                "input_id": uniprot_id,
                "molecules": [
                    {
                        "type": "protein",
                        "sequence": sequence,
                        "msa": {
                            db: {"a3m": {"alignment": a3m, "format": "a3m"}}
                            for db, a3m in msa.items()
                        },
                    }
                ],
                "diffusion_samples": diffusion_samples,
                "output_format": "cif",
            }
        ]
    }

    print(
        f"  [info] OpenFold3 for {uniprot_id} ({len(sequence)} aa, "
        f"MSA {', '.join(f'{db}: {msa_depth(a)}' for db, a in msa.items())}) ..."
    )
    resp = _post_nim(_NIM_OF3_URL, payload, _nim_key(nim_api_key), max_wait=max_wait)
    result = resp.json()

    outputs = result.get("outputs") or []
    samples = (outputs[0].get("structures_with_scores") or []) if outputs else []
    if not samples:
        raise ValueError(
            f"NIM OF3 response for {uniprot_id} has no structures; "
            f"top-level keys: {list(result.keys())}"
        )
    for k, sample in enumerate(samples):
        try:
            if not gemmi.cif.read_string(sample.get("structure") or ""):
                raise ValueError("no data block")
        except Exception as exc:
            raise ValueError(
                f"NIM OF3 sample {k} for {uniprot_id} is not valid mmCIF: {exc}"
            ) from exc

    # The model's own ranking: keep the sample with the highest confidence_score.
    scores = [s.get("confidence_score") for s in samples]
    best = max(
        range(len(samples)),
        key=lambda k: scores[k] if scores[k] is not None else float("-inf"),
    )

    # Samples and provenance first; the top model last, in one step, so that
    # OF3_<id>.cif on disk always means a finished run with its provenance.
    samples_dir = os.path.join(save_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    for k, sample in enumerate(samples):
        path = os.path.join(samples_dir, f"OF3_{uniprot_id}_sample{k}.cif")
        with open(path, "w", encoding="utf-8") as f:
            f.write(sample["structure"])

    provenance = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "nim_endpoint_url": _NIM_OF3_URL,
        "uniprot_id": uniprot_id,
        "sequence_length": len(sequence),
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "msa_type": "colabfold_mmseqs2 (NVIDIA MSA Search NIM)",
        "msa_databases": list(msa),
        "msa_n_sequences": {db: msa_depth(a3m) for db, a3m in msa.items()},
        "diffusion_samples": diffusion_samples,
        "selected_sample": best,
        "selection_rule": "highest confidence_score among the returned samples",
        "sample_scores": [
            {k: v for k, v in s.items() if k != "structure"} for s in samples
        ],
        "nim_response_headers": _provenance_headers(resp),
        "output_cif_path": out_path,
    }
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, default=str)
    with open(out_path + ".part", "w", encoding="utf-8") as f:
        f.write(samples[best]["structure"])
    os.replace(out_path + ".part", out_path)

    print(
        f"  [ok]   OF3 saved -> {out_path}  "
        f"(sample {best}, confidence {scores[best]})"
    )
    return out_path


def download_missing_inputs(
    missing_inputs_csv: str,
    data_dir: str = "data",
    nim_api_key: Optional[str] = None,
    skip_of3: bool = False,
) -> dict:
    """
    Download the structure files listed in missing_inputs.csv (written by the
    pipeline when a file is absent, e.g. after a failed request), then re-run
    the pipeline to include them.

    Parameters
    ----------
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
                accession = uniprot_id or pdb_or_uniprot
                seq = fetch_uniprot_sequence(accession)
                msa = fetch_msa(
                    accession,
                    seq,
                    save_dir=os.path.join(data_dir, "msas"),
                    nim_api_key=nim_api_key,
                )
                predict_openfold3(
                    uniprot_id=accession,
                    sequence=seq,
                    save_dir=os.path.join(data_dir, "openfold3"),
                    nim_api_key=nim_api_key,
                    msa=msa,
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
