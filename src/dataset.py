"""
Build the protein manifest (proteins.csv) from public reference tables.

Every choice is a rule applied to data: which proteins (NMR and cryo-EM both
available, anchored on the scarcest method), which chain of which entry
(one chain per method, chosen by resolution/coverage), and how many proteins
survive each filter (reported, PoseBusters-style).
"""

from __future__ import annotations

import gzip
import io
import os
import re
import time

import numpy as np
import pandas as pd
import requests

METHOD_MAP = {"diffraction": "X-ray", "NMR": "NMR", "EM": "Cryo-EM"}
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
REFERENCE_FILES = {
    "uniprot_segments_observed.csv.gz": "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/uniprot_segments_observed.csv.gz",
    "pdb_entry_type.txt": "https://files.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt",
    "resolu.idx": "https://files.wwpdb.org/pub/pdb/derived_data/index/resolu.idx",
}
UNIPROT_FIELDS = (
    "accession,length,protein_name,organism_name,ft_signal,ft_propep,ft_chain"
)


# ── 1. Reference tables ─────────────────────────────────────────────────────


def download_reference_files(
    ref_dir: str = "data/reference", force: bool = False
) -> dict:
    """Download the three public tables into ref_dir (skipped if present). Returns {name: path}."""
    os.makedirs(ref_dir, exist_ok=True)
    paths = {}
    for name, url in REFERENCE_FILES.items():
        path = os.path.join(ref_dir, name)
        paths[name] = path
        if os.path.exists(path) and not force:
            print(f"  [skip] {name} already present (force=True to refresh)")
            continue
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        print(f"  [ok]   {name}")
    return paths


def sifts_version(path: str) -> str:
    """First line of the SIFTS file, i.e. its release date: report it as the dataset version."""
    with gzip.open(path, "rt") as f:
        return f.readline().lstrip("# ").strip()


def load_segments(path: str) -> pd.DataFrame:
    """SIFTS observed segments: modelled residue ranges of each PDB chain in UniProt numbering."""
    df = pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)
    df = df.rename(
        columns={
            "PDB": "pdb",
            "CHAIN": "chain",
            "SP_PRIMARY": "uniprot",
            "SP_BEG": "sp_beg",
            "SP_END": "sp_end",
        }
    )
    df["sp_beg"] = pd.to_numeric(df["sp_beg"], errors="coerce")
    df["sp_end"] = pd.to_numeric(df["sp_end"], errors="coerce")
    df = df.dropna(subset=["sp_beg", "sp_end"])
    df = df[df["sp_end"] >= df["sp_beg"]]  # a few SIFTS rows are reversed
    n_proteins = df.groupby(["pdb", "chain"])["uniprot"].transform("nunique")
    df = df[
        (n_proteins == 1) & ~df["uniprot"].str.contains("-")
    ]  # no chimeras, no isoforms
    return df[["pdb", "chain", "uniprot", "sp_beg", "sp_end"]]


def load_methods(path: str) -> pd.DataFrame:
    """wwPDB pdb_entry_type.txt: experimental method of each entry."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["pdb", "mol_type", "method"],
        dtype=str,
        keep_default_na=False,
    )
    df = df[df["mol_type"].isin(["prot", "prot-nuc"])]
    df["method"] = df["method"].map(METHOD_MAP)
    return df.dropna(subset=["method"])[["pdb", "method"]]


def load_resolution(path: str) -> pd.DataFrame:
    """wwPDB resolu.idx: resolution of each entry (NaN for NMR / not reported)."""
    rows = []
    with open(path) as f:
        for line in f:
            m = re.match(r"^(\w{4})\s*;\s*(-?\d+(?:\.\d+)?)?\s*$", line)
            if m:
                value = float(m.group(2)) if m.group(2) else np.nan
                rows.append((m.group(1).lower(), value if value > 0 else np.nan))
    return pd.DataFrame(rows, columns=["pdb", "resolution"])


def chain_table(
    segments, methods, resolution, max_xray=2.5, max_em=4.0
) -> pd.DataFrame:
    """One row per (entry, chain, protein) within the quality limits."""
    chains = segments[["pdb", "chain", "uniprot"]].drop_duplicates()
    chains = chains.merge(methods, on="pdb").merge(resolution, on="pdb", how="left")
    ok = (
        (chains["method"] == "NMR")
        | ((chains["method"] == "X-ray") & (chains["resolution"] <= max_xray))
        | ((chains["method"] == "Cryo-EM") & (chains["resolution"] <= max_em))
    )
    return chains[ok]


def proteins_with(chains, methods=("NMR", "Cryo-EM")) -> list[str]:
    """UniProt accessions that have at least one chain of every method in `methods`."""
    per_protein = chains.groupby("uniprot")["method"].agg(set)
    return sorted(per_protein[per_protein.apply(lambda s: set(methods) <= s)].index)


# ── 2. Protein facts from UniProt ───────────────────────────────────────────


def parse_mature(chain_field: str, length: int) -> tuple[int, int, int]:
    """(start, end, n_chains) from UniProt's Chain column, e.g. 'CHAIN 48..157; /note=...'."""
    spans = re.findall(r"CHAIN\s+[<?]?(\d+|\?)\.\.[>?]?(\d+|\?)", chain_field or "")
    if not spans:
        return 1, int(length), 0
    start, end = spans[0]
    start = int(start) if start != "?" else 1
    end = int(end) if end != "?" else int(length)
    return start, end, len(spans)


def fetch_uniprot_facts(
    accessions, cache_path="data/reference/uniprot_facts.csv", batch=100
):
    """Length, name, organism and mature chain of each accession (cached on disk)."""
    cached = (
        pd.read_csv(cache_path, dtype=str, keep_default_na=False)
        if os.path.exists(cache_path)
        else pd.DataFrame(columns=["Entry"])
    )
    todo = [a for a in accessions if a not in set(cached["Entry"])]
    frames = [cached]
    for i in range(0, len(todo), batch):
        query = " OR ".join(f"accession:{a}" for a in todo[i : i + batch])
        r = requests.get(
            UNIPROT_SEARCH,
            params={
                "query": query,
                "fields": UNIPROT_FIELDS,
                "format": "tsv",
                "size": 500,
            },
            timeout=60,
        )
        r.raise_for_status()
        frames.append(
            pd.read_csv(io.StringIO(r.text), sep="\t", dtype=str, keep_default_na=False)
        )
        time.sleep(0.5)
    raw = pd.concat(frames, ignore_index=True).drop_duplicates("Entry")
    raw.to_csv(cache_path, index=False)
    facts = raw.rename(
        columns={
            "Entry": "uniprot",
            "Length": "length",
            "Protein names": "protein_name",
            "Organism": "organism",
            "Chain": "chain_field",
        }
    )
    facts = facts[facts["uniprot"].isin(accessions)].copy()
    facts["length"] = facts["length"].astype(int)
    mature = [parse_mature(c, n) for c, n in zip(facts["chain_field"], facts["length"])]
    facts[["mature_start", "mature_end", "n_chains"]] = pd.DataFrame(
        mature, index=facts.index
    )
    facts["mature_length"] = facts["mature_end"] - facts["mature_start"] + 1
    return facts


def has_alphafold_model(accession: str) -> bool:
    """True if AlphaFold DB has a prediction for this accession (same API as the downloader)."""
    r = requests.get(
        f"https://alphafold.ebi.ac.uk/api/prediction/{accession}", timeout=30
    )
    return r.status_code == 200 and len(r.json()) > 0


# ── 3. Coverage, filters, representatives ───────────────────────────────────


def add_coverage(chains, segments, facts) -> pd.DataFrame:
    """Fraction of the mature chain that each PDB chain actually models."""
    seg = segments.merge(facts[["uniprot", "mature_start", "mature_end"]], on="uniprot")
    seg["overlap"] = (
        np.minimum(seg["sp_end"], seg["mature_end"])
        - np.maximum(seg["sp_beg"], seg["mature_start"])
        + 1
    ).clip(lower=0)
    covered = (
        seg.groupby(["pdb", "chain", "uniprot"])["overlap"]
        .sum()
        .rename("covered")
        .reset_index()
    )
    out = chains.merge(covered, on=["pdb", "chain", "uniprot"])
    out = out.merge(
        facts[["uniprot", "mature_start", "mature_end", "mature_length", "n_chains"]],
        on="uniprot",
    )
    out["coverage"] = out["covered"] / out["mature_length"]
    return out


def filter_with_counts(df, steps, start_label="start", key="uniprot"):
    """Apply (name, function) steps in order; return the result and the count after each step."""
    counts = [(start_label, df[key].nunique())]
    for name, fn in steps:
        df = fn(df)
        counts.append((name, df[key].nunique()))
    return df, pd.DataFrame(counts, columns=["step", "proteins"])


def pick_representatives(chains) -> pd.DataFrame:
    """One chain per method per protein: best resolution (X-ray, cryo-EM) or best coverage (NMR)."""
    rows = []
    for uniprot, g in chains.groupby("uniprot"):
        row = {"uniprot_id": uniprot}
        for method, col in [("X-ray", "xray"), ("NMR", "nmr"), ("Cryo-EM", "cryoem")]:
            m = g[g["method"] == method]
            row.update(
                {
                    f"pdb_{col}": "",
                    f"chain_{col}": "",
                    f"cov_{col}": "",
                    f"res_{col}": "",
                }
            )
            if m.empty:
                continue
            if method == "NMR":
                best = m.sort_values(
                    ["coverage", "pdb", "chain"], ascending=[False, True, True]
                ).iloc[0]
            else:
                best = m.sort_values(
                    ["resolution", "coverage", "pdb", "chain"],
                    ascending=[True, False, True, True],
                ).iloc[0]
                row[f"res_{col}"] = best["resolution"]
            row[f"pdb_{col}"], row[f"chain_{col}"] = best["pdb"].upper(), best["chain"]
            row[f"cov_{col}"] = round(best["coverage"], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def write_manifest(reps, facts, path="proteins.csv") -> pd.DataFrame:
    m = reps.merge(facts, left_on="uniprot_id", right_on="uniprot")
    m["protein_name"] = m["protein_name"].str.split(" (", regex=False).str[0]
    # Different proteins can share a UniProt name (e.g. the same ribosomal protein
    # in two organisms); names are used as keys downstream, so make them unique.
    shared = m["protein_name"].duplicated(keep=False)
    m.loc[shared, "protein_name"] = m["protein_name"] + " (" + m["uniprot_id"] + ")"
    m["notes"] = ""
    cols = [
        "protein_name",
        "uniprot_id",
        "organism",
        "mature_start",
        "mature_end",
        "pdb_xray",
        "chain_xray",
        "pdb_nmr",
        "chain_nmr",
        "pdb_cryoem",
        "chain_cryoem",
        "cov_xray",
        "cov_nmr",
        "cov_cryoem",
        "res_xray",
        "res_cryoem",
        "notes",
    ]
    m[cols].to_csv(path, index=False)
    return m[cols]


# ── 4. What the chosen structures are ───────────────────────────────────────

_METHOD_COLUMNS = {
    "X-ray": ("pdb_xray", "chain_xray"),
    "NMR": ("pdb_nmr", "chain_nmr"),
    "Cryo-EM": ("pdb_cryoem", "chain_cryoem"),
}


def describe_structures(manifest: pd.DataFrame, data_dir: str = "data") -> pd.DataFrame:
    """
    One row per experimental structure of the manifest, read from its file:
    protein_name, uniprot_id, method, pdb_id, chain, n_residues (Cα of the
    chain), n_models (models in the file), n_polymer_chains (1 = the protein alone, more = determined
    inside a complex), resolution, release_date, deposition_date, title, and
    whether each model could have seen the entry in training
    (before_af2_cutoff, before_of3_cutoff; cutoffs in src/analysis.py).
    Files that are not on disk get status "missing".
    """
    from src.analysis import AF2_TRAINING_CUTOFF, OF3_TRAINING_CUTOFF
    from src.metadata import extract_metadata
    from src.parser import load_structure

    rows = []
    for _, row in manifest.iterrows():
        for method, (pdb_col, chain_col) in _METHOD_COLUMNS.items():
            pdb_id = str(row.get(pdb_col, "") or "").strip().upper()
            chain = str(row.get(chain_col, "") or "").strip()
            if not pdb_id:
                continue
            path = os.path.join(data_dir, "experimental", f"{pdb_id}.cif")
            base = {
                "protein_name": row["protein_name"],
                "uniprot_id": row["uniprot_id"],
                "method": method,
                "pdb_id": pdb_id,
                "chain": chain,
            }
            if not os.path.exists(path):
                rows.append({**base, "status": "missing"})
                continue
            meta = extract_metadata(path, uniprot_id=row["uniprot_id"], pdb_id=pdb_id, chain_id=chain)
            rows.append(
                {
                    **base,
                    "status": "ok",
                    "n_residues": meta.n_residues,
                    "n_models": len(load_structure(path)),  # models in the file
                    "n_polymer_chains": meta.n_polymer_chains,
                    "resolution": meta.resolution,
                    "release_date": meta.release_date or "",
                    "deposition_date": meta.deposition_date or "",
                    "title": meta.title or "",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for column in ("release_date", "deposition_date", "title"):
        if column not in df:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)
    df["before_af2_cutoff"] = df["release_date"].ne("") & (df["release_date"] <= AF2_TRAINING_CUTOFF)
    df["before_of3_cutoff"] = df["deposition_date"].ne("") & (df["deposition_date"] < OF3_TRAINING_CUTOFF)
    return df


def composition(manifest: pd.DataFrame) -> pd.DataFrame:
    """How many proteins have which combination of experimental methods (from the manifest)."""
    combos = []
    for _, row in manifest.iterrows():
        methods = [m for m, (col, _) in _METHOD_COLUMNS.items() if str(row.get(col, "") or "").strip()]
        combos.append(" + ".join(methods))
    counts = pd.Series(combos).value_counts()
    return pd.DataFrame({"experimental methods": counts.index, "proteins": counts.to_numpy()})
