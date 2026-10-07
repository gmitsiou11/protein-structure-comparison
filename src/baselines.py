"""
Reference levels of the metrics and the robustness of the headline.

The thesis asks whether a prediction differs from a protein's experimental
structures by no more than they differ from each other.  That needs a ruler:
reference distances measured the same way, from the tightest to the loosest,
and an account of what else (besides conformation) changes the distances.
This module computes both from the comparison records.

1. Method pairs.  Every comparison is labelled with its unordered pair of
   methods (X-ray–NMR, NMR–AF2, ...).  Each protein gives one value per pair
   type (the median over its pairs of that type); tables summarise over
   proteins.  The pairs of one protein share structures, so the protein is the
   unit, as in analysis.py.

2. Closest experimental method: for the proteins with X-ray, NMR and cryo-EM
   structures, which of the three is each prediction closest to.

3. The "within experimental variability" rule and its baseline.
     median rule: median(prediction-experiment) <= median(experiment-experiment)
                  (the rule used in analysis.py)
     strict rule: max(prediction-experiment)    <= max(experiment-experiment)
   ``within_by_baseline`` gives the fraction for several definitions of the
   experimental variability side by side (pooled median, pooled strict,
   X-ray–cryo-EM only, NMR–cryo-EM only, the tightest experimental pair).

4. Context.  Most cryo-EM chains come from assemblies, whereas NMR measures
   the free protein.  ``context_table`` splits the distances by the size of the
   cryo-EM entry (polymer chains): if the NMR–cryo-EM distance grows with the
   assembly while X-ray–cryo-EM does not, part of the gap between methods is a
   gap between "alone" and "in a complex".

5. Copies of the same protein inside one entry: the tightest reference (same
   experiment, same map or crystal).  SIFTS says which chains of an entry are
   the same UniProt protein, so no alignment is needed to find them.  Copies
   that are identical (RMSD below SYMMETRY_RMSD) were made identical by imposed
   symmetry and are reported separately: they measure the symmetry, not
   variability.  ``baseline_levels`` and ``reference_levels_table`` put all
   levels side by side (same entry < NMR ensemble < between methods).

6. Coverage.  The Machaon metrics summarise whole chains, so two chains that
   cover different residues differ even when their shared residues have the
   same conformation.  ``coverage_dependence`` measures how strongly each
   metric follows the length difference of a pair; ``pair_medians_by_coverage``
   and the robustness table repeat the analysis on coverage-matched pairs.

7. Robustness.  ``robustness_table`` recomputes the headline fraction under
   every condition that could change it (data availability, coverage, context,
   training overlap, model confidence, flagged structures, baseline definition).
   A statement about the predictions has to hold across these rows.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

METHOD_ORDER = ["X-ray", "NMR", "Cryo-EM", "AF2", "OF3"]
EXPERIMENTAL = ("X-ray", "NMR", "Cryo-EM")
PAIR_ORDER = [
    "X-ray–NMR",
    "X-ray–Cryo-EM",
    "NMR–Cryo-EM",
    "X-ray–AF2",
    "NMR–AF2",
    "Cryo-EM–AF2",
    "X-ray–OF3",
    "NMR–OF3",
    "Cryo-EM–OF3",
    "AF2–OF3",
]
METRICS = ["rmsd", "w_rdist_norm", "b_phipsi", "t_alpha"]
SYMMETRY_RMSD = 0.01  # Å: copies closer than this are identical by construction

# manifest columns of each experimental method
_MANIFEST_COLUMNS = {
    "X-ray": ("pdb_xray", "chain_xray"),
    "NMR": ("pdb_nmr", "chain_nmr"),
    "Cryo-EM": ("pdb_cryoem", "chain_cryoem"),
}


# ── 1. Method pairs ─────────────────────────────────────────────────────────


def pair_label(comparison: str) -> str:
    """'NMR_vs_AF2' -> 'NMR–AF2' (methods in METHOD_ORDER, so the label is unordered)."""
    a, _, b = comparison.partition("_vs_")
    if a not in METHOD_ORDER or b not in METHOD_ORDER:
        raise ValueError(f"Unknown methods in comparison label {comparison!r}")
    a, b = sorted((a, b), key=METHOD_ORDER.index)
    return f"{a}–{b}"


def records_frame(records: list[dict]) -> pd.DataFrame:
    """Computed cross-method comparisons (NMR ensemble records excluded), with a 'pair' column."""
    rows = [
        r
        for r in records
        if r.get("status", "ok") == "ok" and r.get("category") != "nmr_intra_ensemble"
    ]
    df = pd.DataFrame(rows)
    df["pair"] = df["comparison"].map(pair_label)
    return df


def per_protein_pair_medians(
    records: list[dict], metrics: list[str] = METRICS
) -> pd.DataFrame:
    """One row per (protein, pair type): the median of each metric over that protein's pairs."""
    df = records_frame(records)
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df.groupby(["protein_name", "pair"])[metrics].median().reset_index()


def method_pair_table(records: list[dict], metrics: list[str] = METRICS) -> pd.DataFrame:
    """
    Median over proteins of the per-protein medians, for each pair type,
    with the number of proteins.  Rows in PAIR_ORDER.
    """
    per_protein = per_protein_pair_medians(records, metrics)
    table = per_protein.groupby("pair")[metrics].median()
    table["n_proteins"] = per_protein.groupby("pair").size()
    return table.reindex([p for p in PAIR_ORDER if p in table.index])


def closest_experimental_method(
    records: list[dict], prediction: str = "AF2", metric: str = "rmsd"
) -> pd.Series:
    """
    For the proteins whose prediction was compared with all three experimental
    methods: how often each method is the closest one (smallest per-protein
    median of `metric`).  Ties go to the first method in METHOD_ORDER.
    """
    per_protein = per_protein_pair_medians(records, [metric])
    wide = per_protein.pivot(index="protein_name", columns="pair", values=metric)
    columns = [pair_label(f"{method}_vs_{prediction}") for method in EXPERIMENTAL]
    present = [c for c in columns if c in wide.columns]
    if len(present) < len(columns):
        return pd.Series(dtype=int, name=f"closest to {prediction} by {metric}")
    sub = wide[columns].dropna()
    closest = sub.idxmin(axis=1).map(lambda pair: pair.split("–")[0])
    counts = closest.value_counts().reindex(list(EXPERIMENTAL), fill_value=0)
    counts.name = f"closest to {prediction} by {metric} (n={len(sub)})"
    return counts


# ── 3. Baseline choice ──────────────────────────────────────────────────────

BASELINES = [
    "pooled, median rule",
    "pooled, strict rule",
    "X-ray–Cryo-EM only",
    "NMR–Cryo-EM only",
    "tightest experimental pair",
]
_EXPERIMENTAL_PAIRS = ["X-ray–NMR", "X-ray–Cryo-EM", "NMR–Cryo-EM"]


def _bootstrap_fraction(flags: list[bool], n_boot: int, seed: int) -> tuple[float, float]:
    """95 % percentile interval of the mean of a list of booleans (proteins resampled)."""
    if not flags:
        return np.nan, np.nan
    values = np.asarray(flags, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def within_by_baseline(
    records: list[dict],
    metrics: list[str] = METRICS,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Fraction of proteins whose prediction is within experimental variability,
    for several definitions of "experimental variability" (BASELINES).

    For each protein, the prediction's value is the median of its distances to
    the protein's experimental structures; the baseline value is, by definition:
      pooled, median rule        median over the protein's experimental pairs
      pooled, strict rule        maximum over them (prediction: maximum too)
      X-ray–Cryo-EM only         the X-ray–cryo-EM distance (proteins with X-ray)
      NMR–Cryo-EM only           the NMR–cryo-EM distance (all proteins)
      tightest experimental pair the smallest experimental distance of the protein
    "Within" = prediction value <= baseline value.  Returns one row per
    (metric, prediction, baseline): n proteins, fraction, 95 % bootstrap interval.
    """
    df = records_frame(records)
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    rows = []
    for metric in metrics:
        for prediction, category in (("AF2", "exp_vs_af2"), ("OF3", "exp_vs_of3")):
            flags: dict[str, list[bool]] = {b: [] for b in BASELINES}
            for _, g in df.groupby("protein_name"):
                pred = g.loc[g["category"] == category, metric].dropna()
                exp = g.loc[g["category"] == "exp_vs_exp"].set_index("pair")[metric].dropna()
                if pred.empty or exp.empty:
                    continue
                flags["pooled, median rule"].append(bool(pred.median() <= exp.median()))
                flags["pooled, strict rule"].append(bool(pred.max() <= exp.max()))
                flags["tightest experimental pair"].append(bool(pred.median() <= exp.min()))
                if "X-ray–Cryo-EM" in exp.index:
                    flags["X-ray–Cryo-EM only"].append(bool(pred.median() <= exp["X-ray–Cryo-EM"]))
                if "NMR–Cryo-EM" in exp.index:
                    flags["NMR–Cryo-EM only"].append(bool(pred.median() <= exp["NMR–Cryo-EM"]))
            for baseline in BASELINES:
                lo, hi = _bootstrap_fraction(flags[baseline], n_boot, seed)
                rows.append(
                    {
                        "metric": metric,
                        "prediction": prediction,
                        "baseline": baseline,
                        "n": len(flags[baseline]),
                        "fraction": float(np.mean(flags[baseline])) if flags[baseline] else np.nan,
                        "lo": lo,
                        "hi": hi,
                    }
                )
    return pd.DataFrame(rows)


# ── 4. Context: free protein vs part of an assembly ─────────────────────────

ASSEMBLY_BINS = [(1, 4, "1–4 chains"), (5, 19, "5–19 chains"), (20, 10**9, "20+ chains")]
_CONTEXT_PAIRS = ["X-ray–Cryo-EM", "NMR–Cryo-EM", "Cryo-EM–AF2", "NMR–AF2", "X-ray–AF2"]


def cryoem_assembly_size(records: list[dict]) -> pd.Series:
    """Polymer chains in the cryo-EM entry of each protein (from the record metadata)."""
    sizes = {}
    for r in records:
        for side in ("a", "b"):
            if r.get(f"method_{side}") == "Cryo-EM" and r.get(f"n_polymer_chains_{side}") is not None:
                sizes[r["protein_name"]] = int(r[f"n_polymer_chains_{side}"])
    return pd.Series(sizes, name="cryoem_chains")


def context_table(
    records: list[dict], metrics: list[str] = METRICS, pairs: list[str] = _CONTEXT_PAIRS
) -> pd.DataFrame:
    """
    Median distance (over proteins) for selected method pairs, by the size of the
    protein's cryo-EM entry.  Rows: assembly-size bin x pair; columns: n_proteins
    and a median per metric.  If the NMR–cryo-EM distance grows with assembly
    size while X-ray–cryo-EM and cryo-EM–AF2 do not, the NMR gap is context.
    """
    per_protein = per_protein_pair_medians(records, metrics)
    size = cryoem_assembly_size(records)
    per_protein["cryoem_chains"] = per_protein["protein_name"].map(size)
    rows = []
    for low, high, label in ASSEMBLY_BINS:
        in_bin = per_protein[per_protein["cryoem_chains"].between(low, high)]
        for pair in pairs:
            sub = in_bin[in_bin["pair"] == pair]
            if sub.empty:
                continue
            rows.append(
                {
                    "assembly": label,
                    "pair": pair,
                    "n_proteins": len(sub),
                    **{m: float(sub[m].median()) for m in metrics},
                }
            )
    return pd.DataFrame(rows)


# ── 5. Copies of the same protein inside one entry ──────────────────────────


def find_entry_copies(
    manifest: pd.DataFrame, segments: pd.DataFrame, max_copies: int = 5
) -> pd.DataFrame:
    """
    For every experimental structure in the manifest, the other chains of the
    same entry that SIFTS maps to the same UniProt accession.

    manifest : proteins.csv (dtype=str); segments : dataset.load_segments().
    At most `max_copies` other chains per entry, in sorted chain order, so the
    selection is reproducible.  Returns one row per (structure, copy) with
    columns protein_name, uniprot_id, method, pdb_id, chain_ref, chain_copy,
    n_copies (number of chains of that protein in the entry, reference included).
    """
    chains_of = (
        segments.assign(pdb=segments["pdb"].str.lower())
        .groupby(["pdb", "uniprot"])["chain"]
        .agg(lambda c: sorted(set(c)))
    )
    rows = []
    for _, row in manifest.iterrows():
        for method, (pdb_col, chain_col) in _MANIFEST_COLUMNS.items():
            pdb_id = str(row.get(pdb_col, "") or "").strip()
            chain_ref = str(row.get(chain_col, "") or "").strip()
            if not pdb_id or not chain_ref:
                continue
            chains = chains_of.get((pdb_id.lower(), row["uniprot_id"]), [])
            others = [c for c in chains if c != chain_ref][:max_copies]
            for chain_copy in others:
                rows.append(
                    {
                        "protein_name": row["protein_name"],
                        "uniprot_id": row["uniprot_id"],
                        "method": method,
                        "pdb_id": pdb_id.upper(),
                        "chain_ref": chain_ref,
                        "chain_copy": chain_copy,
                        "n_copies": len(chains),
                    }
                )
    columns = [
        "protein_name",
        "uniprot_id",
        "method",
        "pdb_id",
        "chain_ref",
        "chain_copy",
        "n_copies",
    ]
    return pd.DataFrame(rows, columns=columns)


def compare_entry_copies(
    copies: pd.DataFrame, data_dir: str = "data", verbose: bool = False
) -> list[dict]:
    """
    Compare each copy with the reference chain of the same entry (same file,
    two chains) with pipeline.compare_pair: Machaon metrics and RMSD.
    Rows whose file is not on disk give a record with status 'missing_file'.
    """
    from src.pipeline import compare_pair  # imported here: needs the structure stack

    records = []
    for _, row in copies.iterrows():
        path = os.path.join(data_dir, "experimental", f"{row['pdb_id']}.cif")
        info = {
            "copy_method": row["method"],
            "chain_ref": row["chain_ref"],
            "chain_copy": row["chain_copy"],
            "n_copies": int(row["n_copies"]),
        }
        if not os.path.exists(path):
            records.append(
                {
                    "protein_name": row["protein_name"],
                    "uniprot_id": row["uniprot_id"],
                    "pdb_id_a": row["pdb_id"],
                    "status": "missing_file",
                    "rejection_reason": f"{path} not found",
                    **info,
                }
            )
            continue
        rec = compare_pair(
            protein_name=row["protein_name"],
            uniprot_id=row["uniprot_id"],
            path_a=path,
            path_b=path,
            method_cat_a=row["method"],
            method_cat_b=row["method"],
            chain_id_a=row["chain_ref"],
            chain_id_b=row["chain_copy"],
            pdb_id_a=row["pdb_id"],
            pdb_id_b=row["pdb_id"],
            notes="copy of the same protein in the same entry",
            verbose=verbose,
        )
        rec.update(info)
        records.append(rec)
    return records


def summarise_copies(
    copy_records: list[dict], metrics: list[str] = METRICS
) -> pd.DataFrame:
    """
    Per protein and method: number of copy pairs compared, how many are
    identical by construction (RMSD < SYMMETRY_RMSD), and the median of each
    metric over the remaining (non-identical) pairs.
    """
    ok = [r for r in copy_records if r.get("status") == "ok"]
    if not ok:
        return pd.DataFrame(
            columns=["protein_name", "copy_method", "n_pairs", "n_identical", *metrics]
        )
    df = pd.DataFrame(ok)
    for m in metrics:
        df[m] = pd.to_numeric(df.get(m), errors="coerce")
    df["identical"] = df["rmsd"] < SYMMETRY_RMSD
    rows = []
    for (protein, method), group in df.groupby(["protein_name", "copy_method"]):
        varied = group[~group["identical"]]
        row = {
            "protein_name": protein,
            "copy_method": method,
            "n_pairs": len(group),
            "n_identical": int(group["identical"].sum()),
        }
        for m in metrics:
            row[m] = float(varied[m].median()) if len(varied) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_levels(
    protein_table: pd.DataFrame, copy_summary: pd.DataFrame, metric: str
) -> pd.DataFrame:
    """
    Long table (protein, level, value) for one metric, from the tightest to the
    loosest reference, plus the predictions:
      same entry      median distance between copies of the protein in one entry
      NMR ensemble    median distance representative NMR model -> other models
      between methods median distance between experimental structures
      AF2, OF3        median distance prediction -> experimental structures
    For 'same entry', a protein with copies in several entries gets the median
    over its entries.
    """
    levels = []
    if len(copy_summary):
        same = copy_summary.groupby("protein_name")[metric].median().dropna()
        levels += [("same entry", p, v) for p, v in same.items()]
    for label, column in (
        ("NMR ensemble", f"{metric}__nmr_ens"),
        ("between methods", f"{metric}__exp"),
        ("AF2", f"{metric}__af2"),
        ("OF3", f"{metric}__of3"),
    ):
        if column in protein_table:
            values = protein_table[["protein_name", column]].dropna()
            levels += [(label, p, v) for p, v in zip(values.iloc[:, 0], values.iloc[:, 1])]
    return pd.DataFrame(levels, columns=["level", "protein_name", "value"])


# ── 5b. All reference levels in one table ───────────────────────────────────

LEVEL_ORDER = ["same entry", "NMR ensemble", "between methods", "AF2", "OF3"]


def reference_levels_table(
    protein_table: pd.DataFrame,
    copy_summary: pd.DataFrame,
    metrics: list[str] = METRICS,
) -> pd.DataFrame:
    """
    Median over proteins of every reference level (baseline_levels) and of the
    two predictions, one column per metric, plus the number of proteins per
    level (n_<metric>).  Rows in LEVEL_ORDER.
    """
    rows = {}
    for metric in metrics:
        levels = baseline_levels(protein_table, copy_summary, metric)
        grouped = levels.groupby("level")["value"]
        for level, median in grouped.median().items():
            rows.setdefault(level, {})[metric] = float(median)
        for level, n in grouped.size().items():
            rows.setdefault(level, {})[f"n_{metric}"] = int(n)
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table.reindex([lvl for lvl in LEVEL_ORDER if lvl in table.index])


# ── 6. Coverage: whole-chain metrics also see the residues one side lacks ───

COVERAGE_GAPS = (None, 0.10, 0.05)


def length_gap(n_a, n_b) -> float:
    """1 − min(n_a, n_b) / max(n_a, n_b): the share of the longer chain that the shorter one lacks."""
    n_a, n_b = float(n_a or 0), float(n_b or 0)
    if max(n_a, n_b) <= 0:
        return np.nan
    return 1.0 - min(n_a, n_b) / max(n_a, n_b)


def coverage_frame(records: list[dict], metrics: list[str] = METRICS) -> pd.DataFrame:
    """records_frame() with numeric metrics and the length gap of every pair."""
    df = records_frame(records)
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df["length_gap"] = [
        length_gap(a, b) for a, b in zip(df["n_residues_a"], df["n_residues_b"])
    ]
    return df


def coverage_dependence(records: list[dict], metrics: list[str] = METRICS) -> pd.DataFrame:
    """
    Spearman correlation between each metric and the length gap of the pair,
    per comparison category; the pairs are the points (descriptive only: the
    pairs of one protein share structures).

    The Machaon metrics are computed from whole chains, so a pair whose chains
    have different lengths differs even if the shared residues are identical.
    RMSD is computed on matched residues only: its correlation is the reference
    for how much of the dependence is conformation rather than coverage.
    Columns: category, metric, n_pairs, rho.
    """
    df = coverage_frame(records, metrics)
    rows = []
    for category, g in df.groupby("category"):
        for m in metrics:
            sub = g[[m, "length_gap"]].dropna()
            rho = np.nan
            if len(sub) > 2 and sub["length_gap"].nunique() > 1 and sub[m].nunique() > 1:
                rho = float(sub[m].corr(sub["length_gap"], method="spearman"))
            rows.append({"category": category, "metric": m, "n_pairs": len(sub), "rho": rho})
    return pd.DataFrame(rows)


def protein_max_gap(records: list[dict]) -> pd.Series:
    """Largest length gap among the cross-method pairs of each protein (0 = all chains the same length)."""
    df = coverage_frame(records)
    return df.groupby("protein_name")["length_gap"].max().rename("max_length_gap")


def pair_medians_by_coverage(
    records: list[dict],
    max_gaps=COVERAGE_GAPS,
    pairs: list[str] = None,
    metrics: list[str] = METRICS,
) -> pd.DataFrame:
    """
    Median over proteins (of the per-protein medians) of each pair type, using
    only the pairs whose length gap is at most max_gap (None = every pair).
    pairs: pair labels to keep (default: the three experimental pairs).
    Columns: max_gap ("all" or the threshold), pair, n_proteins, <metrics>.
    """
    pairs = pairs or _EXPERIMENTAL_PAIRS
    df = coverage_frame(records, metrics)
    rows = []
    for max_gap in max_gaps:
        sub = df if max_gap is None else df[df["length_gap"] <= max_gap]
        per_protein = sub.groupby(["protein_name", "pair"])[metrics].median().reset_index()
        for pair in pairs:
            p = per_protein[per_protein["pair"] == pair]
            if p.empty:
                continue
            rows.append(
                {
                    "max_gap": "all" if max_gap is None else max_gap,
                    "pair": pair,
                    "n_proteins": len(p),
                    **{m: float(p[m].median()) for m in metrics},
                }
            )
    return pd.DataFrame(rows)


# ── 7. Robustness of the headline ───────────────────────────────────────────

ROBUSTNESS_CONDITIONS = [
    "all proteins",
    "unflagged experimental structures",
    "X-ray, NMR and cryo-EM (3 experimental pairs)",
    "NMR and cryo-EM only (1 experimental pair)",
    "length gap ≤ 10 % in every pair",
    "length gap ≤ 5 % in every pair",
    "cryo-EM entry with 1–4 chains",
    "cryo-EM entry with 20+ chains",
    "no structure public before the model's training cutoff",
    "model confidence pLDDT ≥ 70",
]
BASELINE_CONDITIONS = {
    "pooled, strict rule": "strict rule (largest ≤ largest)",
    "X-ray–Cryo-EM only": "baseline = X-ray–cryo-EM distance",
    "NMR–Cryo-EM only": "baseline = NMR–cryo-EM distance",
    "tightest experimental pair": "baseline = tightest experimental pair",
}


def protein_strata(records: list[dict], structures: pd.DataFrame = None) -> pd.DataFrame:
    """
    One row per protein with every property the robustness table splits on:
    has_xray, n_experimental, cryoem_chains, max_length_gap, flagged,
    plddt_af2, plddt_of3, seen_af2, seen_of3, dated (analysis.training_overlap).
    """
    from src.analysis import experimental_structures, protein_table, training_overlap

    table = protein_table(records)
    structures = experimental_structures(records) if structures is None else structures
    overlap = training_overlap(structures).set_index("protein_name")
    strata = table[
        ["protein_name", "has_xray", "n_experimental", "cryoem_chains", "flagged", "plddt_af2", "plddt_of3"]
    ].set_index("protein_name")
    strata["max_length_gap"] = protein_max_gap(records)
    for column in ("seen_af2", "seen_of3", "dated"):
        strata[column] = overlap[column].reindex(strata.index)
    return strata.reset_index()


def _condition_mask(strata: pd.DataFrame, condition: str, prediction: str) -> pd.Series:
    """Boolean mask over strata rows for one ROBUSTNESS_CONDITIONS entry."""
    s = strata
    unseen = s[f"seen_{prediction.lower()}"].eq(False) & s["dated"].eq(True)
    plddt = pd.to_numeric(s[f"plddt_{prediction.lower()}"], errors="coerce")
    chains = pd.to_numeric(s["cryoem_chains"], errors="coerce")
    has_xray = s["has_xray"].astype(str).str.lower().eq("true")
    masks = {
        "all proteins": pd.Series(True, index=s.index),
        "unflagged experimental structures": ~s["flagged"].astype(bool),
        "X-ray, NMR and cryo-EM (3 experimental pairs)": has_xray,
        "NMR and cryo-EM only (1 experimental pair)": ~has_xray,
        "length gap ≤ 10 % in every pair": s["max_length_gap"] <= 0.10,
        "length gap ≤ 5 % in every pair": s["max_length_gap"] <= 0.05,
        "cryo-EM entry with 1–4 chains": chains.between(1, 4),
        "cryo-EM entry with 20+ chains": chains >= 20,
        "no structure public before the model's training cutoff": unseen,
        "model confidence pLDDT ≥ 70": plddt >= 70,
    }
    return masks[condition].fillna(False).astype(bool)


def robustness_table(
    records: list[dict],
    structures: pd.DataFrame = None,
    metrics: list[str] = METRICS,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    The headline — the fraction of proteins whose prediction is within
    experimental variability (median rule) — under every condition that could
    change it, with 95 % bootstrap intervals over proteins.

      block "proteins"  the same rule on subsets of proteins (ROBUSTNESS_CONDITIONS)
      block "baseline"  all proteins, other definitions of the experimental
                        variability (within_by_baseline)

    Columns: block, condition, metric, prediction, n, fraction, lo, hi.
    """
    from src.analysis import bootstrap_ci, protein_table

    table = protein_table(records).set_index("protein_name")
    strata = protein_strata(records, structures).set_index("protein_name").reindex(table.index)
    rows = []
    for condition in ROBUSTNESS_CONDITIONS:
        for metric in metrics:
            for prediction in ("AF2", "OF3"):
                mask = _condition_mask(strata, condition, prediction)
                sub = table[mask.values]
                diff = (sub[f"{metric}__{prediction.lower()}"] - sub[f"{metric}__exp"]).dropna()
                flags = (diff <= 0).astype(float).to_numpy()
                est, lo, hi = bootstrap_ci(flags, np.mean, n_boot=n_boot, seed=seed)
                rows.append(
                    {
                        "block": "proteins",
                        "condition": condition,
                        "metric": metric,
                        "prediction": prediction,
                        "n": int(len(flags)),
                        "fraction": est,
                        "lo": lo,
                        "hi": hi,
                    }
                )
    base = within_by_baseline(records, metrics, n_boot=n_boot, seed=seed)
    for _, r in base[base["baseline"].isin(BASELINE_CONDITIONS)].iterrows():
        rows.append(
            {
                "block": "baseline",
                "condition": BASELINE_CONDITIONS[r["baseline"]],
                "metric": r["metric"],
                "prediction": r["prediction"],
                "n": int(r["n"]),
                "fraction": r["fraction"],
                "lo": r["lo"],
                "hi": r["hi"],
            }
        )
    return pd.DataFrame(rows)


def robustness_wide(robust: pd.DataFrame, metrics: list[str] = METRICS) -> pd.DataFrame:
    """robustness_table() as one readable row per condition: 'fraction [lo, hi] (n)' per metric and prediction."""

    def fmt(r):
        if not np.isfinite(r["fraction"]):
            return "–"
        if not np.isfinite(r["lo"]):
            return f"{r['fraction']:.2f} (n={int(r['n'])})"
        return f"{r['fraction']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] (n={int(r['n'])})"

    out = robust.assign(value=robust.apply(fmt, axis=1))
    wide = out.pivot_table(
        index=["block", "condition"], columns=["metric", "prediction"], values="value", aggfunc="first"
    )
    order = ROBUSTNESS_CONDITIONS + list(BASELINE_CONDITIONS.values())
    wide = wide.loc[sorted(wide.index, key=lambda bc: order.index(bc[1]) if bc[1] in order else len(order))]
    columns = [(m, p) for m in metrics for p in ("AF2", "OF3") if (m, p) in wide.columns]
    return wide[columns]
