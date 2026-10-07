"""
Statistical analysis of the comparison results, at the level of proteins.

The comparisons of one protein are not independent: the same experimental
structures and predictions appear in several pairs.  So the unit of analysis
is the protein (protein_table).  For each protein and metric:

  exp       median distance between its experimental structures (1 or 3
            pairs): the experimental variability of that protein
  af2, of3  median distance between the prediction and each experimental
            structure (2 or 3 pairs)
  nmr_ens   distance from the representative NMR model (src/nmr.py) to the
            other models of its ensemble

A prediction is "within experimental variability" when af2 <= exp: it is no
farther from the experiments than they are from each other.  Across proteins
(summarise_proteins) the analysis reports the fraction of proteins within experimental
variability, the median of (prediction - experimental) and the paired effect
size d_z, each with a 95 % bootstrap interval over proteins, for all proteins
and for the proteins whose experimental structures are not flagged
(experimental_structures: engineered variants, tags or propeptides, fibrils
or designed cages).  Everything is descriptive: no hypothesis test is run.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

CATEGORY_EXP_VS_EXP = "exp_vs_exp"
CATEGORY_EXP_VS_AF2 = "exp_vs_af2"
CATEGORY_EXP_VS_OF3 = "exp_vs_of3"
CATEGORY_AF2_VS_OF3 = "af2_vs_of3"
CATEGORY_NMR_INTRA = "nmr_intra_ensemble"

# Protein-level analysis: the three Machaon metrics, then RMSD for reference.
METRICS = ["w_rdist_norm", "b_phipsi", "t_alpha", "rmsd"]
EXPERIMENTAL_METHODS = ("X-ray", "NMR", "Cryo-EM")
N_BOOT = 10_000

# Flags for experimental structures that are not a plain copy of the UniProt protein
MIN_IDENTITY = 0.90  # lower: engineered variant
EXTRA_RESIDUES = 10  # residues not matching the UniProt sequence: tags, fusions
STATE_PATTERN = r"\b(?:amyloid|fibrils?|filaments?|cages?)\b"  # title, whole words

# Could a model have been trained on an experimental entry?
AF2_TRAINING_CUTOFF = "2018-04-30"  # AF2: PDB entries released up to this date
OF3_TRAINING_CUTOFF = "2021-09-30"  # OF3: PDB entries deposited before this date


def infer_category(method_a: str, method_b: str) -> str:
    """
    Infer the comparison category from method labels.

    Parameters
    ----------
    method_a, method_b : method_category strings
        Expected values: "X-ray", "NMR", "Cryo-EM", "AF2", "OF3".

    Returns
    -------
    One of the CATEGORY_* constants.

    Raises
    ------
    ValueError for unrecognised AI method labels.
    """
    ai_labels = {"AF2", "OF3", "OpenFold3"}
    a_is_ai = any(label in method_a for label in ai_labels)
    b_is_ai = any(label in method_b for label in ai_labels)

    if a_is_ai and b_is_ai:
        return CATEGORY_AF2_VS_OF3
    if a_is_ai or b_is_ai:
        ai_label = method_a if a_is_ai else method_b
        if "AF2" in ai_label:
            return CATEGORY_EXP_VS_AF2
        if "OF3" in ai_label or "OpenFold" in ai_label:
            return CATEGORY_EXP_VS_OF3
        raise ValueError(f"Unrecognised AI method label: {ai_label!r}")
    return CATEGORY_EXP_VS_EXP


# ---------------------------------------------------------------------------
# Protein-level analysis
# ---------------------------------------------------------------------------


def _values(records: list[dict], metric: str) -> list[float]:
    """The finite values of one metric in a list of records."""
    out = []
    for r in records:
        v = r.get(metric)
        if v is not None and np.isfinite(float(v)):
            out.append(float(v))
    return out


def _median(records: list[dict], metric: str) -> float:
    vals = _values(records, metric)
    return float(np.median(vals)) if vals else np.nan


def _max(records: list[dict], metric: str) -> float:
    vals = _values(records, metric)
    return float(np.max(vals)) if vals else np.nan


def _methods(record: dict) -> tuple[str, str]:
    """The methods of sides a and b, from the record's label, e.g. "NMR_vs_AF2"."""
    method_a, _, method_b = record.get("comparison", "").partition("_vs_")
    return method_a, method_b


def _experimental_side(record: dict) -> str:
    """'a' or 'b': which side of a record is the experimental structure."""
    return "a" if _methods(record)[0] in EXPERIMENTAL_METHODS else "b"


def experimental_structures(records: list[dict]) -> pd.DataFrame:
    """
    One row per experimental structure, described through its comparison with
    the AF2 model, whose sequence is the UniProt sequence (with the OF3 model
    when a protein has no AF2 comparison):

      identity        sequence identity over the aligned residues
      extra_residues  residues of the experimental chain that are not an
                      identical match to the UniProt sequence: expression tags,
                      fused parts, mutations (a tag that replaces unmodelled
                      residues is aligned as mismatches, not gaps)
      n_polymer_chains, title, release/deposition date of the entry

    and the flags (a flagged structure is not a plain copy of the UniProt protein):

      flag_identity  identity < MIN_IDENTITY          (engineered variant)
      flag_extra     extra_residues >= EXTRA_RESIDUES  (tags, fusion)
      flag_state     title matches STATE_PATTERN       (amyloid, fibril, filament, cage)
    """
    columns = [
        "protein_name",
        "uniprot_id",
        "method",
        "pdb_id",
        "chain",
        "n_residues",
        "identity",
        "extra_residues",
        "n_polymer_chains",
        "release_date",
        "deposition_date",
        "title",
    ]
    ok = [r for r in records if r.get("status", "ok") == "ok"]
    with_af2 = {
        r["protein_name"] for r in ok if r.get("category") == CATEGORY_EXP_VS_AF2
    }
    rows = []
    for r in ok:
        category = r.get("category")
        if not (
            category == CATEGORY_EXP_VS_AF2
            or (category == CATEGORY_EXP_VS_OF3 and r["protein_name"] not in with_af2)
        ):
            continue
        s = _experimental_side(r)
        n_residues = int(r.get(f"n_residues_{s}") or 0)
        identical = int(r.get("n_matched") or 0) - int(r.get("n_mismatches") or 0)
        rows.append(
            {
                "protein_name": r["protein_name"],
                "uniprot_id": r.get("uniprot_id"),
                "method": _methods(r)[0 if s == "a" else 1],
                "pdb_id": r.get(f"pdb_id_{s}"),
                "chain": r.get(f"chain_id_{s}"),
                "n_residues": n_residues,
                "identity": r.get("seq_identity"),
                "extra_residues": n_residues - identical,
                "n_polymer_chains": r.get(f"n_polymer_chains_{s}"),
                "release_date": r.get(f"release_date_{s}") or "",
                "deposition_date": r.get(f"deposition_date_{s}") or "",
                "title": r.get(f"title_{s}") or "",
            }
        )
    df = pd.DataFrame(rows, columns=columns)
    df["identity"] = pd.to_numeric(df["identity"])
    df["flag_identity"] = df["identity"] < MIN_IDENTITY
    df["flag_extra"] = df["extra_residues"] >= EXTRA_RESIDUES
    df["flag_state"] = df["title"].str.contains(STATE_PATTERN, case=False, regex=True)
    df["flagged"] = df[["flag_identity", "flag_extra", "flag_state"]].any(axis=1)
    # could the models have been trained on this entry? (see the cutoff constants)
    released = df["release_date"].ne("")
    deposited = df["deposition_date"].ne("")
    df["before_af2_cutoff"] = released & (df["release_date"] <= AF2_TRAINING_CUTOFF)
    df["before_of3_cutoff"] = deposited & (df["deposition_date"] < OF3_TRAINING_CUTOFF)
    return df


def training_overlap(structures: pd.DataFrame) -> pd.DataFrame:
    """
    Per protein: could the models have seen it during training?

    `structures` has one row per experimental structure with protein_name,
    release_date, deposition_date and the flags before_af2_cutoff /
    before_of3_cutoff (experimental_structures() or dataset.describe_structures()).

      n_structures   experimental structures of the protein in this dataset
      n_before_af2   of them released on or before the AF2 training cutoff
      n_before_of3   of them deposited before the OF3 training cutoff
      seen_af2       at least one structure public before the AF2 cutoff
      seen_of3       at least one structure deposited before the OF3 cutoff
      dated          every structure of the protein has a known date

    A "seen" protein may have been in the model's training data, so agreement
    with its experimental structures can partly be recall rather than
    prediction.  Only the structures in this dataset are counted; other PDB
    entries of the same protein can only turn an "unseen" protein into a
    "seen" one, so the number of unseen proteins is an upper bound.
    """
    if structures.empty:
        return pd.DataFrame(
            columns=[
                "protein_name",
                "n_structures",
                "n_before_af2",
                "n_before_of3",
                "seen_af2",
                "seen_of3",
                "dated",
            ]
        )
    s = structures.copy()
    for column in ("release_date", "deposition_date"):
        s[column] = s[column].fillna("").astype(str)
    s["dated"] = s["release_date"].ne("") & s["deposition_date"].ne("")
    out = s.groupby("protein_name").agg(
        n_structures=("pdb_id", "size"),
        n_before_af2=("before_af2_cutoff", "sum"),
        n_before_of3=("before_of3_cutoff", "sum"),
        dated=("dated", "all"),
    )
    out["n_before_af2"] = out["n_before_af2"].astype(int)
    out["n_before_of3"] = out["n_before_of3"].astype(int)
    out["seen_af2"] = out["n_before_af2"] > 0
    out["seen_of3"] = out["n_before_of3"] > 0
    return out.reset_index()[
        [
            "protein_name",
            "n_structures",
            "n_before_af2",
            "n_before_of3",
            "seen_af2",
            "seen_of3",
            "dated",
        ]
    ]


def protein_table(records: list[dict]) -> pd.DataFrame:
    """
    One row per protein.  For every metric m (columns "<m>__<quantity>"):

      exp          median over the experimental pairs (1 or 3): experimental variability
      af2, of3     median over prediction-vs-experimental pairs (2 or 3)
      af2_of3      AF2 vs OF3
      nmr_ens      median distance representative NMR model -> other models;
                   nmr_ens_max its maximum
      nmr_af2, nmr_of3   distance representative NMR model -> prediction

    plus covariates (length and mean pLDDT of each prediction, OF3 MSA depth,
    chains in the cryo-EM entry, X-ray available) and the flags of the
    protein's experimental structures.  Needs the NMR
    intra-ensemble records too (they give nmr_ens).
    """
    by_protein: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("status", "ok") == "ok":
            by_protein[r["protein_name"]].append(r)

    structures = experimental_structures(records)
    flags = structures.groupby("protein_name")[
        ["flag_identity", "flag_extra", "flag_state", "flagged"]
    ].any()
    cryoem_chains = (
        structures[structures["method"] == "Cryo-EM"]
        .groupby("protein_name")["n_polymer_chains"]
        .max()
    )

    rows = []
    for protein, recs in sorted(by_protein.items()):
        group = defaultdict(list)
        for r in recs:
            group[r.get("category")].append(r)
        af2 = group[CATEGORY_EXP_VS_AF2]
        of3 = group[CATEGORY_EXP_VS_OF3]
        nmr_af2 = [r for r in af2 if "NMR" in _methods(r)]
        nmr_of3 = [r for r in of3 if "NMR" in _methods(r)]
        predicted = af2 or of3  # both are the whole UniProt sequence
        methods = {
            m for r in predicted for m in _methods(r) if m in EXPERIMENTAL_METHODS
        }

        row: dict = {
            "protein_name": protein,
            "uniprot_id": recs[0].get("uniprot_id"),
            "has_xray": "X-ray" in methods,
            "n_experimental": len(methods),
            "length": predicted[0].get("n_residues_b") if predicted else np.nan,
            "plddt_af2": af2[0].get("plddt_mean_b") if af2 else np.nan,
            "plddt_of3": of3[0].get("plddt_mean_b") if of3 else np.nan,
            "msa_depth_of3": of3[0].get("msa_depth_b") if of3 else np.nan,
            "cryoem_chains": cryoem_chains.get(protein, np.nan),
        }
        for m in METRICS:
            row[f"{m}__exp"] = _median(group[CATEGORY_EXP_VS_EXP], m)
            row[f"{m}__af2"] = _median(af2, m)
            row[f"{m}__of3"] = _median(of3, m)
            row[f"{m}__af2_of3"] = _median(group[CATEGORY_AF2_VS_OF3], m)
            row[f"{m}__nmr_ens"] = _median(group[CATEGORY_NMR_INTRA], m)
            row[f"{m}__nmr_ens_max"] = _max(group[CATEGORY_NMR_INTRA], m)
            row[f"{m}__nmr_af2"] = _median(nmr_af2, m)
            row[f"{m}__nmr_of3"] = _median(nmr_of3, m)
        for flag in ("flag_identity", "flag_extra", "flag_state", "flagged"):
            row[flag] = bool(flags[flag].get(protein, False)) if len(flags) else False
        rows.append(row)

    return pd.DataFrame(rows)


def _mean(x: np.ndarray, axis=None):
    return np.mean(x, axis=axis)


def _paired_dz(x: np.ndarray, axis=None):
    """Paired effect size d_z = mean / standard deviation of the differences."""
    mean = np.mean(x, axis=axis)
    sd = np.std(x, axis=axis, ddof=1)
    tiny = sd <= 1e-12 * np.maximum(1.0, np.abs(mean))  # all values equal: undefined
    with np.errstate(divide="ignore", invalid="ignore"):
        d = mean / sd
    return np.where(tiny | ~np.isfinite(d), np.nan, d)


def bootstrap_ci(
    values,
    statistic=np.median,
    n_boot: int = N_BOOT,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """
    Estimate and percentile bootstrap interval of statistic(values).

    `values` has one entry per protein; the proteins are resampled with
    replacement n_boot times (fixed seed: the same numbers on every run).
    `statistic` must accept an `axis` argument (np.median, np.mean, ...).
    Returns (estimate, low, high); low/high are NaN with fewer than 2 values.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    estimate = float(statistic(x))
    if len(x) < 2:
        return estimate, np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = statistic(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    samples = np.asarray(samples, dtype=float)
    if not np.isfinite(samples).any():  # e.g. d_z of identical values
        return estimate, np.nan, np.nan
    low, high = np.nanpercentile(samples, [50 * (1 - level), 50 * (1 + level)])
    return estimate, float(low), float(high)


def summarise_proteins(table: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """
    Across proteins, for each metric and prediction (AF2, OF3), on all proteins
    and on the unflagged ("clean") ones:

      n                 proteins with a value
      within_exp        fraction with prediction <= experimental variability
                        (prediction no farther from the experiments than they
                        are from each other)
      median_diff       median of (prediction - experimental), metric units
      d_z               paired effect size of (prediction - experimental)
      inside_nmr        fraction with d(NMR representative, prediction) <= the
                        largest d(NMR representative, other model): no farther
                        from the representative than the farthest model of its
                        own ensemble (n_nmr proteins: those whose NMR entry has
                        more than one model)
      af2_closer        (row "AF2 vs OF3") fraction where AF2 is closer to the
                        experiments than OF3; median_diff = median(OF3 - AF2)

    Every value comes with a 95 % bootstrap interval (<name>_lo, <name>_hi).
    """
    rows = []
    subsets = [("all", table), ("clean", table[~table["flagged"].astype(bool)])]
    for subset, sub in subsets:
        for m in METRICS:
            for pred in ("af2", "of3"):
                diff = (sub[f"{m}__{pred}"] - sub[f"{m}__exp"]).dropna().to_numpy()
                ens = sub[[f"{m}__nmr_{pred}", f"{m}__nmr_ens_max"]].dropna()
                inside = (ens.iloc[:, 0] <= ens.iloc[:, 1]).to_numpy(dtype=float)
                row = {"subset": subset, "metric": m, "prediction": pred.upper()}
                row["n"] = len(diff)
                row["n_nmr"] = len(inside)  # proteins with an NMR ensemble (>1 model)
                for name, values, stat in (
                    ("within_exp", (diff <= 0).astype(float), _mean),
                    ("median_diff", diff, np.median),
                    ("d_z", diff, _paired_dz),
                    ("inside_nmr", inside, _mean),
                ):
                    est, lo, hi = bootstrap_ci(values, stat, n_boot=n_boot)
                    row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
                rows.append(row)

            diff = (sub[f"{m}__of3"] - sub[f"{m}__af2"]).dropna().to_numpy()
            row = {"subset": subset, "metric": m, "prediction": "AF2 vs OF3"}
            row["n"] = len(diff)
            for name, values, stat in (
                ("af2_closer", (diff > 0).astype(float), _mean),
                ("median_diff", diff, np.median),
            ):
                est, lo, hi = bootstrap_ci(values, stat, n_boot=n_boot)
                row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
            rows.append(row)
    return pd.DataFrame(rows)


def _interval(row: pd.Series, name: str, fmt: str = "{:.2f}") -> str:
    """'0.62 [0.54, 0.70]' for one estimate and its interval."""
    est = row.get(name)
    if est is None or not np.isfinite(est):
        return "-"
    lo, hi = row.get(f"{name}_lo"), row.get(f"{name}_hi")
    if lo is None or not np.isfinite(lo):
        return fmt.format(est)
    return f"{fmt.format(est)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def format_summary(summary: pd.DataFrame, subset: str = "all") -> pd.DataFrame:
    """The summarise_proteins() table as readable strings, one subset."""
    out = []
    for _, row in summary[summary["subset"] == subset].iterrows():
        if row["prediction"] == "AF2 vs OF3":
            continue
        out.append(
            {
                "metric": row["metric"],
                "prediction": row["prediction"],
                "n": int(row["n"]),
                "within experimental variability": _interval(row, "within_exp"),
                "median (prediction - experimental)": _interval(
                    row, "median_diff", "{:.3g}"
                ),
                "d_z": _interval(row, "d_z"),
                "inside NMR ensemble": _interval(row, "inside_nmr"),
                "n (NMR ensemble)": int(row["n_nmr"]),
            }
        )
    return pd.DataFrame(out)


def _to_json_rows(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with NaN replaced by None (valid JSON)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def analyse(records: list[dict]) -> dict:
    """
    Protein-level analysis of all comparison records (status "ok"), including
    the NMR intra-ensemble records.

    Returns a JSON-serialisable dict:
      experimental_structures  one row per experimental structure, with flags
      per_protein              protein_table() rows
      summary                  summarise_proteins() rows
      summary_text             printable summary (print_summary)
    """
    structures = experimental_structures(records)
    table = protein_table(records)
    summary = summarise_proteins(table) if len(table) else pd.DataFrame()

    n_flagged = int(table["flagged"].sum()) if len(table) else 0
    lines = [
        "=" * 78,
        "PROTEIN-LEVEL SUMMARY  (descriptive; 95 % bootstrap intervals over proteins)",
        "=" * 78,
        f"Proteins: {len(table)}   flagged: {n_flagged}   clean: {len(table) - n_flagged}",
    ]
    if len(table):
        lines.append(
            "  flags: low identity {}, extra residues {}, fibril/cage title {}".format(
                int(table["flag_identity"].sum()),
                int(table["flag_extra"].sum()),
                int(table["flag_state"].sum()),
            )
        )
    lines += [
        "",
        "within exp = fraction of proteins whose prediction is no farther from the",
        "  experimental structures than they are from each other (median distances).",
        "inside NMR = fraction whose prediction is no farther from the representative",
        "  NMR model than the farthest other model of the same ensemble (n NMR =",
        "  proteins whose NMR entry has more than one model).",
        "median diff = median over proteins of (prediction - experimental).",
    ]
    for subset in ("all", "clean"):
        if summary.empty:
            break
        lines += ["", f"--- {subset} proteins ---"]
        lines.append(
            f"{'metric':<13} {'pred':<4} {'n':>4}  {'within exp':<19} "
            f"{'median diff':<26} {'inside NMR':<19} {'n NMR':>5}"
        )
        for _, row in summary[summary["subset"] == subset].iterrows():
            if row["prediction"] == "AF2 vs OF3":
                lines.append(
                    f"{row['metric']:<13} AF2 closer than OF3 in "
                    f"{_interval(row, 'af2_closer')} of {int(row['n'])} proteins"
                )
                continue
            lines.append(
                f"{row['metric']:<13} {row['prediction']:<4} {int(row['n']):>4}  "
                f"{_interval(row, 'within_exp'):<19} "
                f"{_interval(row, 'median_diff', '{:.3g}'):<26} "
                f"{_interval(row, 'inside_nmr'):<19} {int(row['n_nmr']):>5}"
            )
    lines.append("=" * 78)

    return {
        "experimental_structures": _to_json_rows(structures),
        "per_protein": _to_json_rows(table),
        "summary": _to_json_rows(summary),
        "summary_text": "\n".join(lines),
    }


def summarise_rejections(rejected_records: list[dict]) -> dict:
    """
    Count rejected and failed comparison records by status and by reason.

    Parameters
    ----------
    rejected_records : records from pipeline.py with status != "ok".

    Returns
    -------
    dict with:
      total              : int  — number of records
      by_status          : dict — counts per status ("rejected", "metric_error")
      by_rejection_class : dict — counts per reason class (CLASS_PATTERNS, else "other")
      printable_summary  : str  — the same as a text table
    """
    import re

    total = len(rejected_records)

    by_status: dict[str, int] = {}
    for r in rejected_records:
        s = r.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    # the reasons written by alignment.align_pair and pipeline.compare_pair
    CLASS_PATTERNS = [
        ("parse_error", r"Failed to parse"),
        ("too_few_residues", r"has only \d+ Cα residues"),
        ("alignment_error", r"Sequence alignment failed"),
        ("metric_error", r"Metric computation failed"),
    ]

    by_class: dict[str, int] = {}
    for r in rejected_records:
        reason = r.get("rejection_reason") or r.get("reason") or ""
        label = next(
            (name for name, pattern in CLASS_PATTERNS if re.search(pattern, reason)),
            "other",
        )
        by_class[label] = by_class.get(label, 0) + 1

    lines = [
        "=" * 60,
        "REJECTED COMPARISONS SUMMARY",
        "=" * 60,
        f"Total rejected / failed : {total}",
        "",
        "By pipeline status:",
    ]
    for s, n in sorted(by_status.items()):
        lines.append(f"  {s:<20s}: {n}")
    lines.append("")
    lines.append("By rejection class (from reason field):")
    for cls, n in sorted(by_class.items(), key=lambda x: -x[1]):
        lines.append(f"  {cls:<25s}: {n}")
    lines += [
        "",
        "Note: pairs are rejected only when a structure cannot be read",
        "(parse errors, missing chain or model, < 5 residues).  Coverage and",
        "identity are reported as warnings, not used to reject pairs.",
        "=" * 60,
    ]

    return {
        "total": total,
        "by_status": by_status,
        "by_rejection_class": by_class,
        "printable_summary": "\n".join(lines),
    }
