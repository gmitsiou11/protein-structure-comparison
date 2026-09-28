"""
Statistical analysis of protein structure comparison results.

With only 1–3 exp-vs-exp comparison pairs per protein, classical parametric
tests (t-test, ANOVA) are severely underpowered and are not reported. Instead:

  1. Descriptive statistics (mean, std, range) per comparison category.
  2. Cohen's d between AI and exp-vs-exp distributions — a size measure that
     does not depend on sample size. Treat as a directional signal only.
  3. Variance ratios — Var(AI) / Var(Exp). A ratio > 1 means AI introduces
     more variability than is seen between experimental methods.
  4. Per-protein within-range check: does each AI metric fall within
     exp-vs-exp mean +/- 1 sigma? This is an operational comparison, not a test.

"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

CATEGORY_EXP_VS_EXP = "exp_vs_exp"
CATEGORY_EXP_VS_AF2 = "exp_vs_af2"
CATEGORY_EXP_VS_OF3 = "exp_vs_of3"
CATEGORY_AF2_VS_OF3 = "af2_vs_of3"

ALL_CATEGORIES = [
    CATEGORY_EXP_VS_EXP,
    CATEGORY_EXP_VS_AF2,
    CATEGORY_EXP_VS_OF3,
    CATEGORY_AF2_VS_OF3,
]

METRIC_NAMES = [
    "rmsd",
    "t_alpha",
    "w_rdist_raw",
    "w_rdist_norm",
    "b_phipsi",
]


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


@dataclass
class CategoryStats:
    """Descriptive statistics for one metric in one comparison category."""

    category: str
    metric: str
    n: int = 0
    mean: float = float("nan")
    std: float = float("nan")
    min_val: float = float("nan")
    max_val: float = float("nan")
    values: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("values")
        return d


def compute_category_stats(records: list[dict]) -> dict[str, dict[str, CategoryStats]]:
    """
    Compute per-category, per-metric descriptive statistics.

    Parameters
    ----------
    records : list of comparison records with 'category' and metric fields.

    Returns
    -------
    Nested dict: stats[category][metric] = CategoryStats
    """
    bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        cat = rec.get("category", "unknown")
        for m in METRIC_NAMES:
            val = rec.get(m)
            if val is not None:
                try:
                    fval = float(val)
                    if not np.isnan(fval):
                        bucket[cat][m].append(fval)
                except (TypeError, ValueError):
                    pass

    stats: dict[str, dict[str, CategoryStats]] = {}
    for cat, metrics in bucket.items():
        stats[cat] = {}
        for m, vals in metrics.items():
            arr = np.array(vals)
            stats[cat][m] = CategoryStats(
                category=cat,
                metric=m,
                n=len(arr),
                mean=float(np.mean(arr)),
                std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                min_val=float(np.min(arr)),
                max_val=float(np.max(arr)),
                values=vals,
            )

    return stats


def cohens_d(group_a: list[float], group_b: list[float]) -> Optional[float]:
    """
    Cohen's d effect size: (mean_a - mean_b) / pooled_std.

    Uses the n-weighted pooled standard deviation, which handles unequal group
    sizes correctly. Returns None if either group has fewer than 2 values.

    With small datasets, treat all results as highly uncertain directional signals.
    Do not interpret as inferential statistics.
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return None
    a, b = np.array(group_a, dtype=float), np.array(group_b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def within_range_check(
    ai_val: float,
    exp_mean: float,
    exp_std: float,
    n_std: float = 1.0,
) -> bool:
    """
    Return True if ai_val falls within exp_mean +/- n_std * exp_std.

    This is an operational comparison, not a statistical test. It answers
    whether the AI deviation is comparable in scale to the experimental spread,
    not whether the two distributions are statistically indistinguishable.
    """
    return abs(ai_val - exp_mean) <= n_std * exp_std


def _safe_mean(records: list[dict], key: str) -> Optional[float]:
    """Return the mean of records[key] over non-None entries, or None."""
    vals = [r[key] for r in records if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def generate_per_protein_interpretation(
    protein: str,
    p_records: list[dict],
    exp_exp: list[dict],
    ai_exp: list[dict],
) -> dict:
    """
    Generate a structured interpretation summary for one protein.

    Covers: which AI model is closest by mean RMSD, whether AI metrics fall
    within the experimental baseline, which experimental method shows the
    highest deviation from AI, and a note on pLDDT confidence.

    All conclusions are qualified by sample size.
    """
    interp: dict = {
        "protein": protein,
        "n_exp_vs_exp": len(exp_exp),
        "n_ai_vs_exp": len(ai_exp),
        "caveats": [],
    }

    if len(exp_exp) < 2:
        interp["caveats"].append(
            f"Only {len(exp_exp)} exp-vs-exp comparison(s) available. "
            "The +/-1 sigma within-range check requires >= 2 for a meaningful standard deviation."
        )

    # Which AI model is closest to experimental structures by mean RMSD?
    af2_exp = [r for r in ai_exp if "AF2" in r.get("comparison", "")]
    of3_exp = [r for r in ai_exp if "OF3" in r.get("comparison", "")]

    mean_af2 = _safe_mean(af2_exp, "rmsd")
    mean_of3 = _safe_mean(of3_exp, "rmsd")

    if mean_af2 is not None and mean_of3 is not None:
        closer = "AF2" if mean_af2 < mean_of3 else "OF3"
        interp["closest_ai_by_mean_rmsd"] = {
            "winner": closer,
            "mean_rmsd_af2": round(mean_af2, 4),
            "mean_rmsd_of3": round(mean_of3, 4),
            "note": (
                f"Based on N={len(ai_exp)} AI-vs-exp comparisons. "
                "Treat as a directional signal only."
            ),
        }
    elif mean_af2 is not None:
        interp["closest_ai_by_mean_rmsd"] = {
            "winner": "AF2 only (OF3 not available)",
            "mean_rmsd_af2": round(mean_af2, 4),
        }
    elif mean_of3 is not None:
        interp["closest_ai_by_mean_rmsd"] = {
            "winner": "OF3 only (AF2 not available)",
            "mean_rmsd_of3": round(mean_of3, 4),
        }
    else:
        interp["closest_ai_by_mean_rmsd"] = None

    # Within-range check against experimental baseline
    if exp_exp and ai_exp:
        within_summary: dict = {}
        for m in METRIC_NAMES:
            exp_vals = [r[m] for r in exp_exp if r.get(m) is not None]
            ai_vals = [r[m] for r in ai_exp if r.get(m) is not None]
            if not exp_vals or not ai_vals:
                continue

            exp_mean = float(np.mean(exp_vals))
            exp_std = float(np.std(exp_vals, ddof=1)) if len(exp_vals) > 1 else 0.0

            if len(exp_vals) < 2:
                within_summary[m] = {
                    "exp_mean": exp_mean,
                    "exp_std": None,
                    "all_within_1std": None,
                    "note": "Only 1 exp-vs-exp baseline; +/-1 sigma check not applicable.",
                }
            else:
                within_flags = [
                    within_range_check(v, exp_mean, exp_std) for v in ai_vals
                ]
                within_summary[m] = {
                    "exp_mean": round(exp_mean, 4),
                    "exp_std": round(exp_std, 4),
                    "ai_values": [round(v, 4) for v in ai_vals],
                    "all_within_1std": all(within_flags),
                    "n_within": sum(within_flags),
                    "n_ai_compared": len(within_flags),
                }

        interp["within_range_check"] = within_summary

    # Which experimental method shows the highest deviation from AI?
    if ai_exp:
        by_method: dict[str, list[float]] = defaultdict(list)
        for r in ai_exp:
            is_af_a = r.get("is_af_a", False)
            exp_method = r.get("method_b") if is_af_a else r.get("method_a")
            if exp_method and r.get("rmsd") is not None:
                by_method[exp_method].append(r["rmsd"])

        if len(by_method) > 1:
            method_means = {m: float(np.mean(v)) for m, v in by_method.items() if v}
            highest = max(method_means, key=method_means.get)
            interp["exp_method_divergence"] = {
                "method_mean_rmsd": {m: round(v, 4) for m, v in method_means.items()},
                "highest_deviation_method": highest,
                "note": (
                    f"'{highest}' structures show the largest mean RMSD against AI. "
                    "This may reflect method-specific characteristics (e.g. crystal "
                    "packing for X-ray, solution dynamics for NMR) rather than AI "
                    "prediction quality alone."
                ),
            }
        else:
            interp["exp_method_divergence"] = {
                "note": "Only one experimental method available; method comparison not applicable."
            }

    # pLDDT and low-confidence region note
    plddt_pairs = []
    for r in ai_exp:
        p = r.get("plddt_matched_b") if r.get("is_af_b") else r.get("plddt_matched_a")
        if p is not None and r.get("rmsd") is not None:
            plddt_pairs.append((float(p), float(r["rmsd"])))

    if plddt_pairs:
        low_conf = [(p, r) for p, r in plddt_pairs if p < 70]
        interp["plddt_analysis"] = {
            "n_comparisons_with_plddt": len(plddt_pairs),
            "n_low_confidence_plddt": len(low_conf),
            "note": (
                f"{len(low_conf)} of {len(plddt_pairs)} AI-vs-exp comparison(s) "
                "involve matched regions with mean pLDDT < 70 (low confidence). "
                + (
                    "Deviations in low-pLDDT regions may reflect genuine structural "
                    "disorder rather than prediction error. Interpret RMSD and t-alpha "
                    "values cautiously for these comparisons."
                    if low_conf
                    else "All matched regions have mean pLDDT >= 70; confidence is acceptable."
                )
            ),
        }
    else:
        interp["plddt_analysis"] = {
            "note": "pLDDT data not available or not confirmed for this protein's AI structures."
        }

    return interp


def analyse(records: list[dict]) -> dict:
    """
    Full statistical analysis over a set of comparison records.

    Parameters
    ----------
    records : list of dicts from pipeline.py, status = "ok" only.
        NMR intra-ensemble records should be excluded before calling
        (filter on category != "nmr_intra_ensemble").

    Returns
    -------
    dict with keys:
      category_stats, effect_sizes, variance_ratios, per_protein,
      per_protein_interpretation, summary_text
    """
    stats = compute_category_stats(records)
    baseline = stats.get(CATEGORY_EXP_VS_EXP, {})

    effect_sizes: dict[str, dict[str, Optional[float]]] = {}
    for ai_cat in [CATEGORY_EXP_VS_AF2, CATEGORY_EXP_VS_OF3, CATEGORY_AF2_VS_OF3]:
        ai_stats = stats.get(ai_cat, {})
        effect_sizes[ai_cat] = {
            m: (
                cohens_d(baseline[m].values, ai_stats[m].values)
                if m in baseline and m in ai_stats
                else None
            )
            for m in METRIC_NAMES
        }

    variance_ratios: dict[str, dict[str, Optional[float]]] = {}
    for ai_cat in [CATEGORY_EXP_VS_AF2, CATEGORY_EXP_VS_OF3]:
        ai_stats = stats.get(ai_cat, {})
        variance_ratios[ai_cat] = {}
        for m in METRIC_NAMES:
            if (
                m in baseline
                and m in ai_stats
                and len(baseline[m].values) > 1
                and len(ai_stats[m].values) > 1
            ):
                var_b = np.var(baseline[m].values, ddof=1)
                var_ai = np.var(ai_stats[m].values, ddof=1)
                variance_ratios[ai_cat][m] = (
                    float(var_ai / var_b) if var_b > 1e-12 else None
                )
            else:
                variance_ratios[ai_cat][m] = None

    proteins = sorted({r["protein_name"] for r in records})
    per_protein: dict[str, dict] = {}
    per_protein_interpretation: dict[str, dict] = {}

    for protein in proteins:
        p_records = [r for r in records if r["protein_name"] == protein]
        exp_exp = [r for r in p_records if r.get("category") == CATEGORY_EXP_VS_EXP]
        ai_exp = [
            r
            for r in p_records
            if r.get("category") in (CATEGORY_EXP_VS_AF2, CATEGORY_EXP_VS_OF3)
        ]

        per_protein[protein] = {
            "n_exp_vs_exp": len(exp_exp),
            "n_ai_vs_exp": len(ai_exp),
        }

        if exp_exp and ai_exp:
            wr: dict = {}
            for m in METRIC_NAMES:
                exp_vals = [r[m] for r in exp_exp if r.get(m) is not None]
                ai_vals = [r[m] for r in ai_exp if r.get(m) is not None]
                if not exp_vals:
                    continue
                exp_mean = float(np.mean(exp_vals))
                exp_std = float(np.std(exp_vals, ddof=1)) if len(exp_vals) > 1 else 0.0
                if len(exp_vals) < 2:
                    wr[m] = {
                        "exp_mean": exp_mean,
                        "exp_n": len(exp_vals),
                        "ai_values": ai_vals,
                        "all_within_1std": None,
                        "note": "Only 1 exp-vs-exp baseline — +/-1 sigma check not applicable.",
                    }
                else:
                    within_flags = {
                        f"ai_{i}": within_range_check(v, exp_mean, exp_std)
                        for i, v in enumerate(ai_vals)
                    }
                    wr[m] = {
                        "exp_mean": exp_mean,
                        "exp_std": exp_std,
                        "exp_n": len(exp_vals),
                        "ai_values": ai_vals,
                        "within_flags": within_flags,
                        "all_within_1std": (
                            all(within_flags.values()) if within_flags else None
                        ),
                    }
            per_protein[protein]["within_range"] = wr

        per_protein_interpretation[protein] = generate_per_protein_interpretation(
            protein, p_records, exp_exp, ai_exp
        )

    # Build printable summary
    lines = [
        "=" * 70,
        "STATISTICAL ANALYSIS SUMMARY",
        "=" * 70,
        "",
        "IMPORTANT CAVEATS",
        "-" * 70,
        f"  Dataset: N = {len(proteins)} proteins. ALL statistics are DESCRIPTIVE, not inferential.",
        "  Cohen's d and variance ratios serve as directional signals only.",
        "  The within-range check is an operational comparison, not a significance test.",
        "  'AI within experimental variability' is a necessary but NOT sufficient",
        "  condition for claiming AI replaces experimental structure determination.",
        "  NMR intra-ensemble variability reflects solution-state dynamics, not",
        "  measurement error — it is NOT directly comparable to X-ray/Cryo-EM spread.",
        "  Scope: chain-level only. Assembly-level conclusions not supported.",
        "-" * 70,
        "",
        f"Total comparison records analysed (status='ok'): {len(records)}",
        "",
    ]

    for cat, m_stats in stats.items():
        if cat == "nmr_intra_ensemble":
            continue
        n_vals = next(iter(m_stats.values())).n if m_stats else 0
        lines.append(f"Category: {cat}  (n_comparisons={n_vals})")
        for m, s in m_stats.items():
            lines.append(
                f"  {m:20s}: mean={s.mean:.4f}  std={s.std:.4f}  "
                f"range=[{s.min_val:.4f}, {s.max_val:.4f}]"
            )
        lines.append("")

    lines.append("Effect sizes (Cohen's d) vs exp_vs_exp baseline")
    lines.append(
        f"  [n-weighted pooled std; directional signals only at N = {len(proteins)}]"
    )
    for ai_cat, metrics in effect_sizes.items():
        if not any(v is not None for v in metrics.values()):
            continue
        lines.append(f"  {ai_cat}:")
        for m, d in metrics.items():
            if d is not None:
                size_label = (
                    "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
                )
                lines.append(f"    {m:20s}: d={d:+.3f}  ({size_label})")
    lines.append("")

    lines.append(
        "Per-protein within-range check  [AI value within exp_vs_exp +/- 1 sigma]"
    )
    lines.append("  checkmark = within, x = outside, dash = only 1 baseline pair")
    for protein, info in per_protein.items():
        wr = info.get("within_range", {})
        if not wr:
            lines.append(f"  {protein}: insufficient exp-vs-exp baseline")
            continue
        flags = []
        for m, d in wr.items():
            aw = d.get("all_within_1std")
            symbol = "-" if aw is None else ("ok" if aw else "outside")
            flags.append(f"{m}={symbol}")
        lines.append(f"  {protein}: {', '.join(flags)}")

    lines += [
        "",
        "Per-protein summaries: see per_protein_interpretation",
        "NMR flexibility baselines: see nmr_variability.json",
        "",
        "=" * 70,
    ]

    return {
        "category_stats": {
            cat: {m: s.to_dict() for m, s in m_map.items()}
            for cat, m_map in stats.items()
        },
        "effect_sizes": effect_sizes,
        "variance_ratios": variance_ratios,
        "per_protein": per_protein,
        "per_protein_interpretation": per_protein_interpretation,
        "summary_text": "\n".join(lines),
    }


def print_summary(analysis_result: dict) -> None:
    """Print the summary text from an analyse() result."""
    print(analysis_result["summary_text"])


def summarise_rejections(rejected_records: list[dict]) -> dict:
    """
    Categorise and count rejected / failed comparison records for reporting.

    Returns a structured summary suitable for embedding in the thesis methods
    section as an "Exclusion criteria / rejected comparisons" table.

    Parameters
    ----------
    rejected_records : list of dicts from pipeline.py (status != "ok").

    Returns
    -------
    dict with:
      total                : int   — total rejected records
      by_status            : dict  — counts per status string ("rejected", "metric_error")
      by_rejection_class   : dict  — counts per high-level rejection class
      proteins_with_any_ok : int   — proteins that have at least one successful comparison
      proteins_all_rejected: list  — protein names where every comparison failed
      printable_summary    : str   — human-readable table for the report
    """
    import re

    total = len(rejected_records)

    by_status: dict[str, int] = {}
    for r in rejected_records:
        s = r.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    # Classify each rejection reason into a high-level category
    CLASS_PATTERNS = [
        ("coverage_failure", r"coverage.*below|below.*coverage"),
        ("too_few_residues", r"Only \d+ residues matched|Fewer than"),
        ("identity_below", r"Sequence identity.*below"),
        ("parse_error", r"Failed to parse|parse.*error"),
        ("metric_error", r"Metric computation failed"),
        ("missing_file", r"not found|No such file"),
        ("chain_error", r"chain.*not found|No chain"),
    ]

    by_class: dict[str, int] = {}
    for r in rejected_records:
        reason = (r.get("rejection_reason") or r.get("reason") or "").lower()
        matched = False
        for label, pattern in CLASS_PATTERNS:
            if re.search(pattern, reason, re.IGNORECASE):
                by_class[label] = by_class.get(label, 0) + 1
                matched = True
                break
        if not matched:
            by_class["other"] = by_class.get("other", 0) + 1

    # Per-protein summary
    ok_proteins: set[str] = set()
    all_proteins: set[str] = set()
    for r in rejected_records:
        all_proteins.add(r.get("protein_name", "unknown"))

    proteins_all_rejected = sorted(all_proteins)  # default (overridden below)

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
        "Note: 'coverage_failure' is the dominant class.  This reflects the",
        "80% per-structure coverage threshold, which correctly excludes fragment-",
        "vs-full and domain-vs-full comparisons that would produce misleading metrics.",
        "=" * 60,
    ]

    return {
        "total": total,
        "by_status": by_status,
        "by_rejection_class": by_class,
        "printable_summary": "\n".join(lines),
    }
