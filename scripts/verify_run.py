"""
Post-run sanity check of a results/ directory.

    python scripts/verify_run.py                      # checks results/
    python scripts/verify_run.py results --proteins N --expect-policy medoid

Exit code 0 when every check passes (warnings do not fail), 1 otherwise.
Run it after notebook 02 (the pipeline).  It reads only the output files, so it
needs no structure data and no heavy imports.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import pandas as pd

METRICS = ["rmsd", "w_rdist_norm", "b_phipsi", "t_alpha"]


class Report:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def ok(self, msg):
        print(f"  [ok]    {msg}")

    def fail(self, msg):
        self.failures += 1
        print(f"  [FAIL]  {msg}")

    def warn(self, msg):
        self.warnings += 1
        print(f"  [warn]  {msg}")

    def check(self, condition, ok_msg, fail_msg):
        (self.ok if condition else self.fail)(ok_msg if condition else fail_msg)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--proteins", type=int, default=None, help="expected number of proteins in the manifest")
    ap.add_argument("--expect-policy", default="medoid", help="NMR model policy the run should have used")
    args = ap.parse_args(argv)
    d, rep = args.results_dir, Report()

    print(f"Checking {d}/")
    records_path = os.path.join(d, "comparisons.json")
    if not os.path.exists(records_path):
        print(f"  [FAIL]  {records_path} not found: run notebook 02 first")
        return 1
    records = _load_json(records_path)
    df = pd.DataFrame(records)

    print("\n1. Records")
    rep.check(len(df) > 0, f"{len(df)} comparison records", "no comparison records")
    if args.proteins:
        n = df["protein_name"].nunique()
        rep.check(n == args.proteins, f"{n} proteins", f"{n} proteins in the records, expected {args.proteins}")
    rejected_path = os.path.join(d, "rejected_pairs.json")
    if os.path.exists(rejected_path):
        rej = _load_json(rejected_path)
        rep.check(len(rej) == 0, "0 rejected pairs", f"{len(rej)} rejected pairs (see rejection_summary.json)")
    else:
        rep.warn("rejected_pairs.json missing")
    if "valid" in df:
        rep.check(df["valid"].all(), "all records valid", f"{int((~df['valid']).sum())} records not valid")

    print("\n2. Metrics are numbers and in range")
    inter = df[~df.get("is_intra_ensemble", False).astype(bool)] if "is_intra_ensemble" in df else df
    for m in METRICS:
        if m not in inter:
            rep.fail(f"column {m} missing")
            continue
        col = pd.to_numeric(inter[m], errors="coerce")
        missing = col.isna().mean()
        bad = (~col.dropna().map(math.isfinite)).sum()
        rep.check(bad == 0, f"{m}: all finite", f"{m}: {bad} non-finite values")
        # b_phipsi is None when a covariance is singular: allowed, but should be rare
        (rep.ok if missing < 0.05 else rep.warn)(f"{m}: {missing:.1%} missing")
        rep.check((col.dropna() >= -1e-9).all(), f"{m}: no negative values", f"{m}: negative values (distances cannot be negative)")
    rep.check(
        (pd.to_numeric(inter["rmsd"], errors="coerce").dropna() < 100).all(),
        "rmsd below 100 Å everywhere",
        "rmsd above 100 Å: a wrong chain or a failed alignment",
    )

    print("\n3. NMR model selection")
    path = os.path.join(d, "nmr_models.csv")
    if not os.path.exists(path):
        rep.fail("nmr_models.csv missing: re-run notebook 02 with the current src/pipeline.py")
    else:
        nm = pd.read_csv(path)
        rep.check(len(nm) > 0, f"{len(nm)} NMR entries recorded", "nmr_models.csv is empty")
        for col in ("model_used", "rule", "n_models", "medoid", "pdb_representative"):
            rep.check(col in nm, f"column {col}", f"column {col} missing")
        if "rule" in nm:
            rules = nm["rule"].value_counts().to_dict()
            print(f"          rules used: {rules}")
            expected = {"medoid": {"medoid", "single model"}, "pdb": {"pdb", "pdb->medoid", "single model"},
                        "first": {"first", "single model"}}[args.expect_policy]
            rep.check(set(rules) <= expected, f"rules consistent with policy {args.expect_policy!r}",
                      f"rules {set(rules) - expected} do not belong to policy {args.expect_policy!r}")
        multi = nm[nm["n_models"] > 1] if "n_models" in nm else nm
        if len(multi) and "model_used" in multi:
            share0 = (multi["model_used"] == 0).mean()
            if args.expect_policy == "medoid":
                # a medoid can be model 0 by chance (about 1/n_models of the entries), not for nearly all of them
                chance = (1 / multi["n_models"]).mean()
                rep.check(share0 < max(0.5, 3 * chance), f"model 0 chosen for {share0:.0%} of multi-model entries (chance level {chance:.0%})",
                          f"model 0 chosen for {share0:.0%} of multi-model entries: the medoid is not being applied")
            else:
                print(f"          model 0 used for {share0:.0%} of multi-model entries")
        if "nmr_model_used" in df:
            nmr_rows = df[df["nmr_model_used"].notna()]
            rep.check(len(nmr_rows) > 0, f"{len(nmr_rows)} NMR comparison records carry nmr_model_used",
                      "no record carries nmr_model_used")

    print("\n4. Provenance")
    summary_path = os.path.join(d, "pipeline_summary.json")
    if os.path.exists(summary_path):
        meta = _load_json(summary_path).get("run_metadata", {})
        pol = meta.get("nmr_model_policy")
        rep.check(pol == args.expect_policy, f"run_metadata.nmr_model_policy = {pol!r}",
                  f"run_metadata.nmr_model_policy = {pol!r}, expected {args.expect_policy!r}")
        commit = meta.get("git_commit")
        if commit:
            print(f"          git commit: {commit}{' (src/ had uncommitted changes)' if meta.get('git_dirty') else ''}")
        else:
            rep.warn("run_metadata has no git commit")
    else:
        rep.warn("pipeline_summary.json missing")

    print(f"\n{rep.failures} failure(s), {rep.warnings} warning(s)")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
