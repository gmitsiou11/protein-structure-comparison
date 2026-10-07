"""
Information Imbalance: how much the metrics share, and what goes with a far prediction.

Information Imbalance (Glielmo et al., PNAS Nexus 2022), with k = 1:

    Delta(A -> B) = (2 / N) * mean over points i of r_B(i, nn_A(i))

nn_A(i) is the nearest neighbour of point i in space A, and r_B(i, j) the rank
of j among the neighbours of i in space B (1 = nearest).  Delta ~ 0: the
neighbours in A are also neighbours in B (A carries B's information);
Delta ~ 1: A says nothing about B.  It is not symmetric: Delta(A -> B) small
and Delta(B -> A) large means A contains B's information and more.

Here a point is one comparison (a pair of structures) and the coordinates are
its metric values (w-rdist, b-phipsi, t-alpha, RMSD), log-transformed and
standardised.  The Differentiable Information Imbalance (DII, Wild et al.,
Nat Commun 2025) is run with the DADApy package in notebook 05; this module
only needs numpy, pandas and scipy.

Reading.  When the reference space is built from the same metrics, Delta
measures redundancy among them, not which metric is closer to a physical
truth: a metric that shares nothing with the others (for instance because it
is dominated by noise) cannot be predicted from them and therefore looks
"informative".  Questions with an independent reference (the metrics ->
RMSD; protein properties -> how far a prediction lands) do not have this
problem.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

METRICS = ["w_rdist_norm", "b_phipsi", "t_alpha", "rmsd"]
LOG_EPS = 1e-4  # added before log10: t-alpha and b-phipsi can be exactly 0


def load_metric_matrix(
    records_or_path, metrics: list[str] = METRICS, include_nmr_ensemble: bool = False
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Metric values of the computed comparisons that have every metric.

    Returns (X, info): X has shape (n_records, len(metrics)), raw values;
    info has protein_name, category and comparison for each row.
    """
    if isinstance(records_or_path, str):
        with open(records_or_path, encoding="utf-8") as f:
            records = json.load(f)
    else:
        records = records_or_path
    rows = [
        r
        for r in records
        if r.get("status", "ok") == "ok"
        and (include_nmr_ensemble or r.get("category") != "nmr_intra_ensemble")
        and all(r.get(m) is not None for m in metrics)
    ]
    X = np.array([[float(r[m]) for m in metrics] for r in rows], dtype=float)
    info = pd.DataFrame(
        {
            "protein_name": [r.get("protein_name") for r in rows],
            "category": [r.get("category") for r in rows],
            "comparison": [r.get("comparison") for r in rows],
        }
    )
    return X, info


def log_standardise(X: np.ndarray, eps: float = LOG_EPS) -> np.ndarray:
    """log10(x + eps), then zero mean and unit variance per column.

    The metrics span several orders of magnitude (b-phipsi ~1e-5..1, RMSD
    ~0.1..40 Å); without the log a few large values decide every neighbour.
    Standardising puts the columns on one scale, so weights are comparable.
    """
    L = np.log10(np.asarray(X, dtype=float) + eps)
    sd = L.std(axis=0)
    sd[sd == 0] = 1.0
    return (L - L.mean(axis=0)) / sd


def neighbour_ranks(Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    ranks[i, j] = rank of point j among the neighbours of point i (1 = nearest),
    Euclidean distance, ties broken at random, the point itself excluded
    (it gets the largest rank).
    """
    D = cdist(Z, Z)
    D = D + rng.uniform(0.0, 1e-9, D.shape)  # random tie-breaking (exact zeros occur)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(len(Z))[:, None]
    ranks[rows, order] = np.arange(1, len(Z) + 1)
    return ranks


def information_imbalance(
    ZA: np.ndarray, ZB: np.ndarray, rng: Optional[np.random.Generator] = None
) -> float:
    """Delta(A -> B) with k = 1 (see module docstring)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    ZA = np.asarray(ZA, dtype=float).reshape(len(ZA), -1)
    ZB = np.asarray(ZB, dtype=float).reshape(len(ZB), -1)
    if len(ZA) != len(ZB):
        raise ValueError("A and B must describe the same points")
    n = len(ZA)
    ranks_A = neighbour_ranks(ZA, rng)
    ranks_B = neighbour_ranks(ZB, rng)
    nearest_in_A = np.argmin(ranks_A, axis=1)
    return float(2.0 / n * ranks_B[np.arange(n), nearest_in_A].mean())


def imbalance_table(
    Z: np.ndarray, names: list[str], rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """
    Delta(row -> column) for every pair of single metrics, plus each metric
    against the full space ('all') and the full space against each metric.
    The diagonal is left empty (a space against itself is trivially ~0).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    labels = list(names) + ["all"]
    spaces = {name: Z[:, [i]] for i, name in enumerate(names)}
    spaces["all"] = Z
    table = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a in labels:
        for b in labels:
            if a != b:
                table.loc[a, b] = information_imbalance(spaces[a], spaces[b], rng)
    return table


def subsample(n_points: int, size: int, seed: int) -> np.ndarray:
    """Indices of a random subsample (all points if size >= n_points)."""
    rng = np.random.default_rng(seed)
    if size >= n_points:
        return np.arange(n_points)
    return np.sort(rng.choice(n_points, size=size, replace=False))


def one_per_protein(info: pd.DataFrame, seed: int) -> np.ndarray:
    """
    Indices of a random sample with one comparison per protein.

    The comparisons of a protein share structures (the same experimental
    structure and prediction appear in several pairs), so their metric values
    are not independent and, in a nearest-neighbour method, a point's closest
    neighbour is often another pair of the same protein.  One comparison per
    protein removes that dependence, at the price of ~150 points instead of
    ~1,200 (use a small k).
    """
    rng = np.random.default_rng(seed)
    chosen = [
        int(rng.choice(group.index.to_numpy()))
        for _, group in info.groupby("protein_name", sort=True)
    ]
    return np.sort(np.array(chosen, dtype=int))


def protein_covariates(
    protein_table: pd.DataFrame,
    experimental_structures: Optional[pd.DataFrame] = None,
    af2_cutoff: str = "2018-04-30",
    length_gaps: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Per-protein covariates for the DII question "which properties of a protein
    go with a prediction far from the experiments?":

      plddt_af2, plddt_of3     model confidence (mean pLDDT)
      length                   residues of the prediction
      cryoem_chains            polymer chains in the cryo-EM entry (complex size)
      has_xray                 1 if an X-ray structure is in the set
      exp_<metric>             experimental variability for each metric
      frac_before_af2_cutoff   share of its experimental entries released before
                               the AF2 training cutoff (needs experimental_structures)
      max_length_gap           largest length difference among its pairs (needs
                               length_gaps, e.g. baselines.protein_max_gap)
    """
    out = pd.DataFrame({"protein_name": protein_table["protein_name"]})
    for column in ("plddt_af2", "plddt_of3", "length", "cryoem_chains"):
        out[column] = pd.to_numeric(protein_table[column], errors="coerce")
    out["has_xray"] = protein_table["has_xray"].astype(str).str.lower().eq("true").astype(float)
    for metric in METRICS:
        out[f"exp_{metric}"] = pd.to_numeric(protein_table[f"{metric}__exp"], errors="coerce")
    if experimental_structures is not None:
        es = experimental_structures.copy()
        es["release_date"] = es["release_date"].fillna("").astype(str)
        released = es["release_date"].ne("")
        es["before"] = released & (es["release_date"] <= af2_cutoff)
        frac = es[released].groupby("protein_name")["before"].mean()
        out["frac_before_af2_cutoff"] = out["protein_name"].map(frac)
    if length_gaps is not None:
        out["max_length_gap"] = out["protein_name"].map(length_gaps)
    return out
