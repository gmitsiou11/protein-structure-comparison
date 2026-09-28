# Thesis: Protein 3D Structure Comparison — Experimental vs ML Predictions

Comparing experimental protein structures (X-ray, NMR, Cryo-EM) against AlphaFold2
and OpenFold3 predictions using three Machaon-based structural metrics (t-alpha,
w-rdist, b-phipsi) plus RMSD as a reference baseline.

Supervisors

- Giannis Emiris
- Panagiotis Kakoulidis

---

## Project Structure

```
data/experimental/    mmCIF files from RCSB PDB
data/alphafold2/      mmCIF predictions from AlphaFold DB (downloaded automatically)
data/openfold3/       mmCIF predictions from OpenFold3 (via NVIDIA NIM endpoint)
src/                  Python modules
notebooks/            Jupyter notebooks (00 → 03)
results/              Output JSON files and plots
proteins.csv          Dataset manifest
```

## Setup

Requires **Python 3.12**.

```bash
pip install -r requirements.txt
```

Open the repo folder in VS Code with the **Jupyter** and **Pylance** extensions.
Run notebooks in order: **00 → 01 → 02 → 03**.

For OpenFold3 predictions via NVIDIA NIM, create a `.env` file at the repo root:

```
NIM_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
```

Get a key at: [https://build.nvidia.com/openfold/openfold3](https://build.nvidia.com/openfold/openfold3)

If `proteins.csv` does not exist, copy `template_proteins.csv` — it has all required columns and no rows.

---

## Source Modules (`src/`)

| Module            | Role                                                                                      |
| ----------------- | ----------------------------------------------------------------------------------------- |
| `parser.py`     | Parse mmCIF files; extract Cα coordinates, φ/ψ dihedral angles and                    |
| `validation.py` | Pre-comparison eligibility checks; computes alignment once and stores results for reuse   |
| `metrics.py`    | RMSD (Kabsch), t-alpha, w-rdist, b-phipsi — all operate on pre-validated alignment       |
| `metadata.py`   | Extract method, resolution, pLDDT from mmCIF; AF2/OF3 confidence range-validated          |
| `analysis.py`   | Descriptive statistics, Cohen's d, variance ratios, per-protein summaries                 |
| `pipeline.py`   | Orchestrator: validation → metadata → metrics → analysis; self-documenting JSON output |
| `downloader.py` | Download mmCIF files from RCSB, AlphaFold DB, and NVIDIA NIM OpenFold3                    |

---

## Notebooks (`notebooks/`)

After editing any `src/` module, restart the kernel before re-running.

| Notebook                         | Purpose                                                                                                                                                               |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `00_data_acquisition.ipynb`    | Download experimental mmCIF files from RCSB, AF2 predictions from AlphaFold DB, and OF3 predictions via NVIDIA NIM. Populates `data/`.                              |
| `01_dataset_exploration.ipynb` | Explore the dataset: verify residue counts, chain IDs, NMR model counts, resolution values, and pLDDT distributions before running the pipeline.                      |
| `02_validated_pipeline.ipynb`  | Run the full comparison pipeline (`run_pipeline`). Produces all JSON output files in `results/`. Re-run after any change to `proteins.csv` or `src/`.         |
| `03_results_analysis.ipynb`    | Load and visualise results: heatmaps, category distributions, NMR variability baselines, correlation matrix, and summary tables.**Canonical results notebook.** |

---

## Pre-Comparison Validation

Before any metric is computed, `src/validation.py` runs a global sequence alignment and rejects pairs that fail the checks below. The same aligned residue index arrays are then reused by all metric functions, guaranteeing every metric operates on the same residues.

| Check                    | Rule                                              | Purpose                                                     |
| ------------------------ | ------------------------------------------------- | ----------------------------------------------------------- |
| Minimum matched residues | `n_matched >= 30`                               | Avoid unstable metrics on very short alignments             |
| Coverage of structure A  | `coverage_a = n_matched / n_residues_a >= 0.80` | Reject if too much of structure A is unmatched              |
| Coverage of structure B  | `coverage_b = n_matched / n_residues_b >= 0.80` | Reject if too much of structure B is unmatched              |
| Sequence identity        | `identical / n_matched >= 0.90`                 | Avoid comparing different isoforms or non-equivalent chains |

Two additional non-fatal warnings are emitted without rejecting the pair: a **size asymmetry warning** when `min(n_a, n_b) / max(n_a, n_b) < 0.70`, and an **internal gap warning** when the alignment contains a consecutive unmatched segment longer than 10 residues. Both appear in the `warnings` field of every output record.

---

## Metrics

The three Machaon metrics (t-alpha, w-rdist, b-phipsi) are the primary contribution of this thesis. RMSD is included as a universal reference baseline.

| Metric              | Description                                                                        | Units         | Structural basis                  |
| ------------------- | ---------------------------------------------------------------------------------- | ------------- | --------------------------------- |
| `rmsd`            | Cα RMSD after Kabsch superposition — reference baseline                          | Å            | Aligned Cα only                  |
| `t_alpha`         | exp(&#124;log(n_tri_A) − log(n_tri_B)&#124;) − 1; alpha-shape surface comparison | dimensionless | All heavy atoms, aligned residues |
| `t_alpha_n_tri_a` | Alpha-shape boundary triangle count for structure A (diagnostic)                   | count         | —                                |
| `t_alpha_n_tri_b` | Alpha-shape boundary triangle count for structure B (diagnostic)                   | count         | —                                |
| `w_rdist_raw`     | 1-Wasserstein distance between flat upper-triangle Cα distance distributions      | Å            | Aligned Cα only                  |
| `w_rdist_norm`    | log10(w_rdist_raw + 1) — the Machaon metric                                       | dimensionless | —                                |
| `b_phipsi`        | Bhattacharyya distance on φ/ψ distributions (circular embedding, moment-based)   | dimensionless | Aligned Cα only                  |
| `b_phipsi_n_a`    | Valid φ/ψ angle pairs used for structure A                                       | count         | —                                |
| `b_phipsi_n_b`    | Valid φ/ψ angle pairs used for structure B                                       | count         | —                                |

All four metrics operate on the **same set of sequence-aligned residues**. For RMSD, w-rdist, and b-phipsi, only Cα atoms at aligned positions are used. For t-alpha, all heavy-atom coordinates are used but restricted to the aligned residue set before the alpha-shape is built.

---

## OpenFold3 / MSA Limitation

All OF3 predictions were submitted to the NVIDIA NIM OpenFold3 endpoint with a **single-sequence MSA** (no homologous sequences), which is functionally equivalent to MSA-free prediction. AF2 predictions from the AlphaFold DB were generated with deep MSAs. Any AF2-vs-OF3 difference is therefore partly confounded by MSA depth and cannot be attributed solely to architectural differences. See `METHODS_NOTES.md` §3.8 and §3.10 item 7.

Provenance details for each OF3 prediction are saved as `OF3_{UniProtID}_provenance.json` alongside the CIF file in `data/openfold3/`.

---

## NMR Ensemble Handling

NMR structures deposit multiple conformers. The pipeline uses the representative model (`nmr_model` column in `proteins.csv`, default 0) for all cross-method comparisons, and separately computes intra-ensemble variability by comparing model 0 against every other deposited model. The intra-NMR variability is a reference baseline — if AI metric values fall within the intra-ensemble range, the prediction is compatible with one accessible conformation of that protein.

## AlphaFold Confidence (pLDDT)

| Source             | pLDDT handling                                                               |
| ------------------ | ---------------------------------------------------------------------------- |
| AF2 (AlphaFold DB) | Extracted from `_atom_site.B_iso_or_equiv`; format confirmed               |
| OF3 (NVIDIA NIM)   | Same extraction + range validation [0, 100]; set to `None` if out of range |

Check `plddt_source_confirmed` before interpreting `plddt_mean` or `plddt_matched`.
pLDDT is not extracted for experimental structures — the same mmCIF field stores
crystallographic B-factors, which are not comparable.

## Statistical Interpretation

**No p-values are reported.** Given only 1–3 exp-vs-exp pairs per protein, classical hypothesis tests are severely underpowered. The analysis reports descriptive statistics and effect sizes only.

| Measure                         | What it is                            | What it is not                      |
| ------------------------------- | ------------------------------------- | ----------------------------------- |
| Descriptive stats (mean ± std) | Summary of observed values            | A test of significance              |
| Cohen's d                       | Effect size between categories        | A claim of statistical significance |
| Variance ratio                  | Spread of AI vs experimental category | A formal F-test                     |
| Within-range check              | AI value within exp ± 1σ            | A statistical test                  |

## Comparison Scope

All comparisons operate at **single polypeptide chain level**. Assembly-level conclusions (oligomeric states, inter-chain contacts) are out of scope. Every output record carries `comparison_scope = "chain_level"`.

---

## Output Files (`results/`)

| File                           | Contents                                                             |
| ------------------------------ | -------------------------------------------------------------------- |
| `all_metrics_validated.json` | All comparison records with full metadata (primary output)           |
| `rejected_pairs.json`        | Failed comparisons with rejection reasons                            |
| `nmr_variability.json`       | Per-protein NMR flexibility baselines (mean ± std over model pairs) |
| `analysis.json`              | Category stats, Cohen's d, variance ratios, per-protein summaries    |
| `pipeline_summary.json`      | Analysis + metadata + pipeline documentation (excluding raw records) |
| `missing_inputs.csv`         | Structure files expected by the manifest but not found on disk       |
