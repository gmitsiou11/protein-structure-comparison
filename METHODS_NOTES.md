# Methods Notes — Thesis Methods Section Reference

Methodological decisions underlying the pipeline, with justification for each choice.
Intended as a reference document for thesis writing — not a user guide.

---

## 3.1 Dataset Construction

Proteins were selected targeting at least two experimental structures per protein from
different methods (X-ray crystallography, solution NMR, or cryo-EM), so that
within-protein experimental variability can serve as a baseline for AI comparison.
Structures were downloaded in mmCIF format from the RCSB Protein Data Bank (rcsb.org).
AlphaFold2 predictions were obtained from the AlphaFold Protein Structure Database
(alphafold.ebi.ac.uk) via its API, which always returns the latest available prediction
for a given UniProt accession. OpenFold3 predictions were generated from the canonical
UniProt sequence for each protein, submitted to the NVIDIA NIM OpenFold3 endpoint
(build.nvidia.com/openfold/openfold3) via `downloader.predict_openfold3()`.

All structures are handled exclusively in mmCIF format.

---

## 3.2 Structural Comparison Scope

All comparisons are performed at **single-chain level** (chain A unless otherwise noted
in `proteins.csv`). This matches the per-UniProt-sequence scope of AlphaFold predictions.
Biological assembly-level comparisons (oligomeric states, inter-chain contacts) are
explicitly out of scope.

---

## 3.3 Pre-Comparison Validation

Before any metric is computed, each pair of structures undergoes eligibility validation:

1. **Cα extraction.** Cα coordinates and one-letter sequences are extracted from the
   specified chain and model index. Structures with fewer than 5 Cα residues are rejected.
2. **Global sequence alignment** (Needleman-Wunsch, match = 1, mismatch = 0,
   gap open = −1, gap extend = −0.5). Global alignment is appropriate because both
   structures are expected to represent the same protein; local alignment would risk
   matching unrelated regions.
3. **Minimum matched residues.** Pairs with fewer than 30 aligned residues are rejected.
   Below this threshold all metrics become unreliable, particularly the Gaussian
   assumption underlying b-phipsi.
4. **Coverage threshold.** Both `coverage_a = matched / n_a` and
   `coverage_b = matched / n_b` must be ≥ 80%. Rejecting on both sides prevents
   fragment-vs-full or domain-vs-full comparisons from producing misleading metrics.
5. **Sequence identity.** At least 90% of matched positions must share the same amino
   acid, avoiding pairings of non-homologous regions or different isoforms.
6. **Unknown residue fraction.** More than 10% "X" residues in either structure triggers
   a warning; the comparison proceeds but metric reliability is flagged as reduced.
7. **Non-fatal warnings.** Two conditions emit warnings without causing rejection:
   a *size asymmetry warning* when `min(n_a, n_b) / max(n_a, n_b) < 0.70` (indicating
   one structure covers a substantially longer region, e.g. a signal peptide present
   in the AF prediction but absent from the experimental chain), and an *internal gap
   warning* when the alignment contains a consecutive unmatched segment longer than 10
   residues (one full helix turn), flagging possible domain insertions that can distort
   the w-rdist distance distribution. Both are recorded in the `warnings` field of
   every output record.

The alignment is computed once and stored; all metric functions reuse the same index
arrays, guaranteeing every metric operates on the same residue pairs.

---

## 3.4 Structural Metrics

### Shared structural basis

All four metrics operate on the **same set of sequence-aligned residues**. RMSD, w-rdist,
and b-phipsi use Cα coordinates at aligned positions only. t-alpha uses all heavy-atom
coordinates restricted to the aligned residue set before the alpha-shape is built. This
alignment-consistent design prevents structural scope mismatches — for example, unmatched
signal peptides in AF predictions are excluded from all metrics, not just RMSD.

### 3.4.1 RMSD (Reference Metric)

Root-mean-square deviation of Cα positions after least-squares superposition (Kabsch
algorithm). RMSD is the universal reference metric in structural biology and provides
an absolute deviation scale in Ångströms. The Kabsch algorithm finds the optimal
rotation R via SVD decomposition with reflection correction (det(R) = +1 enforced).

RMSD is included as a reference baseline for interpretability. The primary metrics of
this thesis are the three Machaon metrics below.

### 3.4.2 t-Alpha (Machaon)

t-alpha compares the **alpha-shape surface geometries** of two structures (Machaon
reference: `pdbhandler.get_mesh_triangles`, `scanner.py` metric_index = 2).

For each structure:

1. Filter all heavy-atom coordinates to the **sequence-aligned residues only**.
2. Normalise to [0, 1] per axis with per-column MinMax scaling (matching Machaon's
   `pdbhandler.load_points` with `MinMaxScaler`, `normalized=True`).
3. Build a 3D alpha-shape via Open3D's
   `TriangleMesh.create_from_point_cloud_alpha_shape` with alpha = 0.085 (the
   hardcoded Machaon parameter).
4. Count boundary triangles: `n`, store `log(n)`.

Comparison:

```
t_alpha = exp(|log(n_A) - log(n_B)|) - 1  =  max(n_A/n_B, n_B/n_A) - 1
```

The metric is 0 when both structures have the same surface complexity and grows without
bound as the ratio diverges. It is dimensionless and reported as a single field
`t_alpha`

**Naming note.** The supervisor's informal name "α-carbon torsion surface metric" can
be misleading: "torsion" here does not refer to backbone dihedrals (that is b-phipsi),
and "α-carbon" does not mean Cα-only coordinates. The name derives from the alpha-shape
algorithm (Edelsbrunner & Mücke, 1994) parameterised by `alpha`.

**Deviation from original Machaon.** The original operates on all heavy-atom coordinates
of the full chain without any alignment filter. This pipeline additionally restricts the
all-atom point cloud to sequence-aligned residues only (via `ca_index_filter` in
`parser.get_all_atom_coords`), making t-alpha's structural scope consistent with the
other three metrics. The alpha-shape library (Open3D) and the alpha parameter (0.085)
are unchanged from the original.

### 3.4.3 w-rdist (Machaon)

w-rdist computes the 1-Wasserstein (Earth Mover's) distance between the global pairwise
Cα distance distributions of two structures (Machaon reference: `scanner.py`
metric_index = 1, `calculate_residue_distances`).

For N aligned Cα positions, the upper triangle of the N×N pairwise distance matrix
(C(N,2) values) is extracted and treated as a 1D distribution:

```
w_rdist_raw  = W1(dist_A, dist_B)     (Å)
w_rdist_norm = log10(W1 + 1)          (dimensionless)
```

The log10 compression reduces size-dependency (larger proteins have larger absolute W1
values). The +1 offset ensures the result is 0 for identical distributions.

**Deviation from original Machaon.** The original computes distances over all chain
residues. This pipeline uses only sequence-aligned Cα positions, which is the more
principled choice — non-homologous residues would contaminate the distance distribution.
This means w-rdist values here are not numerically comparable to Machaon paper values
for the same pair. `scipy.stats.wasserstein_distance` is used in place of Machaon's
manual CDF implementation; the algorithm is identical.

### 3.4.4 b-phipsi (Machaon)

Bhattacharyya distance between the backbone dihedral angle distributions of two
structures (Machaon reference: `bhattacharyyadistance.py`, `multivariate_compare`),
restricted to sequence-aligned residues.

**Circular embedding.** Dihedral angles are periodic — −180° and +180° are the same
point. Each angle θ is embedded as (cos θ, sin θ), giving a 4D vector per residue:
[cos φ, sin φ, cos ψ, sin ψ]. This makes Euclidean distance geometrically correct for
angular data.

**Bhattacharyya formula** (applied in 4D embedding space):

```
D_B = (1/8)(μ_A − μ_B)ᵀ Σ_avg⁻¹ (μ_A − μ_B)  +  (1/2) ln(det(Σ_avg) / √(det(Σ_A)·det(Σ_B)))
where  Σ_avg = (Σ_A + Σ_B) / 2
```

**Covariance regularisation.** A small diagonal term ε·I (ε = 1×10⁻⁶) is added before
inversion to prevent numerical failure when backbone angles cluster tightly (e.g. a
rigid helix produces a near-singular 4×4 covariance). The original Machaon guards
against this only via a `det_cov ≤ 0` rejection check; regularisation is more robust
because it retains the comparison instead of silently discarding it.

**Gaussian assumption — scope.** b-phipsi is a moment-difference metric: it captures
differences in the mean and covariance of the circular embedding, not the full shape of
the Ramachandran distribution. Two proteins with identical secondary structure composition
but different loop geometries can score near zero; two proteins where one is more helical
score high even if their secondary structures are locally similar. b-phipsi is best
understood as a distributional-level backbone conformation signal and should always be
interpreted alongside the spatial metrics.

**Residue filtering.** Terminal residues and residues adjacent to prolines lack valid
φ/ψ angles and are excluded before embedding. A minimum of 6 valid angle pairs per
structure is required; below this the 4×4 covariance is singular and the metric returns
None. Because b-phipsi compares distributions rather than paired observations, the valid
residue counts may differ between the two structures (`b_phipsi_n_a`, `b_phipsi_n_b`).

**Deviation from original Machaon.** The original operates on raw degree values,
which introduces discontinuity errors near ±180°. This pipeline applies circular
embedding before computing the Bhattacharyya distance.

---



## 3.5 AlphaFold Confidence (pLDDT)

pLDDT scores AF model confidence per residue on a 0–100 scale (< 50: very low,
50–70: low, 70–90: confident, > 90: very high).

For **AF2 (AlphaFold DB)**: pLDDT is stored in `_atom_site.B_iso_or_equiv` — a
documented convention confirmed in the AlphaFold DB specification.

For **OF3 (NVIDIA NIM)**: same extraction plus range validation. Values outside
[0, 100] cause confidence to be set to None with an explanatory `plddt_note`.
Check `plddt_source_confirmed` before interpreting `plddt_mean` or `plddt_matched`.

pLDDT is not applicable to experimental structures: the same mmCIF field stores
crystallographic B-factors, which measure thermal motion, not model confidence.

---

## 3.6 NMR Ensemble Handling

NMR structures deposit multiple conformers representing solution-state dynamics.

**Representative model.** The `nmr_model` column in `proteins.csv` specifies which
conformer is used for all NMR-vs-experimental and NMR-vs-AI comparisons. The default
is model 0: wwPDB deposition guidelines require depositing authors to place their
highest-quality model first.

**Intra-ensemble variability.** Model 0 is compared against all other deposited models
(k = 1, 2, …, N−1). Results are aggregated as mean ± std (ddof = 1) across all
model-0-vs-model-k pairs and reported in `nmr_variability.json`.

**Interpretation.** NMR intra-ensemble variability reflects physically real solution-state
conformational flexibility, not measurement error. It is fundamentally different from
X-ray/cryo-EM positional uncertainty or AI prediction error. It is used as a reference
baseline: if AI metric values fall within the intra-ensemble range, the prediction is
compatible with one accessible conformation of that protein.

---

## 3.7 Statistical Analysis

Given only 1–3 exp-vs-exp comparison pairs per protein. The analysis relies on
descriptive statistics and effect sizes, which remain interpretable at any sample size.

**Reported measures:**

- Mean ± std per comparison category per metric
- Cohen's d between AI and experimental categories (n-weighted pooled std; directional
  signal only, not an inferential claim)
- Variance ratio: Var(AI) / Var(exp_vs_exp)
- Per-protein within-range check: whether AI metric falls within exp_vs_exp mean ± 1σ

**Within-range check interpretation.** "AI within experimental variability" means the AI
prediction deviates from one experimental source by no more than two experimental methods
deviate from each other. This is a useful reference point but not sufficient to claim AI
replaces experiments — experimental methods additionally provide ground-truth electron
density, ensemble information, and structural evidence for ligands and post-translational
modifications.

---

## 3.8 OpenFold3 vs AlphaFold3 

This pipeline uses **OpenFold3** (aqlaboratory/openfold-3) via the NVIDIA NIM hosted
endpoint (build.nvidia.com/openfold/openfold3). OpenFold3 is an open-source
reimplementation of the AlphaFold3 architecture, distinct from the proprietary
AlphaFold3 model (alphafoldserver.com). The two share the same architecture (Abramson
et al., Nature 2024) but differ in training data, weights, and infrastructure.
All predictions labelled `OF3_` in this pipeline were generated exclusively via NVIDIA NIM.

**MSA handling caveat.** All OF3 predictions were submitted with a single-sequence MSA
(functionally equivalent to MSA-free prediction). AF2 predictions from the AlphaFold DB
used deep evolutionary MSAs. AF2-vs-OF3 metric differences in this thesis reflect
"with-MSA vs. without-MSA" effects at least as much as architectural differences, and
should not be attributed to model architecture alone. 

Provenance details for each OF3 prediction (endpoint URL, sequence hash, diffusion
samples, confidence scores, NIM model version headers) are saved as
`OF3_{UniProtID}_provenance.json` in `data/openfold3/`.

---

## 3.9 Adapted Implementation — Relationship to Machaon

This pipeline is an adapted port of the Machaon metrics, not a wrapper around the
Machaon codebase. The metric mathematics are faithful to the Machaon paper and reference
source files. Adaptations:

- All file I/O uses gemmi (mmCIF) rather than BioPython (PDB format).
- Global sequence alignment (BioPython PairwiseAligner) is applied before all
  Cα-based metrics; the original Machaon assumes identical residue counts.
- All three Machaon metrics are restricted to sequence-aligned residues; the original
  operates on full chain residues (see §3.4.2, §3.4.3 for per-metric rationale).
- b-phipsi adds circular embedding and Tikhonov regularisation not present in the
  original (see §3.4.4).
- Pipeline orchestration, validation logic, output schema, and NMR ensemble handling
  are original contributions.

---

## 3.10 Known Limitations

1. **Gaussian approximation in b-phipsi.** b-phipsi captures differences in the first
   two moments of the circular angle embedding, not the full Ramachandran distribution
   shape. Two concrete failure modes: *false near-zero* — a bimodal distribution
   (e.g. 50% helix, 50% strand) can have the same circular mean and covariance as a
   broad unimodal distribution; *false large* — structurally similar proteins score
   high if one has more disordered loops, different proline/glycine content, or if the
   NMR representative model is an outlier conformer. b-phipsi should always be
   interpreted alongside the spatial metrics, not in isolation.
2. **Alpha-shape sensitivity.** The alpha parameter (0.085) was calibrated by Machaon
   for normalised all-atom PDB coordinate clouds. mmCIF structures may include slightly
   different atom sets (e.g. alternate conformations). The per-column MinMax
   normalisation applied before meshing keeps the parameter scale-invariant, but the
   triangle count is sensitive to the density and geometry of the point cloud.
3. **Static prediction vs conformational ensemble.** AF2 and OF3 each produce one
   structure. Comparing against NMR model 0 is asymmetric — the NMR model is one
   sample from a physical ensemble, the AI prediction is not. AI predictions are most
   meaningfully compared against X-ray structures (single static conformation).
4. **Crystal packing artefacts.** X-ray structures may show artificially constrained
   flexible regions due to crystal contacts. High X-ray-vs-NMR deviation in flexible
   domains may reflect packing rather than true conformational difference.
5. **Cα-only scope for RMSD, w-rdist, and b-phipsi.** These three metrics are blind to
   side-chain rotamer differences, ligands and cofactors, alternate conformer occupancies,
   and fine atomic detail in intrinsically disordered regions. t-alpha partially
   addresses this by using all heavy atoms for the surface, but at the gross topology
   level only.
