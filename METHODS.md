# Methods

This document describes the data, measurements and analyses of the study. Implementation details are
documented in the modules of `src/` and in the notebooks. Quantities that depend on a particular run
(dataset size, results) are reported by the notebooks rather than here.

---

## 1. Study design

The study asks whether a predicted protein structure can stand in for an experimental one. For each
protein, the structures determined by different experimental methods (X-ray crystallography, solution
NMR, cryo-EM) differ from one another; this spread defines the protein's **experimental variability**.
A prediction (AlphaFold2, OpenFold3) is considered *within experimental variability* when it is, by a
given measure, no farther from the experimental structures than these are from each other.

Distances between structures are measured with the three metrics of Machaon [1] (b-phipsi, w-rdist,
t-alpha), which compare per-chain descriptors and require no residue correspondence. Cα-RMSD is
reported alongside as a residue-level reference. Before the predictions are assessed, the metrics are
calibrated: the implementation is validated, the factors other than conformation that change their
values are measured, and reference levels give their scale (Section 5).

The protein is the unit of analysis. The comparisons of one protein share structures and are therefore
not independent; each protein contributes one value per quantity, and results are reported as
fractions and medians over proteins with bootstrap confidence intervals. The analysis is descriptive
and performs no hypothesis tests.

## 2. Dataset

### 2.1 Sources

The dataset is generated from public resources (`src/dataset.py`, notebook 00):

| Resource | Content used |
|---|---|
| SIFTS [2] | mapping of every PDB chain to a UniProt accession, with the observed residue ranges |
| wwPDB [3] | experimental method and resolution of every PDB entry |
| UniProt [4] | sequence, length and annotated mature chain of each candidate protein |
| AlphaFold Database [5] | availability of an AlphaFold2 model for each accession |

The SIFTS release used is printed by notebook 00 and identifies the state of the dataset.

### 2.2 Selection criteria

Criteria are applied in the order below, and the number of proteins remaining after each is recorded
(`results/dataset_selection_counts.csv`).

| # | Criterion | Rationale |
|---|---|---|
| 0 | Resolution ≤ 2.5 Å (X-ray), ≤ 4.0 Å (cryo-EM); all NMR entries | poorly determined structures are not used as references |
| 1 | Structures by both NMR and cryo-EM | every protein needs at least two experimental methods; NMR is the scarcest method, so the set is built from it. X-ray structures are included where available |
| 2 | A single mature chain in UniProt | for polyproteins, an experimental structure covers one cleavage product while a prediction covers the whole precursor |
| 3 | Mature chain of 50–300 residues | enough residues for stable per-chain descriptors; the size range accessible to solution NMR; prediction memory and time scale as N² and N³ |
| 4 | An NMR chain and a cryo-EM chain each covering ≥ 80 % of the mature chain | the metrics summarise whole chains, so unmodelled residues would register as structural difference |
| 5 | An AlphaFold Database model exists | required for the comparison |

### 2.3 Chain selection

One chain represents each experimental method: the chain with the best resolution for X-ray and
cryo-EM, and the chain with the highest coverage of the mature chain for NMR (ties broken
alphabetically by entry and chain). A protein therefore has either three experimental structures
(X-ray, NMR, cryo-EM; three experimental pairs) or two (NMR, cryo-EM; one pair). All comparisons are
between single chains; multi-chain assemblies are outside the scope of the study.

## 3. Structures

### 3.1 Experimental structures

Experimental structures are retrieved from the RCSB PDB as mmCIF files and read with gemmi [6]
(`src/parser.py`). Only polymer residues are read; ions, waters and ligands are ignored. Alternate
conformations are resolved per atom by the highest occupancy (the first location on ties) and per
residue position by the last residue type, which reproduces the behaviour of Biopython [7] and hence of
Machaon.

**NMR representative model.** An NMR entry contains an ensemble of models. The entry is represented by
its **medoid**, the model with the smallest mean Cα-RMSD to all other models after superposition
(`src/nmr.py`; ensembles larger than 100 models are subsampled evenly). Two alternative rules are
available: the model designated as representative by the depositors
(`_pdbx_nmr_representative.conformer_id`), or the first model. The model used for each protein and the
rule are recorded in `results/nmr_models.csv`.

### 3.2 Predicted structures

**AlphaFold2.** Models are downloaded from the AlphaFold Database [5] by UniProt accession (latest
model version). They are produced by the full AlphaFold2 pipeline [8], including deep multiple sequence
alignments and structural templates.

**OpenFold3.** Models are predicted with the OpenFold3 [9] NIM on NVIDIA's hosted API, from the
UniProt canonical sequence (identical to the sequence of the AlphaFold2 model). For every protein a
multiple sequence alignment (MSA) is first obtained from the NVIDIA MSA Search NIM, which runs the
ColabFold [10] MMseqs2 [11] search against UniRef30 and the ColabFold environmental database (E-value
10⁻⁴, at most 500 sequences per database). OpenFold3 is run with this MSA, without templates, and
returns five diffusion samples; the sample with the highest confidence score is kept.

Both predictions cover the full UniProt sequence. For every model the download or request parameters,
MSA depth, sample scores and service response headers are stored in a provenance file next to it.

### 3.3 Model confidence

The predicted local distance difference test score (pLDDT) is read from the B-factor field of the Cα
atoms of the predicted models and averaged over the chain; for OpenFold3 the values are checked to lie
in the pLDDT range. It is not read for experimental structures, where the field holds atomic
displacement parameters.

## 4. Pairwise comparison

### 4.1 Comparison categories

For every protein, every pair of its structures is compared (`src/pipeline.py`, notebook 02):

| Category | Pairs | Role |
|---|---|---|
| experimental – experimental | X-ray–NMR, X-ray–cryo-EM, NMR–cryo-EM | experimental variability |
| experimental – prediction | each experimental structure with AlphaFold2 and with OpenFold3 | prediction against experiment |
| prediction – prediction | AlphaFold2 – OpenFold3 | agreement between the predictions |
| NMR ensemble | representative model with every other model of its ensemble | reference level (Section 5) |

Each comparison yields one record (`results/comparisons.json`) holding the four measures, the alignment
statistics, and the method, dates, chain count and pLDDT of both structures.

### 4.2 Sequence alignment

The two chains of a pair are aligned once with a global alignment (Biopython `PairwiseAligner`; match 1,
mismatch 0, gap opening −1, gap extension −0.5; `src/alignment.py`). The alignment describes the pair
(number of matched residues, coverage of each chain, sequence identity) and supplies the residue pairs
for RMSD. It is not used by the Machaon metrics. A pair is excluded only when a structure cannot be read
or a chain has fewer than five Cα atoms; low coverage or identity are recorded as warnings.

### 4.3 Machaon metrics

Each structure is reduced to a descriptor computed from its own chain, and two structures are compared
only through their descriptors, following the definitions of Machaon [1] (`src/metrics.py`). The two
chains may therefore differ in length and sequence.

**b-phipsi.** The backbone dihedral angles (φ, ψ), in degrees, of all residues for which both are
defined are summarised by their mean μ and 2 × 2 covariance Σ (at least six residues are required).
Two structures are compared by the Bhattacharyya distance [12] between the two Gaussian distributions:

$$
D_B = \tfrac{1}{8}\,(\mu_A-\mu_B)^{\top}\,\Sigma^{-1}(\mu_A-\mu_B) \;+\; \tfrac{1}{2}\ln\frac{\det\Sigma}{\sqrt{\det\Sigma_A\,\det\Sigma_B}},
\qquad \Sigma = \tfrac{1}{2}(\Sigma_A+\Sigma_B).
$$

The distance is undefined when a covariance matrix is singular.

**w-rdist.** The descriptor of a chain is the set of all pairwise Cα–Cα distances. Two chains are
compared by the 1-Wasserstein distance W₁ between the two empirical distributions, each normalised by
its own size, and the metric is reported as

$$
\text{w-rdist} = \log_{10}(W_1 + 1).
$$

**t-alpha.** The heavy-atom coordinates of a chain are scaled to [0, 1] along each axis (min–max
scaling), and the alpha shape [13] of the scaled points is computed with Open3D [14] (α = 0.085). The
descriptor is the number of triangles *n* of the resulting surface mesh, and two chains are compared by

$$
\text{t-alpha} = \exp\!\big(\lvert \ln n_A - \ln n_B \rvert\big) - 1 = \frac{\max(n_A, n_B)}{\min(n_A, n_B)} - 1 .
$$

A value of zero indicates identical descriptors, not identical structures; in particular, permuting the
residues of a chain leaves all three metrics unchanged.

### 4.4 Cα-RMSD

RMSD is the only correspondence-based measure. The Cα atoms of the aligned residue pairs are superposed
by the Kabsch algorithm [15] and the root-mean-square deviation is computed. It is reported only when
the paired residues are at least 50 % identical; below this threshold the two chains are unlikely to
represent the same protein, whereas point mutants and engineered variants are paired correctly.

### 4.5 Implementation and validation

The metrics are implemented independently of the Machaon code base: structures are read with gemmi and
distances computed with SciPy. To validate the implementation, `src/machaon_reference.py` repeats
Machaon's own feature extraction (Biopython parsing, min–max scaling, Open3D alpha shapes) and formulas;
the two implementations agree to rounding precision on X-ray, NMR and cryo-EM chains, including chains
with alternate conformations (notebook 03).

Machaon compares two structures without alignment: its metrics are defined to be independent of the
length and the orientation of a structure, so that neither superposition nor residue correspondence is
needed, and structures are analysed in their raw form, missing residues included [1]. Sequence alignment
is used only in its segment-search mode, to locate the regions to be compared, and in the post-hoc
evaluation of ranked hits; whole-chain comparisons, the setting here, use none. (In practice the per-axis
min–max scaling makes t-alpha depend on orientation; Section 5.) The implementation differs from Machaon in three respects:

* **Heavy atoms for t-alpha.** Machaon uses every atom in the file. NMR structures include hydrogen
  atoms whereas most X-ray and cryo-EM structures do not, so hydrogens are excluded to keep the
  descriptor comparable across methods; the effect is quantified in Section 5.
* **Polymer residues only.** Non-polymer residues are not read, so that, for example, a calcium ion
  (atom name CA) cannot enter the Cα distances.
* **Minimum number of angles.** b-phipsi requires at least six residues with both dihedral angles,
  which is always met for chains of 50–300 residues.

## 5. Metric calibration

Notebook 03 validates the metrics and establishes how their values are read: which factors other than
conformation change them, and how large they are at three reference levels (`src/baselines.py`,
`src/sensitivity.py`).

* **Validation.** Agreement with Machaon's reference procedure (Section 4.5).
* **Residue order.** A chain compared with a copy whose residues have been shuffled: the Machaon metrics
  remain zero, whereas RMSD does not.
* **Coverage.** Because the descriptors summarise whole chains, residues present in one chain and absent
  from the other contribute to the distance. This is measured by truncating chains (0–30 % of residues
  removed from the termini) and by the Spearman correlation of each metric with the length gap of a
  pair, 1 − (shorter length / longer length). Coverage-matched subsets (length gap ≤ 10 % and ≤ 5 %)
  repeat the experimental reference levels.
* **Orientation.** Min–max scaling along the coordinate axes makes t-alpha depend on the orientation of
  the structure. Each chain is compared with 20 random rotations of itself to obtain the orientation
  noise floor. A variant that first rotates every structure onto its principal axes removes this
  dependence; it is computed for every comparison as a sensitivity analysis, while the primary analysis
  uses Machaon's definition.
* **Hydrogens.** t-alpha computed from heavy atoms and from all atoms on the same pairs.
* **Reference levels.** Three levels of agreement, from the tightest to the loosest: two copies of the
  protein in the same PDB entry (identified through SIFTS; copies within 0.01 Å RMSD, identical by
  imposed symmetry, are excluded), the models of one NMR ensemble, and structures determined by
  different methods. Each protein contributes one value per level (the median over its pairs).

## 6. Statistical analysis

### 6.1 Protein-level summaries

For protein $p$ and measure $d$, with experimental structures $E_1, \dots, E_m$ and prediction $P$
(notebook 04, `src/analysis.py`):

$$
v_{\text{exp}}(p) = \operatorname*{median}_{i<j}\, d(E_i, E_j), \qquad
v_{\text{pred}}(p) = \operatorname*{median}_{i}\, d(P, E_i).
$$

The prediction is within experimental variability when $v_{\text{pred}}(p) \le v_{\text{exp}}(p)$. For each
measure and prediction the following are reported over proteins:

* the fraction of proteins within experimental variability;
* the median of the difference $\delta(p) = v_{\text{pred}}(p) - v_{\text{exp}}(p)$;
* the paired effect size $d_z = \bar{\delta} / s_{\delta}$, the mean difference divided by its standard
  deviation [16];
* for proteins with a multi-model NMR entry, the fraction for which the prediction is no farther from the
  representative NMR model than the farthest model of the same ensemble;
* for AlphaFold2 against OpenFold3, the fraction of proteins for which AlphaFold2 is closer to the
  experimental structures, and the median of the difference.

### 6.2 Confidence intervals

All estimates are given with 95 % percentile bootstrap intervals [17] obtained by resampling proteins
with replacement (10,000 resamples for the main estimates; fixed random seed).

### 6.3 Flagged experimental structures

Experimental structures that are not a plain copy of the UniProt protein are flagged: sequence identity
below 90 % to the UniProt sequence (engineered variants), ten or more residues not matching it (tags,
fusions), or entry titles naming amyloid, fibrils, filaments or cages. Results are reported for all
proteins and for proteins without flagged structures.

### 6.4 Robustness and sensitivity

The fraction within experimental variability is recomputed under each of the following conditions:

| Factor | Conditions |
|---|---|
| Structure quality | all proteins; proteins without flagged structures |
| Estimate of the variability | three experimental pairs; a single NMR–cryo-EM pair |
| Coverage | length gap ≤ 10 % and ≤ 5 % in every pair |
| Assembly context | cryo-EM entry with 1–4 polymer chains; with 20 or more |
| Training overlap | proteins with no structure public before the model's training cutoff (Section 6.6) |
| Model confidence | mean pLDDT ≥ 70 |
| Definition of the baseline | strict rule (largest prediction distance ≤ largest experimental distance); X-ray–cryo-EM distance only; NMR–cryo-EM distance only; tightest experimental pair |

### 6.5 Method pairs and context

For every pair of methods (e.g. X-ray–AlphaFold2, NMR–cryo-EM), each protein contributes the median
distance over its pairs of that type, and the distribution over proteins is summarised. For proteins
with all three experimental methods, the experimental method closest to each prediction is determined.
Distances are further stratified by the number of polymer chains in the cryo-EM entry, since cryo-EM
chains are frequently part of large assemblies while NMR measures the isolated protein.

### 6.6 Training overlap

AlphaFold2 was trained on PDB entries released up to 30 April 2018 [8] and OpenFold3 on entries
deposited before 30 September 2021 (OpenFold3 model card). A protein is considered seen by a model when
at least one of its experimental structures in the dataset was public before the corresponding cutoff.
Proteins without dates are excluded from this analysis.

## 7. Information content of the metrics

Notebook 05 (`src/information_imbalance.py`) examines how much information the measures share and which
protein properties are associated with a prediction far from the experiments, using the Information
Imbalance [18]. For two representations A and B of the same N points,

$$
\Delta(A \rightarrow B) = \frac{2}{N} \sum_{i=1}^{N} r^{B}_{i,\,\mathrm{nn}_A(i)},
$$

where $\mathrm{nn}_A(i)$ is the nearest neighbour of point $i$ in A and $r^{B}_{ij}$ the rank of $j$ among
the neighbours of $i$ in B. Values near 0 indicate that A contains the neighbourhood information of B;
values near 1 indicate that it contains none. Metric values enter as standardised
log₁₀(value + 10⁻⁴).

1. **Redundancy among the measures.** Points are comparisons; Δ is computed between every pair of
   measures (three random subsamples of 1,000 comparisons, and one comparison per protein to avoid
   repeated proteins), and the Differentiable Information Imbalance (DII) [19], implemented in DADApy
   [20], is used for backward feature selection with all four measures as the target.
2. **Machaon metrics and RMSD.** DII from the three Machaon metrics to RMSD, with the learned weight of
   each metric.
3. **Orientation.** Whether the information of t-alpha not shared with the other measures is
   orientation noise, by comparing Machaon's frame with the principal-axes frame.
4. **Protein properties.** Points are proteins. DII backward selection from protein properties (mean
   pLDDT, length, size of the cryo-EM assembly, availability of X-ray, share of entries before the
   AlphaFold2 cutoff, largest length gap, experimental variability) to the distances of the prediction
   from the experimental structures.

When the target is built from the same measures as the input, Δ quantifies redundancy rather than
accuracy: a measure dominated by noise shares little with the others and cannot be removed. Analyses 2
and 4 have independent targets and are not affected.

## 8. OpenFold3 input representations

Notebook 06 extracts the input embedder of the trained OpenFold3 and computes per-residue
representations for the proteins of the dataset. It runs on Google Colab and its results are not used
in Sections 4–7.

The OpenFold3 release and checkpoint are pinned in the notebook and belong to the same model family,
with the same training cutoff, as the NIM predictions. The input embedder (`InputEmbedderAllAtom`,
corresponding to Algorithms 1–3 of AlphaFold3 [21]) is instantiated alone and only its weights are
loaded from the checkpoint (strict loading). Its output is verified to match that of the full model on
the same feature batch.

Input features are produced by the OpenFold3 inference data pipeline without the model, from the
UniProt sequence alone (no MSA). Each residue's reference conformer is generated stochastically (RDKit
ETKDG [22], followed by a random rotation and translation), and OpenFold3 seeds its random state only
after featurisation; the random state is therefore fixed before featurisation, so that the
representation is a deterministic function of the sequence and the seed. The module returns a
single-residue representation before and after its final projection and an initial pair representation.
The representations are analysed for the information they carry about residue identity and spatial
proximity, and compared with the trunk representation of the full model on a subset of proteins.

## 9. Software and reproducibility

The analysis is written in Python with NumPy, SciPy, pandas, gemmi [6], Biopython [7], Open3D [14],
DADApy [20] and Matplotlib; exact versions are pinned in `requirements.txt`. Random procedures use fixed
seeds. Each run of the comparison pipeline records its metadata in `results/pipeline_summary.json`
(timestamp, git commit, library versions, NMR model rule and an MD5 checksum of every input structure).
Because OpenFold3 diffusion is stochastic and the hosted model may change, OpenFold3 predictions are
reproducible from the stored files and their provenance rather than by re-prediction.

## 10. Limitations

1. **Descriptor-based metrics.** The Machaon metrics compare per-chain summaries and can assign zero
   distance to different structures with matching summaries; RMSD is reported alongside for this reason.
2. **Coverage.** Whole-chain descriptors combine conformational differences with differences in the
   modelled residues; coverage-matched subsets bound the effect at the cost of fewer proteins.
3. **Orientation dependence of t-alpha**, of a magnitude comparable to differences between methods; the
   principal-axes variant quantifies the remaining signal.
4. **b-phipsi** compares the mean and covariance of the dihedral angles, not the full shape of their
   distribution, and treats the angles as linear variables (no periodicity at ±180°), as in Machaon.
5. **One chain per method.** For proteins with several conformational states, the result depends on
   the structure that represents each method.
6. **Context.** Cryo-EM chains frequently come from assemblies, while NMR measures the isolated protein.
7. **Training overlap.** Most proteins had a structure public before the models' training cutoffs; the
   subset of unseen proteins is small.
8. **Static predictions** are compared with ensembles and with structures determined in crystal or
   assembly environments; the scope is limited to single chains.
9. **Unequal prediction inputs.** AlphaFold Database models use deep MSAs and templates, while the
   OpenFold3 models use ColabFold MSAs and no templates, and the training cutoffs differ. The comparison
   of AlphaFold2 with OpenFold3 therefore concerns these two sets of models rather than the two
   architectures.
10. **Hosted prediction service.** OpenFold3 is accessed through NVIDIA's hosted API; the served model
    version is outside the control of the study and is recorded only when reported in the response
    headers.
11. **Variability estimate.** For proteins without an X-ray structure, the experimental variability
    rests on a single NMR–cryo-EM pair.

## References

1. Kakoulidis, P. et al. *Commun. Biol.* **6**, 752 (2023). https://doi.org/10.1038/s42003-023-05076-7
2. Velankar, S. et al. SIFTS: Structure Integration with Function, Taxonomy and Sequences resource.
   *Nucleic Acids Res.* **41**, D483–D489 (2013).
3. Berman, H., Henrick, K. & Nakamura, H. Announcing the worldwide Protein Data Bank.
   *Nat. Struct. Biol.* **10**, 980 (2003).
4. The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2023.
   *Nucleic Acids Res.* **51**, D523–D531 (2023).
5. Varadi, M. et al. AlphaFold Protein Structure Database. *Nucleic Acids Res.* **50**, D439–D444
   (2022).
6. Wojdyr, M. GEMMI: a library for structural biology. *J. Open Source Softw.* **7**, 4200 (2022).
7. Cock, P. J. A. et al. Biopython. *Bioinformatics* **25**, 1422–1423 (2009).
8. Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. *Nature* **596**,
   583–589 (2021).
9. OpenFold3. https://github.com/aqlaboratory/openfold-3
10. Mirdita, M. et al. ColabFold: making protein folding accessible to all. *Nat. Methods* **19**,
    679–682 (2022).
11. Steinegger, M. & Söding, J. MMseqs2 enables sensitive protein sequence searching for the analysis
    of massive data sets. *Nat. Biotechnol.* **35**, 1026–1028 (2017).
12. Bhattacharyya, A. On a measure of divergence between two statistical populations defined by their
    probability distributions. *Bull. Calcutta Math. Soc.* **35**, 99–109 (1943).
13. Edelsbrunner, H. & Mücke, E. P. Three-dimensional alpha shapes. *ACM Trans. Graph.* **13**, 43–72
    (1994).
14. Zhou, Q.-Y., Park, J. & Koltun, V. Open3D: a modern library for 3D data processing.
    arXiv:1801.09847 (2018).
15. Kabsch, W. A solution for the best rotation to relate two sets of vectors. *Acta Cryst. A* **32**,
    922–923 (1976).
16. Lakens, D. Calculating and reporting effect sizes to facilitate cumulative science.
    *Front. Psychol.* **4**, 863 (2013).
17. Efron, B. & Tibshirani, R. J. *An Introduction to the Bootstrap* (Chapman & Hall, 1993).
18. Glielmo, A. et al. Ranking the information content of distance measures. *PNAS Nexus* **1**,
    pgac039 (2022).
19. Wild, R. et al. Automatic feature selection and weighting in molecular systems using
    Differentiable Information Imbalance. *Nat. Commun.* (2025).
    https://doi.org/10.1038/s41467-024-55449-7
20. Glielmo, A. et al. DADApy: distance-based analysis of data-manifolds in Python. *Patterns* **3**,
    100589 (2022).
21. Abramson, J. et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3.
    *Nature* **630**, 493–500 (2024).
22. Riniker, S. & Landrum, G. A. Better informed distance geometry: using what we know to improve
    conformation generation. *J. Chem. Inf. Model.* **55**, 2562–2574 (2015).
