# TMEImmune v2 — package status

Working note for the v2 release and the accompanying paper. Covers what each module contains, what
changed during this development pass, what has actually been verified, and what is still open.

Version `2.0.0` (from 1.3.3) · last updated 2026-09-23

---

## 1. Status at a glance

| Area | State |
|---|---|
| ISAFN scoring (expression + fusion) | done, reproduces your notebook exactly |
| Sex imputation from expression | done |
| Expression unit auto-detection → log2(TPM+1) | done |
| Gene-identifier harmonisation (Ensembl / Entrez / alias → symbol) | done |
| Gene coverage / targeted-panel detection | done |
| Clinical harmonisation across cohorts | done |
| Multi-cohort merge with provenance | done |
| Performance table, top-5 ROC/KM | done |
| Caching + parallelism + benchmark harness | done |
| Batch-correction comparison | **partly** — ComBat arm unrun |
| Decision curve analysis | not started |
| Redundancy analysis | not started |
| Docker image v2 | not started |

**Two things to look at before moving on** — see §7: the
`MANIFEST.in` packaging issue that silently undoes the 90 MB size reduction.

---

## 2. Module inventory

New in this development pass is marked **[new]**; substantially rewritten is **[rewritten]**.

### `data_processing.py` (1037 lines) — the largest change

Constants: `EXPRESSION_TYPES`, `PANEL_COVERAGE` (0.20), `PANEL_MIN_GENES` (2000),
`TRANSCRIPTOME_METHODS`, `CLINICAL_SYNONYMS`, `RESPONSE_MAP`, `CANCER_SYNONYMS`, `DRUG_SYNONYMS`,
`BATCH_METHODS`.

| Function | Purpose |
|---|---|
| `load_data(path, cache=True)` | **[rewritten]** packaged data loader, now memory-cached (20 MB cap) |
| `clear_data_cache()` | **[new]** empty that cache |
| `read_gct` / `read_data` | file readers (unchanged) |
| `detect_expression_type(df, axis)` | **[new]** counts / TPM / CPM / FPKM / log / z-score / centered |
| `reference_genes(reference)` | **[new]** the gene universe to compare against: `'isafn'` (12,013 training genes), `'annotation'`, or your own list |
| `assess_gene_coverage(df, reference, min_coverage, min_genes)` | **[new]** returns `n_input_genes`, `n_reference_genes`, `n_overlap`, `coverage`, `likely_targeted_panel`, `incompatible_methods` |
| `to_log2tpm(df, gene_length, detected, verbose, force, rescale, allow_tpm, reference)` | **[new]** converts to log2(TPM+1); `allow_tpm='auto'` refuses the sum-to-1e6 step on a targeted panel |
| `harmonize(df, path, gene_length, convert_ids, aggregate, verbose, force, rescale, mapping, allow_tpm)` | **[new]** identifiers + units in one call, with a full transformation report |
| `is_log2_transformed(df)` / `log2_transform(df, log2)` | **[new]** GEO2R-style log detection and log2(x+1) |
| `normalization(..., log2=None)` | existing; gained auto-log2 before normalising |
| `harmonize_clinical(clin, sample_col, response_col, sex_col, met_col, drug_col, cancer_col, age_col, cohort, verbose)` | **[new]** column-name matching + value standardisation |
| `combat(df, batch, covariates, method, verbose)` | **[new]** inmoose ComBat; `'auto'` picks `pycombat_seq` for integer counts, `pycombat_norm` otherwise |
| `correct_batch(df, batch, method, covariates, verbose)` | **[new]** one of `none / combat / minmax_global / minmax_cohort / zscore_cohort` |
| `merge_cohorts(expression, clinical, mutation, how, mapping, gene_length, batch_method, protect, verbose)` | **[new]** per-cohort harmonisation then stacking; returns `(expr, clin, mut, report)` |
| `gene_id_needed`, `_mutation_ids_to_symbol`, `_norm_key`, `_find_column` | **[new]** helpers |

`harmonize_clinical` output columns: `resp` (1/0), `sex` (`Male`/`Female`), `met1` (1/0), `drug`
(ISAFN's names), `cancer_type` (`melanoma`/`bladder`/`ccrcc`/`unknown`), `age`, `source`. Original
columns are kept untouched.

### `gene_id.py` (242 lines) — **[new module]**

| Function | Purpose |
|---|---|
| `annotation_available()` | is `data/gene_annotation.npz` present |
| `load_annotation()` | cached load (symbols, Ensembl, Entrez, length, alias map) |
| `detect_id_type(ids, sample=2000)` | symbol / ensembl / entrez / refseq / mixed / unknown |
| `read_mapping(mapping)` | **[new]** normalise a user mapping — dict, Series, 2-col frame or csv; auto-detects which column holds the symbols |
| `to_symbol(df, id_type, aggregate, keep_unmapped, verbose, mapping)` | index → HGNC symbols; `mapping=` overrides the packaged table |
| `_relabel(...)` | shared relabel + duplicate-merge path |
| `gene_lengths(symbols=None)` | union-exon lengths for counts → TPM |

### `ISAFN.py` (484 lines) — **[rewritten]**

Constants: `CANCER_MAP` (`melanoma` 0, `ccrcc` 1, `unknown` 2, `bladder` 3), `DRUG_MAP`, `SEX_MAP`,
`GENE_META_COLS`, `MUT_META_COLS`, `MMR_GENES`, `Y_GENES`, `TRT_PD1_PDL1`.

| Function | Purpose |
|---|---|
| `load_isafn_models()` | unpickles `fusion_models_final.pkl`, redirects `model_utils.*` → `isafn_models`, maps to CPU |
| `isafn_norm(X_test, clin_test, cohort_col, gender, features)` | cohort-wise min-max alignment to the stored training range |
| `impute_sex(df_gene, log2, min_gap)` | sex from chrY genes + XIST |
| `mutation_to_matrix(maf, sample_col, gene_col, binary, variant_col, exclude_variants)` | long-format MAF → samples × genes |
| `_encode_sex / _encode_drug / _encode_met` | covariate encoders |
| `_encode_cancer(s, default_key)` | **[new]** per-sample cancer type → `CANCER_MAP` |
| `msi_status`, `compute_mutation`, `_mutation_summary`, `_build_inputs`, `predict_fusionmodel` | internals |
| `isafn_score(...)` | the main entry point |

`isafn_score` signature: `df_gene, clin, test_clinid, df_mut, gender_col, drug_col, met_col, cancer,
sex_output, source_col, log2, impute_sex, gene_length, rescale, convert_ids, cancer_col`.
Returns `isafn_expr_prob/pred` and, with mutations, `isafn_fusion_prob/pred`.
`attrs`: `imputed_sex`, `input_report`.

Mutation input column order is `['total_count', 'mut_count', 'mut_max', 'mut_mean', 'cancer_type',
'trt1', 'sex']` — MSI is not used.

### `TME_score.py` (296 lines)

`get_score(df, method, clin, test_clinid)` — ESTIMATE, ISTME, NetBio, SIA.
Signature helpers: `get_geomean_score`, `get_avgmean_score`, `get_ratio_score`, `get_impres_score`,
`get_ssgsea_score`.

`get_all_score(df, clin, response_col, source_col, df_mut, gender_col, drug_col, met_col, cancer,
sex_output, impute_sex, cancer_col, skip_failed)` — **[new]** all 21 columns: CYT1, CYT2, IFNr, TLS,
TIS, TIP Hot, TIP Cold, CS Polarity, IMPRES, TGFb, ISTME, ESTIMATE, SIA, NetBio, and the four ISAFN
columns. With `skip_failed=True` (default) a score that cannot be computed returns NaN and the reason
lands in `output.attrs['failed_scores']`; `output.attrs['gene_coverage']` holds the coverage report.

### `optimal.py` (515 lines) — **[rewritten]**

| Function | Purpose |
|---|---|
| `assign_type(score, upper_p, lower_p)` | High/Medium/Low grouping (M/L labels were swapped — fixed) |
| `optimal_ICI` / `optimal_survival` | public wrappers over `_optimal_ICI` / `_optimal_survival` |
| `performance_table(auc_dict, c_dict, logrank_p, n_ici, n_surv)` | **[new]** every score's performance |
| `get_performance(..., top_n=5, return_table=False)` | **[new]** plots the top 5 only, returns the full table |
| `batch_separability(expr, batch, n_components, seed)` | **[new]** how predictable the cohort still is (PCA + CV logistic + silhouette) |
| `compare_batch_correction(expression, clinical, score_fn, methods, response_col, source_col, protect, leave_one_out, verbose)` | **[new]** signal kept vs cohort effect removed |

### `netbio.py` (153 lines) / `nb_utilities.py` (311 lines)

`netbio.py`: `_TRAIN_CACHE`, `clear_netbio_cache()`, `_fingerprint()`, `_select_proximal_pathways()`
(vectorised hypergeometric), `_train_netbio()` (cached), `get_netbio(...)` — returns a **Series
indexed by sample**, not a bare array.

`nb_utilities.py`: `_rank_matrix()` (cached), `clear_ssgsea_cache()`, `ssgsea()` (closed-form, fast),
`ssgsea_reference()` (the original, kept for verification),
`parse_reactomeExpression_and_immunotherapyResponse()` (reads the trimmed
`data/Gide/gide_training.npz`), `expression_StandardScaler()` (vectorised),
classes `nb_pathway`, `netbio_data`.

### `benchmark.py` (180 lines) — **[new module]**

`clear_all_caches`, `timeit`, `score_tasks`, `benchmark_scores`, `benchmark_ssgsea`,
`compare_implementations`, `plot_benchmark`.

### `parallel.py` (223 lines) — **[new module]**

`score_cohorts`, `METRICS`, `_boot_indices`, `_auc_bootstrap_fast`, `bootstrap_metric`,
`bootstrap_best_score` (optimism-corrected, out-of-bag).

### Unchanged modules

`estimateScore.py`, `ISTME.py`, `SIAscore.py`, `isafn_models.py` (lean copies of `mut_model`,
`gene_model`, `Classifier_fusion`).

### `__init__.py`

Exports `get_score`, `get_all_score`, `isafn_score`, `impute_sex`, `mutation_to_matrix`, `to_symbol`,
`detect_id_type`, `gene_lengths`, plus `*` from `data_processing`, `estimateScore`, `ISTME`,
`netbio`, `SIAscore`, `optimal`, `nb_utilities`.

---

## 3. Packaged data

| Path | Size | Notes |
|---|---|---|
| `data/gene_annotation.npz` | 1.7 MB | 109,173 symbols · 77,938 Ensembl · 45,256 Entrez · 77,118 lengths · 58,835 aliases. Built from `gencode.v50.basic.annotation.gtf.gz` + `hgnc_complete_set.txt` + `symbol2entrez.csv` |
| `data/Gide/gide_training.npz` | 0.9 MB | trimmed NetBio training table |
| `data/Gide/expression_mRNA.norm3.txt` | **91 MB** | **superseded — see §7** |
| `data/Gide/pathway_expression_ssgsea.txt` | 1.7 MB | superseded |
| `data/isafn/` | 2.8 MB | `fusion_models_final.pkl`, `features_output.json`, `cohort_minmax_global.json` |
| `data/c2.all.v7.2.symbols.gmt` | 3.8 MB | Reactome/C2 gene sets |
| `data/nb_biomarker/` | 4.1 MB | NetBio biomarker tables |
| `data/gene_sets_full.json` | 620 KB | signature gene sets |

### `tools/`

- `build_gene_annotation.py` — `--gtf`, `--hgnc`, `--symbol2entrez`; union-exon lengths, alias map.
  Merges the sources **field-wise** so a symbol keeps the GENCODE length *and* the HGNC Entrez id.
- `build_gide_training.py` — builds the trimmed NetBio training table.

---

## 4. What changed, by theme

### ISAFN completion and validation
Finished `isafn_score`; built `isafn_models.py` and the `model_utils` redirect so the pickle loads
without your training code. Fixed draft bugs: dotted data paths, wrong `predict_fusionmodel`
signature, covariates dropped by `reindex`, sex-specific normalisation not using the male/female
ranges, `if df_mut != None`. Made ISAFN the default score rather than optional.

Reproduced your Riaz results exactly once imputed sex (20 M / 20 F from chrY genes) and `met=1` were
accounted for: **male AUC 0.7344, female AUC 0.6863** — identical to your training numbers.

### Sex imputation
`impute_sex()` uses chrY genes (`RPS4Y1, DDX3Y, KDM5D, EIF1AY, UTY, ZFY`) plus XIST: scans upward for
the first gap ≥ `min_gap`, accepting the split only if the lower group has higher XIST, else falls
back to a per-sample chrY-vs-XIST comparison. Sex enters **every** ISAFN model — the sex embedding,
the male/female adapters, and the cohort gate in `gene_model` — including the merged model.

### Evaluation fixes
Fixed three `optimal.py` bugs (swapped M/L labels, `clinical_factors` ignored, c-index box overflow),
limited ROC/KM plots to the top 5 scores, and added a performance table covering all of them.

### Reproducibility bug (important)
`netbio_data.common_id` used `list(set(...))`, so with Python hash randomisation **NetBio scores were
attached to the wrong samples and changed between runs**. Fixed by using an ordered intersection and
returning a sample-indexed Series. Any NetBio numbers produced before this fix should be recomputed.

### Performance
ssGSEA rewritten in closed form with a cached rank matrix — **~20–200× faster**, output verified
identical to the original (`ssgsea_reference`). NetBio training cached and pathway selection
vectorised. `benchmark.py` and `parallel.py` added (joblib, vectorised bootstrap AUC).

### Input harmonisation
Unit auto-detection and conversion to log2(TPM+1); an explicit notice when data look z-scored.
Gene-identifier conversion. Gene-length lookup from the packaged annotation.

### Gene annotation build fix
The source merge kept one row per symbol by a "completeness score", but GENCODE rows (length +
Ensembl) tied with HGNC rows (Entrez + Ensembl) and `sort_values` is not stable for a single column —
so about 26,000 lengths were being discarded. Now each field is filled from the first source that has
it: **lengths went from 36,184 to 77,118**.

### Multi-cohort harmonisation (this session)
Driven by the three synthetic cohorts. Added `assess_gene_coverage`, panel gating, `mapping=`,
`harmonize_clinical`, `merge_cohorts`, `combat`/`correct_batch`, `cancer_col`, and
`get_all_score(skip_failed=True)`. See §6.

---

## 5. Decisions on record

| Decision | Choice |
|---|---|
| Response encoding | **CR + PR = responder; SD + PD = non-responder** (applies to all future data) |
| Multi-cohort gene set | **union, keeping NaN** where a cohort did not measure a gene |
| Targeted-panel input | TPM conversion **prohibited**, original scale preserved, incompatibility reported |
| Cancer type across cohorts | per-sample `cancer_col`; urothelial → `bladder`; compare per-cohort vs merged |
| ComBat | use `inmoose.pycombat`, do not reimplement |
| Batch-correction criterion | signal kept **and** cohort effect removed, plus leave-one-cohort-out |
| Cross-sex AUC comparison | not valid (different models, different sample sets) — excluded |
| Survival analysis | dropped to secondary; focus is immune-response prediction |
| Sex-difference methodology, per-sample explanations | covered by your other paper, out of scope here |

---

## 6. Multi-cohort test results

Three synthetic cohorts, deliberately inconsistent:

| | ids | scale | response | cancer | mutation |
|---|---|---|---|---|---|
| A | symbol | log2(TPM+1) | Responder/Non-responder → 25/55 | melanoma | yes |
| B | Ensembl-like | TPM | CR/PR/SD/PD → 23/48 | melanoma | yes (Ensembl ids in a `Hugo_Symbol` column) |
| C | symbol | raw counts | 1/0 → 19/52 | urothelial → **bladder** | **no** |

All identifiers, scales and clinical column names were detected automatically. Merged result:
**79 genes × 155 samples, 240 missing values — exactly B's 5 missing genes × 48 samples, nowhere
else.** All seven combinations (A, B, C, A+B, A+C, B+C, A+B+C) score end to end without error.

**Findings:**

1. **Per-cohort scoring and merged scoring are identical** — ISAFN r = 1.0000 over 155 samples.
   Because `source_col` makes ISAFN min-max align within each cohort, merging changes nothing. Worth
   stating in the paper as a property rather than presenting as two conditions.

2. **SIA and NetBio genuinely cannot run on an 80-gene panel.** NetBio previously died with an opaque
   sklearn "all 45 fits failed"; the real cause is that only 78 genes are shared with the training
   data, yielding **0 proximal pathways**. It now raises a message naming gene coverage.

3. **`minmax_global` and `minmax_cohort` are indistinguishable to the batch metric** (both 0.82 /
   0.0655) — they differ by a per-gene affine map that the standardisation inside
   `batch_separability` undoes. They differ only downstream (mean AUC 0.593 vs 0.585). Uncorrected is
   1.00 / 0.584, so both corrections do remove real cohort structure.

4. **Panel detection fixed a scale mismatch.** Forcing 80-gene samples to sum to 1e6 inflated values
   by ~275× (about 8 log2 units), leaving cohort medians at 6.84 / 13.12 / 12.59. With the panel gate
   they sit at 6.84 / 7.17 / 5.10. Transcriptome-wide input is unaffected (Riaz: 99.99% coverage).

---

## 7. Open items

### a) `MANIFEST.in` undoes the 90 MB size reduction — **packaging bug**
`setup.py` correctly lists only `data/Gide/gide_training.npz`, but `MANIFEST.in` still has
`recursive-include TMEImmune/data/Gide *`, and with `include_package_data=True` that pulls the 91 MB
`expression_mRNA.norm3.txt` back into the distribution. `data/` is currently **103 MB**. Suggested
fix — replace that line with:

```
include TMEImmune/data/Gide/gide_training.npz
```

and decide whether to keep the raw Gide files in the repo at all (they are only reachable via
`parse_reactomeExpression_and_immunotherapyResponse(..., raw=True)`).

### d) `rescale` default for gene-subset TPM
Still undecided for transcriptome-scale input where some genes are absent (difference measured
earlier: 0.014 in probability, 0.005 in AUC). The panel gate handles the extreme case; this is the
middle ground.

### e) Not started
Decision curve analysis · redundancy analysis · Docker image v2 · ISAFN paper citation placeholder in
the README.

---

## 8. Verification status — read this before trusting any number

**Every ISAFN figure produced in this environment comes from a NumPy re-implementation of the model's
forward pass, not from PyTorch.** `torch`, `gseapy`, `statsmodels`, `lifelines`, `inmoose` and
`rnanorm` cannot be installed here (no network access to the package index), so they are stood in for
by a NumPy reference implementation plus stub modules.

The re-implementation reproduced your notebook's Riaz AUCs to four decimal places, which is good
evidence it is faithful — but it is not the shipped code path.

**Please re-run these in `myenv` with the real dependencies:**

```bash
python tests/test_isafn.py        # ISAFN on pseudo data
python tests/test_harmonize.py    # unit detection and conversion
python tests/test_gene_id.py      # identifier conversion
python tests/test_cohorts.py      # three-cohort harmonisation  <- includes the ComBat arm
python tests/test_efficiency.py   # caching and speed
python tests/test_parallel.py     # parallel scoring and bootstrap
python tests/riaz_example.py      # Riaz end to end
python tests/riaz40_compare.py    # the 40-sample comparison against your results
python tests/test_evaluation.py   # decision curve analysis and redundancy
```

Note `TGFb` fails in my environment with `NotImplementedError` — that is the gseapy stub, not a real
failure; it should pass for you.

If you hit `ImportError: cannot import name ... from 'TMEImmune'`, a stale copy in
`site-packages` is shadowing the repo — `pip install -e .` from the project root fixes it.

### Evaluation functionality added 2026-09-23

Two evaluation entry points were added to `optimal.py`, covered by `tests/test_evaluation.py`. The
numbers below are from Riaz pre-treatment (49 samples, 10 responders, 20.4% response rate) and, like
everything else in this section, were produced with the NumPy stand-in rather than PyTorch.

**Decision curve analysis** — `net_benefit`, `decision_curve`, `plot_decision_curve`.

Each score is put on a probability scale by a single-variable logistic fit, because net benefit is
only defined against a risk threshold. `cv=` gives out-of-fold calibration; the default fits in
sample, which is the textbook setup but optimistic in absolute terms. The comparison between scores
stays fair either way, since all are calibrated identically.

Scores are ranked by `area_above_default` — net benefit above the better of treat-all and treat-none.
Raw net benefit cannot be used for ranking: at a 0.01 threshold every score simply treats everyone, so
its best net benefit is just the response rate, identical for all of them.

| score | area_above_default | best_advantage | best_threshold | useful range |
|---|---|---|---|---|
| TLS | 0.0080 | 0.0350 | 0.30 | 0.11–0.53 |
| isafn_expr_prob | 0.0067 | 0.0590 | 0.17 | 0.06–0.36 |
| TIS | 0.0060 | 0.0612 | 0.35 | 0.08–0.36 |
| CYT1 | 0.0047 | 0.0492 | 0.27 | 0.11–0.33 |
| CYT2 | 0.0012 | 0.0483 | 0.24 | 0.14–0.25 |

With `cv=5` ISAFN ranks first (0.0052 against TLS 0.0048): it loses least to out-of-fold calibration,
which is the more defensible number to quote.

**Redundancy** — `score_correlation`, `discordant_pairs`, `disagreement_cases`. Nothing is fitted;
these are rank comparisons on the scores as they are.

ISAFN forms its own cluster (Spearman r = 0.11–0.38 against the other scores), while CYT1 and TIS are
near-duplicates at r = 0.957.

| comparator | AUC ISAFN | AUC other | difference | pairs disagreeing | ISAFN wins | sign test p |
|---|---|---|---|---|---|---|
| CYT2 | 0.6897 | 0.5641 | 0.1256 | 41.3% | 65.2% | 0.0001 |
| TLS | 0.6897 | 0.6205 | 0.0692 | 40.5% | 58.9% | 0.0409 |
| CYT1 | 0.6897 | 0.6256 | 0.0641 | 31.5% | 60.2% | 0.0300 |
| TIS | 0.6897 | 0.6308 | 0.0590 | 34.6% | 58.5% | 0.0579 |

ISAFN disagrees with the other scores on 32–41% of responder/non-responder pairs and wins 59–65% of
those, which is the argument that it is not a relabelling of an existing score. Among the 10 patients
ISAFN and TIS rank most differently, 30% responded where ISAFN ranks higher against 20% where TIS
does, on a 20.4% base rate — illustrative only at that sample size.

At n = 49 with 10 responders the decision-curve differences are well within noise. The redundancy
sign tests are better powered, since they are paired over hundreds of pairs.

**Two implementation bugs found and fixed while building this, both of which would have changed the
conclusion:**

1. *Calibration was not scale-invariant.* sklearn's `LogisticRegression` defaults to L2 with `C=1.0`.
   ISAFN's raw probabilities span only 0.424–0.523, so the coefficient needed is about 30 and the
   penalty shrank it, collapsing the calibrated risks to 0.203–0.205 — effectively constant. ISAFN
   came out last with `area_above_default = 0.0000`, which would have read as "best AUC, no clinical
   utility". The predictor is now standardised and fitted with `penalty=None`; ISAFN's risks span
   0.054–0.402 and it ranks second. The test asserts that multiplying a score by 1000 and adding 7
   leaves its curve unchanged.

2. *Tie handling in the pair counting.* Win and loss counts reproduce the AUC difference exactly only
   when no scores tie, and TLS has ties. `discordant_pairs` now reports `n_tied_pairs` and
   `n_clear_pairs` separately, with the sign test using only the clear-cut pairs, and the test
   validates the pairwise AUCs against `roc_auc_score`.

`net_benefit` is checked against its closed forms: a perfect predictor returns the response rate at
every threshold, treating everyone reproduces the treat-all line, treating no one gives zero, and
treat-all at a threshold equal to the response rate is exactly zero.

---

## 9. Suggested next steps

1. data_processing if zscore or centered detected and data can't be recovered, instead of return the result of that, add a step for the user to choose if they want to continue with this incorrect data. just show them the warning and let them click or input yes or no to continue or not. test the function with a zscore pseudo data. also update to log2tpm
2. Apply the `MANIFEST.in` fix (§7b) and rebuild, confirming the wheel is ~12 MB rather than ~103 MB. I added include TMEImmune/data/Gide/gide_training.npz and removed the line recursive include gide. don't keep the raw gide file in the repo
3. Run the test suite in `myenv`, especially the ComBat arm. i have run the test suite combat runs
4. Then: decision curve analysis, redundancy analysis, Docker v2. add the decision curve analysis and redundancy analysis first, next we will do with the docker v2.
