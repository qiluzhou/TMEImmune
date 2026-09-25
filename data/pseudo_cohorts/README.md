# Synthetic Multi-Cohort TME/ICI Harmonization Dataset

Synthetic data for testing a TME/ICI harmonization pipeline. No real patient data are included.

## Files
Each cohort has clinical data and expression data. Mutation data are available for Cohorts A and B only.

- `Cohort_A_clinical.csv`
- `Cohort_A_expression.csv`
- `Cohort_A_mutation.csv`
- `Cohort_B_clinical.csv`
- `Cohort_B_expression.csv`
- `Cohort_B_mutation.csv`
- `Cohort_C_clinical.csv`
- `Cohort_C_expression.csv`
- `gene_mapping_reference.csv`
- `schema_manifest.csv`
- `dataset_metadata.json`

## Intentional heterogeneity
- Different sample-ID column names
- Gene symbols vs Ensembl-like IDs
- log2(TPM+1)-like, TPM, and raw-count-like expression
- Different response encodings
- Different clinical column names and categorical encodings
- Missing genes in Cohort B
- Different mutation schemas
- Mutation data absent from Cohort C

## Recommended tests
1. Sample-ID harmonization
2. Gene-ID mapping
3. Expression-scale detection/transformation
4. Clinical variable harmonization
5. Response-label harmonization
6. Cancer/treatment harmonization
7. Missing-gene reporting
8. Optional mutation integration
9. Cohort provenance preservation
10. Consistent TME scoring after harmonization

The synthetic generator contains a known latent immune-response signal and stromal-response signal for software-validation purposes only.
