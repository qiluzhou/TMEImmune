import io
import os
import json
import pickle
import warnings
from functools import lru_cache
import numpy as np
import pandas as pd
from TMEImmune.data_processing import is_log2_transformed, detect_expression_type, to_log2tpm

# torch is imported lazily (inside functions) so that `import TMEImmune` keeps working
# for users who only need the non-deep-learning scores.

_ISAFN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "isafn")

CANCER_MAP = {'melanoma': 0, 'ccrcc': 1, 'unknown': 2, 'bladder': 3}
DRUG_MAP = {
    'Pembrolizumab': 0,
    'Nivolumab': 1,
    'Ipilimumab + Pembrolizumab': 2,
    'Ipilimumab + Nivolumab': 3,
    'Bevacizumab+Atezolizumab': 4,
    'Atezolizumab': 5,
    'unknown': 6,
}
SEX_MAP = {'M': 0, 'MALE': 0, 'F': 1, 'FEMALE': 1}   # anything else -> 2 (unknown)
MET_UNKNOWN, SEX_UNKNOWN, DRUG_UNKNOWN = 2, 2, 6
SOURCE_UNKNOWN = 6      # cohort id used for cohorts that were not in training
TRT_PD1_PDL1 = 0        # treatment class: anti-PD1 / anti-PDL1 based

# column order expected at the end of the model inputs (see isafn_models.gene_model / mut_model)
GENE_META_COLS = ['drug', 'met1', 'source1', 'cancer_type', 'trt1', 'sex']
MUT_META_COLS = ['total_count', 'mut_count', 'mut_max', 'mut_mean', 'cancer_type', 'trt1', 'sex']

MMR_GENES = ['MLH1', 'MSH2', 'MSH3', 'MSH6', 'PMS2']


# --------------------------------------------------------------------------------------
# loading packaged resources
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_json(name):
    """Load a packaged ISAFN json (the min-max table is 2 MB and is read for every sex group).
    Cached, so callers must not modify the returned object in place."""
    with open(os.path.join(_ISAFN_DIR, name)) as f:
        return json.load(f)


_MODEL_CACHE = {}


def load_isafn_models():
    """Load the pretrained ISAFN models (cached after the first call).

    The pickle references classes from a module called ``model_utils`` (the training code);
    they are redirected to ``TMEImmune.isafn_models``. Tensors are mapped to CPU.
    """
    if 'models' in _MODEL_CACHE:
        return _MODEL_CACHE['models']
    import torch
    from TMEImmune import isafn_models

    class _Unused(torch.nn.Module):
        """Stand-in for training-time modules stored in the pickle but never called at prediction
        (e.g. SelfAttention sub-layers, ResidualModel). Their weights are loaded but ignored."""
        def forward(self, *args, **kwargs):
            raise RuntimeError("this module is not used for ISAFN prediction")

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module in ('model_utils', '__main__'):
                return getattr(isafn_models, name, _Unused)
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
            return super().find_class(module, name)

    with open(os.path.join(_ISAFN_DIR, "fusion_models_final.pkl"), "rb") as f:
        models = _Unpickler(f).load()
    for modality in models.values():
        for tup in modality.values():
            for m in tup:
                if hasattr(m, 'eval'):
                    m.eval()
    _MODEL_CACHE['models'] = models
    return models


# --------------------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------------------
def isafn_norm(X_test, clin_test, cohort_col='source', gender=None, features=None):
    """
    Cohort-wise min-max alignment of gene expression to the training range.
    Each cohort (clin_test[cohort_col]) is min-max scaled per gene and mapped onto the
    global [min, max] of the training data (precomputed in cohort_minmax_global.json).

    X_test: samples x genes dataframe
    clin_test: dataframe indexed by the same samples, containing cohort_col
    gender: None for the merged model, 'male' or 'female' for the sex-specific ranges
    features: genes to keep (default: all genes with a stored global range)
    """
    global_minmax = _load_json("cohort_minmax_global.json")
    global_name = 'global' if gender is None else 'global_' + gender.lower()
    global_min = pd.Series(global_minmax[global_name]['min'])
    global_max = pd.Series(global_minmax[global_name]['max'])

    if features is None:
        features = list(global_min.index)
    features = list(dict.fromkeys(features))                 # unique, keep order
    global_min, global_max = global_min.reindex(features), global_max.reindex(features)
    global_range = (global_max - global_min).replace(0, 1)

    X = X_test.reindex(columns=features, fill_value=0).astype(float).fillna(0)
    aligned_all = X.copy()
    cohorts = clin_test.loc[X.index, cohort_col]
    for cohort in cohorts.unique():
        idx = (cohorts == cohort).values
        cohort_data = X.loc[idx]
        cohort_min, cohort_max = cohort_data.min(), cohort_data.max()
        cohort_range = (cohort_max - cohort_min).replace(0, 1)
        aligned = ((cohort_data - cohort_min) / cohort_range) * global_range + global_min
        aligned_all.loc[idx] = aligned.values
    return aligned_all


def msi_status(pos):
    """1 if any mismatch-repair gene (MLH1, MSH2, MSH3, MSH6, PMS2) is mutated."""
    pos_selected = pos[pos.columns.intersection(MMR_GENES)]
    return (pos_selected.sum(axis=1) > 0).astype(int)


def compute_mutation(mut_df):
    mut_idx = pd.Index(mut_df.index.astype(str)).str.strip()
    counts = mut_idx.to_series().value_counts()
    maxc = int(counts.max()) if not counts.empty else 1
    counts_norm = (counts / maxc).rename('tm_norm')
    return counts_norm.fillna(0).astype(float)


def _encode_sex(s):
    return s.astype(str).str.strip().str.upper().map(SEX_MAP).fillna(SEX_UNKNOWN).astype(int)


def _encode_drug(s):
    key = lambda x: str(x).replace(' ', '').lower()
    lookup = {key(k): v for k, v in DRUG_MAP.items()}
    codes = s.map(lambda x: lookup.get(key(x), DRUG_UNKNOWN) if pd.notna(x) else DRUG_UNKNOWN)
    unknown = sorted(set(s[codes == DRUG_UNKNOWN].dropna().astype(str)) - {'unknown'})
    if unknown:
        warnings.warn(f"ISAFN: drug(s) {unknown} not seen in training, encoded as 'unknown'", UserWarning)
    return codes.astype(int)


def _encode_cancer(s, default_key='unknown'):
    """Per-sample cancer type -> the codes ISAFN was trained with; unrecognised values use default_key."""
    from TMEImmune.data_processing import CANCER_SYNONYMS
    default = CANCER_MAP.get(default_key, CANCER_MAP['unknown'])

    def enc(x):
        if pd.isna(x):
            return default
        v = str(x).strip().lower()
        v = CANCER_SYNONYMS.get(v, v)
        return CANCER_MAP.get(v, default)

    codes = s.map(enc).astype(int)
    unknown = sorted({str(v).strip() for v, c in zip(s, codes)
                      if pd.notna(v) and c == default and
                      CANCER_SYNONYMS.get(str(v).strip().lower(), str(v).strip().lower()) not in CANCER_MAP})
    if unknown:
        warnings.warn(f"ISAFN: cancer type(s) {unknown} not in {list(CANCER_MAP)}; "
                      f"encoded as '{default_key}'", UserWarning)
    return codes


def _encode_met(s):
    """Metastasis status -> 0 (no), 1 (yes), 2 (unknown). Accepts 0/1, bool or yes/no-like strings."""
    yes = {'1', '1.0', 'yes', 'y', 'true', 't', 'm1', 'met', 'metastatic', 'metastasis'}
    no = {'0', '0.0', 'no', 'n', 'false', 'f', 'm0', 'nonmetastatic', 'non-metastatic', 'primary'}

    def enc(x):
        if pd.isna(x):
            return MET_UNKNOWN
        v = str(x).strip().lower()
        return 1 if v in yes else 0 if v in no else MET_UNKNOWN
    return s.map(enc).astype(int)


def mutation_to_matrix(maf, sample_col='Tumor_Sample_Barcode', gene_col='Hugo_Symbol', binary=False,
                       variant_col=None, exclude_variants=('Silent', 'synonymous_variant')):
    """
    Convert long-format mutation calls (one row per variant, e.g. a MAF file) into the
    samples x genes matrix used by ISAFN (number of mutations of each gene in each sample).

    maf: dataframe with one row per mutation
    sample_col, gene_col: columns holding the sample ID and the gene symbol
    binary: if True, return 1/0 (mutated or not) instead of mutation counts
    variant_col: optional column with the variant classification; rows whose classification
                 contains any of exclude_variants (e.g. silent mutations) are dropped
    """
    df = maf.dropna(subset=[sample_col, gene_col])
    if variant_col is not None and exclude_variants:
        pattern = '|'.join(exclude_variants)
        df = df[~df[variant_col].astype(str).str.contains(pattern, case=False, regex=True)]
    mat = pd.crosstab(df[sample_col].astype(str), df[gene_col].astype(str))
    mat.index.name, mat.columns.name = None, None
    return (mat > 0).astype(int) if binary else mat


Y_GENES = ['RPS4Y1', 'DDX3Y', 'KDM5D', 'EIF1AY', 'UTY', 'ZFY']


def impute_sex(df_gene, log2=None, min_gap=1.0):
    """
    Impute biological sex from the expression of chrY genes (RPS4Y1, DDX3Y, KDM5D, EIF1AY, UTY, ZFY) and XIST.

    df_gene: gene expression, samples as index and gene symbols as columns (one cohort)
    log2: None to detect automatically and apply log2(x+1) to non-log data; True / False to force
    min_gap: minimal gap (log2 units) in the chrY score separating females from males

    chrY score = mean log2 expression of the chrY genes. Female samples sit at the bottom of the score
    with little spread, so the cohort is scanned upward from the lowest score and split at the first gap of at
    least min_gap; the split is accepted when the lower group also has higher XIST than the upper group.
    Without such a split (e.g. a single-sex cohort) each sample is called male when its chrY score is higher
    than its XIST expression. This is a heuristic: use recorded sex whenever it is available.
    Output: pandas Series of 'M' / 'F' indexed by sample (NaN if no chrY gene is available).
    """
    genes = [g for g in Y_GENES if g in df_gene.columns]
    if not genes:
        warnings.warn("impute_sex: no chrY genes found in the expression data; sex not imputed", UserWarning)
        return pd.Series(np.nan, index=df_gene.index, dtype=object)
    cols = genes + (['XIST'] if 'XIST' in df_gene.columns else [])
    x = df_gene[cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    if log2 is None:
        log2 = not is_log2_transformed(x)
    if log2:
        x = np.log2(x.clip(lower=0) + 1)
    y_score = x[genes].mean(axis=1)
    xist = x['XIST'] if 'XIST' in x.columns else None

    v = np.sort(y_score.values)
    gaps = np.diff(v)
    for i in np.where(gaps >= min_gap)[0]:
        cut = (v[i] + v[i + 1]) / 2
        low, high = y_score < cut, y_score >= cut
        if xist is None or xist[low].median() > xist[high].median():
            return pd.Series(np.where(high, 'M', 'F'), index=df_gene.index)
        break                                   # first large gap is not a female/male split

    # no clear female/male split (e.g. single-sex cohort): compare chrY with XIST per sample
    ref = xist if xist is not None else pd.Series(0.0, index=x.index)
    return pd.Series(np.where(y_score > ref, 'M', 'F'), index=df_gene.index)


_impute_sex = impute_sex     # alias: isafn_score has an argument with the same name


def _mutation_summary(df_mut):
    """Per-sample summary features computed on the full mutation matrix."""
    mut = df_mut.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    X_bin = (mut > 0).astype(float)
    out = pd.DataFrame(index=mut.index)
    total = mut.sum(axis=1)                          # number of mutations per sample
    out['total_count'] = total / total.max() if total.max() > 0 else 0.0   # scaled to the cohort maximum (as compute_mutation)
    out['mut_count'] = np.log1p(X_bin.sum(axis=1))
    out['mut_max'] = np.log1p(mut.max(axis=1).clip(lower=0))
    out['mut_mean'] = mut.mean(axis=1)
    return mut, out


def _build_inputs(gene_norm, mut, mut_summary, meta, gene_features, mut_features):
    """Assemble gene and mutation model matrices in the column order the networks expect."""
    X_gene = pd.concat([gene_norm.reindex(columns=gene_features, fill_value=0),
                        meta[GENE_META_COLS]], axis=1)
    if mut is None:
        X_mut_genes = pd.DataFrame(0.0, index=gene_norm.index, columns=range(len(mut_features)))
        summ = pd.DataFrame(0.0, index=gene_norm.index, columns=MUT_META_COLS[:4])
    else:
        X_mut_genes = mut.reindex(index=gene_norm.index, columns=list(dict.fromkeys(mut_features)), fill_value=0)
        X_mut_genes = X_mut_genes[mut_features]             # selected lists may repeat genes
        summ = mut_summary.reindex(gene_norm.index).fillna(0)
    X_mut = pd.concat([X_mut_genes.reset_index(drop=True),
                       summ[MUT_META_COLS[:4]].reset_index(drop=True),
                       meta.loc[gene_norm.index, MUT_META_COLS[4:]].reset_index(drop=True)], axis=1)
    return X_gene.values.astype(np.float32), X_mut.values.astype(np.float32)


# --------------------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------------------
def predict_fusionmodel(X_mut, X_gene, feature_extractor_gene, feature_extractor_mut, classifier, threshold):
    """Run one ISAFN model. X_mut / X_gene are float numpy arrays (samples x features)."""
    import torch
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    target_mutation = torch.tensor(X_mut, dtype=torch.float32, device=device)
    target_gene = torch.tensor(X_gene, dtype=torch.float32, device=device)

    for m in (feature_extractor_gene, feature_extractor_mut, classifier):
        m.to(device)
        m.eval()

    with torch.no_grad():
        outputs_mut, outputs_sex, outputs_cancer = feature_extractor_mut(target_mutation, target_mutation.shape[0])
        outputs_gene, met, drug = feature_extractor_gene(target_gene)
        logits = classifier(outputs_gene, outputs_mut, None, outputs_sex, outputs_cancer, met, drug).view(-1)
        probs = torch.sigmoid(logits)

    thr = 0.5 if (threshold is None or np.isinf(threshold)) else float(threshold)
    y_prob = probs.cpu().numpy().astype(float)
    y_pred = (y_prob >= thr).astype(int)
    return y_prob, y_pred


def isafn_score(df_gene, clin, test_clinid=None, df_mut=None, gender_col=None, drug_col=None, met_col=None,
                cancer='melanoma', sex_output=True, source_col=None, log2=None, impute_sex=False,
                gene_length='auto', rescale=True, convert_ids=True, cancer_col=None, confirm=None):
    """
    ISAFN prediction of immune checkpoint inhibitor (ICI) response.

    df_gene: gene expression, samples as index and gene symbols as columns
             (log-scale expression, e.g. log2(TPM+1), is expected).
    clin: clinical dataframe, samples as index.
    test_clinid: response column in clin (optional; not used for prediction, kept for API consistency).
    df_mut: optional mutation matrix, samples as index and gene symbols as columns
            (mutation counts or 0/1). When given, the gene+mutation fusion model is also run.
    gender_col: column of clin with sex coded "M"/"F" (Male/Female also accepted). Missing -> unknown.
    drug_col: column of clin with the ICI drug, e.g. "Pembrolizumab", "Nivolumab",
              "Ipilimumab + Nivolumab", ... Missing / unseen drugs -> unknown.
    met_col: column of clin with metastasis status (0/1, yes/no or M0/M1). Missing -> unknown.
    cancer: cancer type of the cohort: 'melanoma', 'ccrcc', 'bladder' or 'unknown'. Used for every
            sample that cancer_col does not give a value for.
    cancer_col: column of clin holding a per-sample cancer type, for a merged matrix spanning more than
                one tumour type. Values are matched against 'melanoma', 'ccrcc', 'bladder' and
                'unknown' (data_processing.harmonize_clinical writes this column as 'cancer_type');
                anything unrecognised falls back to the cancer= argument.
    sex_output: if True, male samples get the male-specific model output and female samples the
                female-specific model output (samples with unknown sex get the merged model).
                If False, every sample gets the merged (sex-agnostic) model output.
    source_col: optional clin column identifying cohorts/batches; expression is normalized within
                each cohort. By default all samples are treated as one cohort.
    impute_sex: if True, samples with missing / unknown sex (or all samples when gender_col is None) get
                their sex imputed from chrY gene and XIST expression (see impute_sex). The imputed values are
                stored in output_df.attrs['imputed_sex']. Sex matters for every ISAFN model, including the
                merged one (sex embedding, sex adapters and the cohort gate), so imputation is recommended when
                sex is not recorded.
    log2: ISAFN works on log2(TPM+1) expression. None (default) detects the input type (raw counts,
          TPM/CPM, FPKM, log-scale, z-scored) and converts it, printing what it found; True applies
          log2(x+1) without any other conversion; False uses the values as they are.
    gene_length: gene lengths in bases (Series indexed by gene symbol), used to turn raw counts into
                 TPM; without it counts become CPM, which ignores gene length
    rescale: rescale each sample to sum to 1e6 when the data are FPKM-like (see data_processing.to_log2tpm)
    convert_ids: map Ensembl ids, Entrez ids or outdated symbols onto current HGNC symbols before scoring
                 (needs the packaged gene annotation; see TMEImmune.gene_id)
    confirm: ask before a transformation that would change the shape of the expression data rather than
             just its units (see data_processing.to_log2tpm). 'auto' asks when someone can answer and
             declines otherwise; None (default) follows data_processing.set_confirm()

    Output: dataframe indexed by sample with columns
        isafn_expr_prob, isafn_expr_pred            (expression model)
        isafn_fusion_prob, isafn_fusion_pred        (gene + mutation model, only when df_mut is given)

    Note: expression is min-max aligned within each cohort, so scores depend on the whole cohort
    passed in -- score a cohort together rather than one sample at a time.
    """
    if not isinstance(df_gene, pd.DataFrame) or not isinstance(clin, pd.DataFrame):
        raise TypeError("df_gene and clin must be pandas dataframes")
    for col in (gender_col, drug_col, met_col, source_col, cancer_col):
        if col is not None and col not in clin.columns:
            raise ValueError(f"column '{col}' not found in clin")
    cancer_key = str(cancer).strip().lower()
    if cancer_key not in CANCER_MAP:
        warnings.warn(f"ISAFN: cancer type '{cancer}' not in {list(CANCER_MAP)}; using 'unknown'", UserWarning)
        cancer_key = 'unknown'

    # ---- align samples ----
    gene = df_gene.loc[~df_gene.index.duplicated(keep='first'), ~df_gene.columns.duplicated(keep='first')]
    gene = gene.apply(pd.to_numeric, errors='coerce')
    samples = gene.index.intersection(clin.index)
    if len(samples) == 0:
        raise ValueError("no overlapping samples between df_gene (index) and clin (index); "
                         "df_gene should have samples as rows and genes as columns")
    gene = gene.loc[samples]
    input_report = None
    if convert_ids:
        from TMEImmune import gene_id
        id_kind = gene_id.detect_id_type(gene.columns)['type']
        if id_kind != 'symbol' or gene_id.annotation_available():
            converted, id_report = gene_id.to_symbol(gene.T, aggregate='max', verbose=False)
            gene = converted.T
            if id_report.get('renamed') or id_kind != 'symbol':
                warnings.warn(f"ISAFN: gene identifiers detected as {id_kind}; {id_report['mapped']} of "
                              f"{id_report['n_input']} mapped to HGNC symbols", UserWarning)
    if log2 is None:
        # detect what the input is (counts, TPM, FPKM, log, z-scored) and convert it to log2(TPM+1),
        # which is what the models were trained on; z-scored data cannot be converted and are reported
        harmonized, input_report = to_log2tpm(gene.T, gene_length=gene_length, verbose=False, rescale=rescale,
                                              confirm=confirm)
        gene = harmonized.T
    elif log2:
        gene = np.log2(gene.clip(lower=0) + 1)
    clin_s = clin.loc[~clin.index.duplicated(keep='first')].loc[samples]

    mut, mut_summary = None, None
    if df_mut is not None:
        mut_in = df_mut.loc[~df_mut.index.duplicated(keep='first'), ~df_mut.columns.duplicated(keep='first')]
        mut_samples = samples.intersection(mut_in.index)
        if len(mut_samples) == 0:
            raise ValueError("no overlapping samples between df_mut (index) and df_gene/clin")
        if len(mut_samples) < len(samples):
            warnings.warn(f"ISAFN: {len(samples) - len(mut_samples)} samples have no mutation data; "
                          "their fusion scores are NaN", UserWarning)
        mut, mut_summary = _mutation_summary(mut_in.loc[mut_samples])

    # ---- encoded clinical covariates ----
    meta = pd.DataFrame(index=samples)
    meta['drug'] = _encode_drug(clin_s[drug_col]) if drug_col else DRUG_UNKNOWN
    meta['met1'] = _encode_met(clin_s[met_col]) if met_col else MET_UNKNOWN
    meta['source1'] = SOURCE_UNKNOWN
    meta['cancer_type'] = _encode_cancer(clin_s[cancer_col], cancer_key) if cancer_col \
        else CANCER_MAP[cancer_key]
    meta['trt1'] = TRT_PD1_PDL1
    meta['sex'] = _encode_sex(clin_s[gender_col]) if gender_col else SEX_UNKNOWN
    imputed = None
    if impute_sex:
        missing_sex = meta.index[meta['sex'] == SEX_UNKNOWN]
        if len(missing_sex) > 0:
            # use the whole cohort to separate the sexes, then fill only the missing ones
            imputed = _impute_sex(gene, log2=False).loc[missing_sex].dropna()
            meta.loc[imputed.index, 'sex'] = imputed.map(SEX_MAP).astype(int)
            warnings.warn(f"ISAFN: sex imputed from chrY/XIST expression for {len(imputed)} sample(s): "
                          f"{imputed.value_counts().to_dict()}", UserWarning)
    cohort_df = pd.DataFrame({'source': clin_s[source_col] if source_col else 'test'}, index=samples)

    selected = _load_json("features_output.json")
    needed = set(selected['selected_gene'])
    if sex_output:
        needed |= set(selected['selected_gene_male']) | set(selected['selected_gene_female'])
    missing = sorted(needed - set(gene.columns))
    if missing:
        warnings.warn(f"ISAFN: {len(missing)} of {len(needed)} model genes are missing from df_gene and are "
                      f"set to the training minimum (e.g. {missing[:5]}); predictions may be less reliable",
                      UserWarning)
    models = load_isafn_models()

    # (model name, samples, feature-list suffix, normalization range)
    groups = [('merged', samples, '', None)]
    if sex_output:
        groups += [('male', samples[(meta['sex'] == 0).values], '_male', 'male'),
                   ('female', samples[(meta['sex'] == 1).values], '_female', 'female')]

    preds = {}
    for name, idx, suf, gender in groups:
        if len(idx) == 0:
            continue
        if len(idx) < 5:
            warnings.warn(f"ISAFN: only {len(idx)} sample(s) in the {name} group; cohort-wise "
                          "normalization is unreliable for very small groups", UserWarning)
        gene_feats, mut_feats = selected['selected_gene' + suf], selected['selected_mut' + suf]
        gene_norm = isafn_norm(gene.loc[idx], cohort_df.loc[idx], cohort_col='source',
                               gender=gender, features=gene_feats)
        X_gene, X_mut = _build_inputs(gene_norm, mut, mut_summary, meta.loc[idx], gene_feats, mut_feats)
        res = {}
        m = models['gene'][name]
        res['isafn_expr_prob'], res['isafn_expr_pred'] = predict_fusionmodel(X_mut, X_gene, m[0], m[1], m[3], m[4])
        if mut is not None:
            has_mut = np.asarray(idx.isin(mut.index))
            m = models['gene+mut'][name]
            prob = np.full(len(idx), np.nan)
            pred = np.full(len(idx), np.nan)
            if has_mut.any():
                p, y = predict_fusionmodel(X_mut[has_mut], X_gene[has_mut], m[0], m[1], m[3], m[4])
                prob[has_mut], pred[has_mut] = p, y
            res['isafn_fusion_prob'], res['isafn_fusion_pred'] = prob, pred
        preds[name] = pd.DataFrame(res, index=idx)

    output_df = preds['merged'].astype(float)
    if sex_output:
        for name in ('male', 'female'):
            if name in preds:
                output_df.loc[preds[name].index, :] = preds[name][output_df.columns].values
    for c in output_df.columns:
        if c.endswith('_pred'):
            output_df[c] = output_df[c].astype('Int64')
    if imputed is not None:
        output_df.attrs['imputed_sex'] = imputed
    if input_report is not None:
        output_df.attrs['input_report'] = input_report
    return output_df
