from TMEImmune import estimateScore, netbio, SIAscore
from TMEImmune import ISTME as ISM
from TMEImmune import nb_utilities as nbu
from TMEImmune import data_processing
from TMEImmune import ISAFN
import pandas as pd
import warnings
import gseapy as gp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gmean

## geometric mean

CYT1 = ['GZMA', 'PRF1']
CYT2 = ['B2M', 'HLA-A', 'HLA-B', 'HLA-C', 'CASP8']
IFNr = ['CD3D', 'IDO1', 'CIITA', 'CD3E', 'CCL5', 'GZMK', 'CD2', 'HLA-DRA', 'CXCL3', 'IL2RG', 'NKG7', 'HLA-E', 'CXCR6', 'LAG3', 'TAGAP', 'CXCL10', 'STAT1', 'GZMB']
TLS = ['BCL6', 'CD86', 'CXCR4', 'LAMP3', 'SELL', 'CCR7', 'CXCL13', 'CCL21', 'CCL19']
TIS = ['CD276', 'HLA-DQA1', 'CD274', 'IDO1', 'HLA-DRB1', 'HLA-E', 'CMKLR1', 'PDCD1LG2', 'PSMB10', 'LAG3', 'CXCL9', 'STAT1', 'CD8A', 'CCL5', 'NKG7', 'TIGIT', 'CD27', 'CXCR6']

## average mean
TIP_hot = ['CXCL9', 'CXCL10', 'CXCL11', 'CXCR3', 'CD3','CD4','CD8A','CD8B', 'CD274', 'PDCD1', 'CXCE4', 'CCL5']
TIP_cold = ['CXCL1','CXCL2', 'CCL20']

## ratio
CS_polarity = ['SPP1', 'CXCL9']

## IMPRES
IMPRES = {'Gene1': ['PDCD1','CD27', 'CTLA4', 'CD40', 'CD86', 'CD28', 'CD80', 'CD274', 'CD86', 'CD40', 'CD86', 
                'CD40', 'CD28', 'CD40', 'TNFRSF14'],
            'Gene2': ['TNFSF4', 'PDCD1', 'TNFSF4', 'CD28', 'TNFSF4', 'CD86', 'TNFSF9', 'VSIR', 'HAVCR2', 'PDCD1',
                      'CD200', 'CD80', 'CD276', 'CD274', 'CD86']}

## ssGSEA
TGFb = {'TGFb':['SLC20A1', 'XIAP', 'TGFBR1', 'BMPR2','FKBP1A', 'SKIL']}

def get_score(df, method, clin = None, test_clinid = None):
    """
    Compute ESTIMATE, ISTME, NetBio, SIA scores for the input gene expression data.
    df: a pandas dataframe having gene symbol as the first column or row index
    method: TME scoring methods choosing from ESTIMATE, ISTME, NetBio, SIA. The method statement can be one method, or a list of multiple methods.
    Output: a pandas dataframe with sample ID as index, and columns are the computed scores
    """

    # test if input is a pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas dataframe")


    # set gene symbol as row index if not the index column
    first_col = df.iloc[:,0]
    has_letters = first_col.apply(lambda x: any(char.isalpha() for char in str(x)))
    if has_letters.all():
        df1 = df.copy()
        df1.index = first_col
        df1 = df1.iloc[:,1:]
    else:
        ind_letters = df.index.to_series().apply(lambda x: any(char.isalpha() for char in str(x)))
        if not ind_letters.any():
            raise ValueError("Index contains invalid gene name. Gene symbols must be in the first column or row index")
        else:
            df1 = df.copy()

    # convert non-float values to float
    all_floats = df1.map(lambda x: isinstance(x, float)).all().all()
    if not all_floats:
        try:
            df1 = df1.astype(float)
        except ValueError:
            raise ValueError("Input contains non-float convertible values.")    

    # remove name missing genes
    if df1.index.isna().any():
        df1 = df1[df1.index.notna()]
        warnings.warn("Input dataframe index contains NA values", UserWarning)

    # for duplicated gene index, keep the first one
    if df1.index.duplicated().any():
        df1 = df1[~df1.index.duplicated(keep='first')]
        warnings.warn("Input dataframe contains duplicated indices", UserWarning)

    score_df = pd.DataFrame()
    score_df.index = df1.columns

    if isinstance(method, str):
        method = [method]
    
    for score in method:
        if score == "ESTIMATE":
            score_est = estimateScore.ESTIMATEscore(df1)
            score_df = pd.concat([score_df, score_est], axis=1)
        elif score == "ISTME":
            score_istme = ISM.istmeScore(df1)
            score_df = pd.concat([score_df, score_istme], axis = 1)
        elif score == "NetBio":
            if any(arg is None for arg in [clin, test_clinid]):
                raise ValueError("NetBio score needs input for clinical dataset and response column id")
            df2 = df1.reset_index()
            score_nb = netbio.get_netbio(df2, clin, test_clinid)
            if isinstance(score_nb, pd.Series):
                score_nb = score_nb.reindex(score_df.index)          # align by sample, not by position
            else:
                score_nb = pd.Series(score_nb, name = "NetBio", index = score_df.index)
            score_nb.name = "NetBio"
            score_df = pd.concat([score_df, score_nb], axis = 1)
        elif score == "SIA":
            score_sia = SIAscore.sia_score(df1)
            score_df = pd.concat([score_df, score_sia], axis = 1)
        else:
            raise ValueError("Invalid scoring method")
    
    return score_df


def get_geomean_score(df, sig, name = None):
    missing = set(sig) - set(df.index)
    if not missing:
        sig_df = df.loc[sig]
    else:
        common_sig = list(set(sig) & set(df.index))
        sig_df = df.loc[common_sig]
        print(f"{name}: {missing} not in dataframe")
    
    gmeans = sig_df.apply(lambda row: gmean(row), axis=0)
    return gmeans


def get_avgmean_score(df, sig, name = None):
    missing = set(sig) - set(df.index)
    if not missing:
        sig_df = df.loc[sig]
    else:
        common_sig = list(set(sig) & set(df.index))
        sig_df = df.loc[common_sig]
        print(f"{name}: {missing} not in dataframe")
    
    avgmeans = sig_df.mean(axis = 0)
    return avgmeans

def get_ratio_score(df, sig, name = None):
    numerator = sig[1]
    denominator = sig[0]
    if denominator not in df.index:
        print(f"{name}: denominator does not exist")
        ratio_score = pd.Series(None, index=df.columns, dtype=int)
        #raise ValueError("denominator does not exist")
    elif numerator not in df.index:
        ratio_score = pd.Series(0, index=df.columns, dtype=int)
    else:
        ratio_score = df.loc[numerator]/df.loc[denominator]
    return ratio_score

def get_impres_score(df, sig, name = None):
    Gene1 = sig['Gene1']
    Gene2 = sig['Gene2']
    missing = set(Gene1 + Gene2) - set(df.index)
    scores = pd.Series(0, index=df.columns, dtype=int)
    if missing:
        print(f"{name}: these genes are missing: {missing}")
    pairs_in = 0
    for g1, g2 in zip(Gene1, Gene2):
        g1_in = g1 in df.index
        g2_in = g2 in df.index
        if g1_in and g2_in:
            scores += (df.loc[g1] > df.loc[g2]).astype(int)
            pairs_in += 1
        # elif g1_in and not g2_in:
        #     continue
        # elif g2_in and not g1_in:
        #     continue
        else:
            continue
    scores = scores * 15/pairs_in
    return scores

def get_ssgsea_score(df, sig):
    ss = gp.ssgsea(data=df,gene_sets=sig,min_size=1,outdir=None,verbose=True,sample_norm_method = "rank")
    score = ss.res2d[['Name', 'NES']]
    score.index = score['Name']
    score = score.drop(columns = 'Name')
    return score


def get_all_score(df, clin, response_col = "resp", source_col = None, df_mut = None,
                  gender_col = None, drug_col = None, met_col = None, cancer = 'melanoma', sex_output = True,
                  impute_sex = False, cancer_col = None, skip_failed = True):
    """
    Compute all TME / ICI-response scores for the input gene expression data:
    CYT1, CYT2, IFNr, TLS, TIS, TIP Hot, TIP Cold, CS Polarity, IMPRES, TGFb (ssGSEA),
    ISTME, ESTIMATE, SIA, NetBio and ISAFN.
    df: a pandas dataframe having gene symbol as the first column or row index, samples as columns
    clin: clinical dataframe with sample ID as index, containing the response column
    response_col: response column in clin (needed by NetBio)
    source_col: optional cohort column in clin; scores are min-max scaled within each cohort
    ISAFN expects log-scale expression.
    df_mut, gender_col, drug_col, met_col, cancer, cancer_col, sex_output, impute_sex: passed to
        ISAFN.isafn_score
    skip_failed: when a score cannot be computed from this input -- most often because the matrix is a
        targeted panel rather than a transcriptome -- return it as a column of NaN and carry on, instead
        of aborting every other score. The reasons are listed in output.attrs['failed_scores'], and
        output.attrs['gene_coverage'] records how much of a reference transcriptome the input covers.
        Set False to raise the original error instead
    Output: a pandas dataframe with sample ID as index, and columns are the computed scores
            (scores min-max scaled to [0, 1]; *_pred columns are left as 0/1 predictions)
    """
    first_col = df.iloc[:,0]
    has_letters = first_col.apply(lambda x: any(char.isalpha() for char in str(x)))
    if has_letters.all():
        df1 = df.copy()
        df1.index = first_col
        df1 = df1.iloc[:,1:]
    else:
        ind_letters = df.index.to_series().apply(lambda x: any(char.isalpha() for char in str(x)))
        if not ind_letters.any():
            raise ValueError("Index contains invalid gene name. Gene symbols must be in the first column or row index")
        else:
            df1 = df.copy()

    common_samples = df1.columns.intersection(clin.index)
    df1 = df1[common_samples]
    df1 = df1.loc[~df1.index.duplicated(keep='first')]
    failed = {}

    def _try(name, fn, columns = None):
        """Run one score; on failure record why and return NaN columns, so the rest still run."""
        try:
            return fn()
        except Exception as exc:
            if not skip_failed:
                raise
            failed[name] = f"{type(exc).__name__}: {exc}"
            cols = columns or [name]
            return pd.DataFrame(np.nan, index = df1.columns, columns = cols)

    simple = [('CYT1', lambda: get_geomean_score(df1, CYT1, "CYT1")),
              ('CYT2', lambda: get_geomean_score(df1, CYT2, "CYT2")),
              ('IFNr', lambda: get_geomean_score(df1, IFNr, "IFNr")),
              ('TLS', lambda: get_geomean_score(df1, TLS, "TLS")),
              ('TIS', lambda: get_geomean_score(df1, TIS, "TIS")),
              ('TIP Hot', lambda: get_avgmean_score(df1, TIP_hot, "TIP Hot")),
              ('TIP Cold', lambda: get_avgmean_score(df1, TIP_cold, "TIP Cold")),
              ('CS Polarity', lambda: get_ratio_score(df1, CS_polarity, "CS Polarity")),
              ('IMPRES', lambda: get_impres_score(df1, IMPRES, "IMPRES")),
              ('TGFb', lambda: get_ssgsea_score(df1, TGFb))]
    parts = []
    for name, fn in simple:
        block = _try(name, fn)
        block = block.to_frame() if isinstance(block, pd.Series) else block
        block.columns = [name]
        parts.append(block)
    score_df = pd.concat(parts, axis = 1)
    score_list = [score_df]

    # each of these is computed separately so that one unusable score does not take the others with it
    for name in ['ISTME', 'ESTIMATE', 'SIA', 'NetBio']:
        score_list.append(_try(name, lambda n = name: get_score(
            df1, method = n, clin = clin, test_clinid = response_col)))

    # ISAFN expects samples as rows and genes as columns
    isafn_cols = ['isafn_expr_prob', 'isafn_expr_pred'] + \
                 (['isafn_fusion_prob', 'isafn_fusion_pred'] if df_mut is not None else [])
    score_list.append(_try('ISAFN', lambda: ISAFN.isafn_score(
        df1.T, clin = clin, test_clinid = response_col, df_mut = df_mut,
        gender_col = gender_col, drug_col = drug_col, met_col = met_col,
        cancer = cancer, cancer_col = cancer_col, sex_output = sex_output,
        source_col = source_col, impute_sex = impute_sex), columns = isafn_cols))

    score_df = pd.concat(score_list, axis = 1).apply(pd.to_numeric, errors = 'coerce')
    if failed:
        warnings.warn("TMEImmune: these scores could not be computed from this input and are returned "
                      "as NaN: " + "; ".join(f"{k} ({v.split(':')[0]})" for k, v in failed.items()) +
                      ". See output.attrs['failed_scores'] for the full reasons.", UserWarning)

    # binary predictions are kept as they are; every other score is min-max scaled (per cohort)
    pred_cols = [c for c in score_df.columns if str(c).endswith('_pred')]
    num_df = score_df.drop(columns = pred_cols)

    def _scale(block):
        block = block.replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)
        block = block.fillna(block.median())
        return MinMaxScaler(feature_range=(0, 1)).fit_transform(block)

    score_df_scaled = num_df.copy()
    if source_col is not None:
        for cohort in clin.loc[num_df.index, source_col].unique():
            idx = clin.index[clin[source_col] == cohort].intersection(num_df.index)
            score_df_scaled.loc[idx] = _scale(num_df.loc[idx])
    else:
        score_df_scaled.loc[:, :] = _scale(num_df)
    score_df_scaled = pd.concat([score_df_scaled, score_df[pred_cols]], axis = 1)
    score_df_scaled.attrs['failed_scores'] = failed
    try:
        score_df_scaled.attrs['gene_coverage'] = data_processing.assess_gene_coverage(df1)
    except Exception:
        pass
    return score_df_scaled
