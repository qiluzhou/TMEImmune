import pandas as pd
import numpy as np
import os, json, re
#from cmapPy.pandasGEXpress.parse_gct import parse
from rnanorm import TMM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from inmoose.pycombat import pycombat_seq
import warnings
from collections import defaultdict
import importlib.resources as pkg_resources


_DATA_CACHE = {}
_CACHE_MAX_BYTES = 20 * 1024 ** 2      # files larger than this are re-read instead of cached


def clear_data_cache():
    """Empty the cache of packaged data files (gene sets, GMT, biomarker tables)."""
    _DATA_CACHE.clear()


def _data_file_size(path):
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.getsize(os.path.join(base, "data", path))
    except OSError:
        return 0


def load_data(path, cache = True):
    """ load data from the package's data folder

    Parsed files are cached in memory (the Reactome GMT and the gene set JSON take seconds to parse and
    are read on every score), so callers must not modify the returned object in place.
    cache = False re-reads the file; files above 20 MB are never cached.
    """
    if cache and path in _DATA_CACHE:
        return _DATA_CACHE[path]
    data = _load_data_uncached(path)
    if cache and _data_file_size(path) <= _CACHE_MAX_BYTES:
        _DATA_CACHE[path] = data
    return data


def _load_data_uncached(path):
    if path.endswith(".csv"):
        with pkg_resources.open_text("TMEImmune.data", path) as f:
            df = pd.read_csv(f)
        return df
        
    elif path.endswith(".json"):
        with pkg_resources.open_text("TMEImmune.data", path) as f:
        #with open(file_path, "r") as f:
            data = json.load(f)
        return data
        
    elif path.endswith(".gmt"):
        output = defaultdict(list)
        output_list = []
        #f = open(file_path,'r')
        with pkg_resources.open_text("TMEImmune.data", path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                if 'REACTOME' in line[0]:
                    reactome = line[0]
                    output_list.append(reactome)
                    #output[reactome].extend(line[2:])
                    for i in range(2, len(line)):
                        gene = line[i]
                        output[reactome].append(gene)
        #f.close()
        return output
       
    else: # path.endswith(".txt"):
        if "/" in path:
            subfolder_name, file_name = path.split("/")
            data_path = "TMEImmune.data." + subfolder_name
        else:
            data_path = "TMEImmune.data"
        with pkg_resources.open_text(data_path, file_name) as f:
            df = pd.read_table(f)
        return df

def read_gct(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # metadata
    num_rows, num_cols = map(int, lines[1].strip().split('\t')[0:2])
    # read data from the third lines
    df = pd.read_csv(file_path, sep='\t', skiprows=2, index_col=0)
    assert df.shape[0] == num_rows and df.shape[1] == num_cols, "Dimension mismatch"
    return df


def read_data(path):
    """
    Format: supports txt, csv, gct input
    Input: gene expression matrix with gene symbol as the first column, samples in columns and gene symbols in rows. 
           Duplicated or missing genes are not allowed.
    Output: a pandas dataframe where gene symbols are the row index and samples are columns
    """
    # gct, txt, csv, first column as genes
    _, file_extension = os.path.splitext(path)
    if file_extension == '.txt':
        df = pd.read_table(path, sep = "\t", index_col=0) 

    elif file_extension == '.csv':
        df = pd.read_csv(path, index_col=0)

    elif file_extension == '.gct':
        df = read_gct(path)
        #df = df.data_df

    else:
        raise TypeError(file_extension + " file is not supported")

    genes = df.index
    if genes.duplicated().any():
        dup_genes = genes[genes.duplicated()]
        warn = str("Duplicate genes: " + dup_genes.unique())
        warnings.warn(warn, category=UserWarning)
    
    if genes.isnull().any():
        warnings.warn("Exist NA's in the input genes", category=UserWarning)

    return (df)



# --------------------------------------------------------------------------------------
# input harmonization: detect what the expression matrix is, and convert it to log2(TPM+1)
# --------------------------------------------------------------------------------------
EXPRESSION_TYPES = {
    'counts': 'raw read counts',
    'tpm_or_cpm': 'per-million normalized (TPM or CPM: both sum to 1e6 per sample)',
    'fpkm_or_tpm': 'length/library normalized but not summing to 1e6 (FPKM, RPKM or rescaled TPM)',
    'log': 'already log-transformed',
    'zscore': 'z-scored (standardized)',
    'centered': 'centered or otherwise transformed to negative values',
    'unknown': 'could not be determined',
}


def detect_expression_type(df, axis = 'sample'):
    """
    Guess what kind of expression matrix this is.

    df: expression matrix, gene symbols as index and samples as columns
    axis: which axis z-scoring would have been applied to ('sample' = per column, 'gene' = per row);
          both are checked and reported anyway
    Output: dict with
        type: one of EXPRESSION_TYPES
        log: whether the values are on a log scale
        log_base: 'log2', 'ln' or None when the base cannot be told apart
        convertible: whether the data can be turned into log2(TPM+1)
        message: a sentence describing what was found
        stats: the numbers the decision was made from
    """
    values = df.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {'type': 'unknown', 'log': False, 'log_base': None, 'convertible': False,
                'message': 'the matrix has no numeric values', 'stats': {}}

    col_sums = np.nansum(np.where(np.isfinite(values), values, np.nan), axis=0)
    gene_mean, gene_sd = np.nanmean(values, axis=1), np.nanstd(values, axis=1)
    samp_mean, samp_sd = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    stats = {
        'min': float(finite.min()), 'max': float(finite.max()),
        'q99': float(np.percentile(finite, 99)), 'median_col_sum': float(np.median(col_sums)),
        'fraction_negative': float((finite < 0).mean()), 'fraction_zero': float((finite == 0).mean()),
        'fraction_integer': float(np.mean(np.isclose(finite, np.round(finite)))),
        'median_sample_mean': float(np.median(samp_mean)), 'median_sample_sd': float(np.median(samp_sd)),
        'median_gene_mean': float(np.median(gene_mean)), 'median_gene_sd': float(np.median(gene_sd)),
    }

    def result(kind, log, base, convertible, message):
        return {'type': kind, 'log': log, 'log_base': base, 'convertible': convertible,
                'message': message, 'stats': stats}

    # negative values: standardized or centered data, which cannot be converted back
    if stats['fraction_negative'] > 0.001:
        d_sample = abs(stats['median_sample_mean']) + abs(stats['median_sample_sd'] - 1)
        d_gene = abs(stats['median_gene_mean']) + abs(stats['median_gene_sd'] - 1)
        if min(d_sample, d_gene) < 0.35:
            if abs(d_sample - d_gene) < 0.1:
                where = 'genes and samples'
            else:
                where = 'samples' if d_sample < d_gene else 'genes'
            return result('zscore', False, None, False,
                          f"the data look z-scored ({where} have mean about 0 and sd about 1), "
                          "so the original expression scale cannot be recovered")
        return result('centered', False, None, False,
                      "the data contain negative values (centered, log-ratio or batch-corrected), "
                      "so the original expression scale cannot be recovered")

    # log scale: small range, not integer counts
    if stats['max'] < 30 and stats['median_col_sum'] < 1e5:
        base = 'log2' if stats['max'] > 13 else None
        note = "already log-transformed" if base == 'log2' else \
               "already log-transformed, but log2 and natural log cannot be told apart from the values"
        return result('log', True, base, True, note)

    # per-million normalized
    if 0.99e6 <= stats['median_col_sum'] <= 1.01e6:
        return result('tpm_or_cpm', False, None, True,
                      "samples sum to 1e6, so the data are TPM or CPM; treated as TPM")

    # raw counts
    if stats['fraction_integer'] > 0.99:
        return result('counts', False, None, True,
                      "the values are whole numbers, so the data are raw read counts")

    return result('fpkm_or_tpm', False, None, True,
                  "the values are continuous and positive but do not sum to 1e6 per sample: FPKM / RPKM, "
                  "or TPM of a subset of the genes. Samples are rescaled to sum to 1e6, which is the exact "
                  "FPKM -> TPM conversion; pass rescale=False to keep the values as they are")


# how much of a reference transcriptome an input has to cover before TPM is meaningful
PANEL_COVERAGE = 0.20
PANEL_MIN_GENES = 2000

# methods that rank or normalise across the whole transcriptome, so a targeted panel breaks them
TRANSCRIPTOME_METHODS = {
    'ESTIMATE': 'ranks every gene in the sample (ssGSEA); a panel changes every rank',
    'NetBio': 'ssGSEA over Reactome pathways, which needs transcriptome-wide ranks',
    'ISAFN': 'aligns expression to a stored 12,013-gene training range before scoring',
}


def reference_genes(reference = 'isafn'):
    """
    The gene universe an input is compared against by assess_gene_coverage.

    reference: 'isafn'      the 12,013 genes ISAFN aligns against (the training transcriptome)
               'annotation' every symbol in the packaged gene annotation that has a length
               or any iterable of gene symbols
    """
    if not isinstance(reference, str):
        return pd.Index(pd.unique(pd.Index(reference).astype(str)))
    if reference == 'isafn':
        import json
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data', 'isafn', 'cohort_minmax_global.json')
        with open(path) as fh:
            return pd.Index(list(json.load(fh)['global']['min']))
    if reference == 'annotation':
        from TMEImmune import gene_id
        return pd.Index(gene_id.gene_lengths().dropna().index)
    raise ValueError(f"unknown reference '{reference}'; use 'isafn', 'annotation' or a list of symbols")


def assess_gene_coverage(df, reference = 'isafn', min_coverage = PANEL_COVERAGE,
                         min_genes = PANEL_MIN_GENES):
    """
    How much of a reference transcriptome an expression matrix actually covers.

    A targeted panel and a whole transcriptome need different preprocessing: TPM, CPM and ssGSEA ranks
    are all defined relative to every transcript in the sample, so computing them from a few hundred
    genes silently produces numbers that look right and are not. This tells the two apart.

    df: expression matrix with gene symbols as index (samples as columns), or an iterable of symbols
    reference: see reference_genes
    Output: dict with n_input_genes, n_reference_genes, n_overlap, coverage (fraction of the reference
            present), likely_targeted_panel, and incompatible_methods -- the scores that cannot be
            computed honestly from this input.
    """
    genes = pd.Index(df.index if hasattr(df, 'index') else df).astype(str)
    genes = pd.Index(pd.unique(genes.str.strip()))
    try:
        ref = reference_genes(reference)
    except (FileNotFoundError, KeyError, ValueError):
        ref = pd.Index([])
    n_ref = len(ref)
    overlap = int(len(genes.intersection(ref))) if n_ref else 0
    coverage = overlap / n_ref if n_ref else float('nan')
    panel = bool(n_ref and (coverage < min_coverage or len(genes) < min_genes))
    return {'n_input_genes': int(len(genes)), 'n_reference_genes': int(n_ref),
            'n_overlap': overlap, 'coverage': coverage, 'reference': reference,
            'likely_targeted_panel': panel,
            'incompatible_methods': dict(TRANSCRIPTOME_METHODS) if panel else {}}


# transformations that change the shape of the data rather than just its units. Each is something a
# reasonable person might still want to do -- but not without being told.
RISKY_TRANSFORMATIONS = {
    'not_recoverable': 'the original expression scale cannot be recovered from these values',
    'no_gene_lengths': 'gene length is not corrected for, so genes are not comparable within a sample',
    'genes_dropped': 'genes without a known length are dropped from the matrix',
    'rescaled': 'every sample is rescaled to a fixed total, which changes each gene relative to it',
    'log_base_unknown': 'the log base is assumed, and the wrong assumption stretches the whole scale',
    'targeted_panel': 'a per-million total computed over a panel is not comparable to a real TPM',
}


_CONFIRM_DEFAULT = 'auto'


def set_confirm(value = 'auto'):
    """
    Set what every later call does when a conversion would change the shape of the data, so you do not
    have to pass confirm= to each one.

    value: True  convert without asking (use in a notebook or script once you know your input)
           False never convert where a concern was raised
           'auto' (the default) ask when somebody can answer; with nobody to ask, stop only on a
                  concern that would make the result meaningless
    Output: the previous setting, so it can be put back.

    An explicit confirm= on a call still wins over this.
    """
    global _CONFIRM_DEFAULT
    if value not in (True, False, 'auto'):
        raise ValueError("confirm must be True, False or 'auto'")
    previous, _CONFIRM_DEFAULT = _CONFIRM_DEFAULT, value
    return previous


def get_confirm():
    """The current session-wide confirmation setting (see set_confirm)."""
    return _CONFIRM_DEFAULT


def _can_prompt():
    """
    Whether there is anybody who could answer a question right now.

    A terminal can be asked. So can a Jupyter kernel, whose stdin is not a tty but whose input() still
    reaches the person. A script, a pipe or a scheduled run cannot, and asking there would print a
    prompt nobody sees before failing, so we do not even try.
    """
    import sys
    stdin = getattr(sys, 'stdin', None)
    if stdin is None or getattr(stdin, 'closed', False):
        return False
    try:
        if stdin.isatty():
            return True
    except (ValueError, OSError):
        return False
    try:                                              # a notebook: input() works without a tty
        ip = get_ipython()                            # noqa: F821  (only defined inside IPython)
        return bool(getattr(ip, 'kernel', None))
    except NameError:
        return False


def _ask_yes_no(question, default = False):
    """
    Ask the person at the keyboard a yes/no question.

    Returns True, False, or None when there is nobody to ask -- a script, a scheduled run, a pipe. The
    caller decides what to do in that case; it must never block.
    """
    if not _can_prompt():
        return None
    try:
        reply = input(question).strip().lower()
    except (EOFError, OSError, RuntimeError):
        return None
    except KeyboardInterrupt:
        return False
    if reply in ('y', 'yes'):
        return True
    if reply in ('n', 'no'):
        return False
    return default


def confirm_transformations(concerns, confirm = 'auto', verbose = True):
    """
    Put the questionable parts of a conversion to the user before doing them.

    concerns: list of dicts with 'code', 'detail' and 'consequence', as to_log2tpm builds them
    confirm: 'auto' asks when someone is there to answer. With nobody to ask it declines only when a
             concern is marked 'blocking' -- one that makes the result meaningless, such as z-scored
             input -- and otherwise proceeds with the warning, so existing scripts keep working. True
             proceeds without asking; False declines without asking
    Output: (proceed, how) -- how is 'no concerns', 'answered yes', 'answered no', 'assumed' or
            'nobody to ask'
    """
    if not concerns:
        return True, 'no concerns'
    if confirm is True:
        return True, 'assumed'
    if confirm is False:
        return False, 'assumed'

    # Whenever no answer can be had -- a script, a pipeline, a scheduled run, or a prompt the person
    # abandoned -- blocking a conversion would turn a warning into a silent no-op in code that has been
    # running for months. So only a concern that makes the result meaningless stops it, which is what
    # force= already did; everything else proceeds with its warning and is still put to whoever is there.
    blocking = [c for c in concerns if c.get('severity') == 'blocking']

    def _unanswered():
        if verbose:
            codes = ", ".join(c['code'] for c in concerns)
            if blocking:
                print(f"TMEImmune: the conversion was left undone ({codes}) because there is nobody to "
                      f"confirm it. Pass confirm=True to convert regardless, or fix the input.")
            else:
                print(f"TMEImmune: proceeding without confirmation ({codes}); there is nobody to ask. "
                      f"Pass confirm=False to stop instead.")
        return not blocking, 'nobody to ask'

    if not _can_prompt():
        return _unanswered()

    lines = ["", "TMEImmune: this conversion would change the data in ways that may not be appropriate:"]
    for i, c in enumerate(concerns, 1):
        lines.append(f"  {i}. {c['detail']}")
        lines.append(f"     -> {c['consequence']}")
    lines.append("")
    lines.append("Continuing produces numbers that look normal but may not mean what they usually mean.")
    print("\n".join(lines))

    answer = _ask_yes_no("Continue with the conversion anyway? [y/N]: ", default=False)
    if answer is None:                       # the prompt could not be answered after all
        return _unanswered()
    if verbose:
        print("TMEImmune: " + ("converting as described above." if answer else
                               "leaving the data unchanged."))
    return answer, 'answered yes' if answer else 'answered no'


def to_log2tpm(df, gene_length = None, detected = None, verbose = True, force = False, rescale = True,
               allow_tpm = 'auto', reference = 'isafn', confirm = None):
    """
    Convert an expression matrix to log2(TPM + 1), which is the scale the ISAFN models were trained on.

    df: expression matrix, gene symbols as index and samples as columns
    gene_length: optional Series of gene lengths in bases, indexed by gene symbol. Needed to turn raw
                 counts into TPM; without it counts are converted to CPM, which ignores gene length.
                 'auto' takes the lengths from the packaged gene annotation when it has them
    detected: result of detect_expression_type, computed here when not given
    force: convert even when the data look z-scored or centered (the result is not really TPM)
    rescale: for FPKM-like input, rescale every sample to sum to 1e6 (the FPKM -> TPM conversion). Set
             False when the matrix holds TPM for a subset of the genes, where the sums are below 1e6
             because genes are missing rather than because the unit differs
    allow_tpm: whether the sum-to-1e6 step (which assumes the matrix holds a whole transcriptome) may
               be applied. 'auto' checks the input with assess_gene_coverage and refuses on a targeted
               panel, leaving the values on their own scale; True forces it, False never rescales
    confirm: what to do when the conversion would change the shape of the data and not just its units
             -- z-scored input that cannot be inverted, counts with no gene lengths, genes dropped for
             want of a length, a per-sample rescale, or an assumed log base. 'auto' (default) describes
             the problem and asks for a yes or no; with nobody there to answer (a script, a scheduled
             run) it declines and leaves the matrix alone. True converts without asking, False declines
             without asking. force=True implies confirm=True. Left as None it follows the session-wide
             setting, which set_confirm() changes
    Output: (converted dataframe, report dict with the detection result, the steps applied, and the
            concerns raised -- report['concerns'] and report['confirmed'])
    """
    detected = detected or detect_expression_type(df)
    kind = detected['type']

    coverage = assess_gene_coverage(df, reference=reference)
    concerns = []
    if allow_tpm == 'auto':
        allow_tpm = not coverage['likely_targeted_panel']
        if not allow_tpm:
            concerns.append({'code': 'targeted_panel', 'severity': 'advisory',
                             'detail': f"this matrix holds {coverage['n_input_genes']} genes and covers "
                                       f"{coverage['coverage']:.1%} of the "
                                       f"{coverage['n_reference_genes']}-gene reference transcriptome, "
                                       f"so it looks like a targeted panel",
                             'consequence': f"TPM and CPM are defined over a whole transcriptome, so the "
                                            f"values are left on their own scale and "
                                            f"{', '.join(coverage['incompatible_methods'])} cannot be "
                                            f"computed honestly from this input"})
    if not allow_tpm:
        rescale = False
    if isinstance(gene_length, str) and gene_length == 'auto':
        gene_length = None
        if kind == 'counts':
            try:
                from TMEImmune import gene_id
                lengths = gene_id.gene_lengths()
                gene_length = lengths if lengths.notna().any() else None
            except (FileNotFoundError, KeyError):
                gene_length = None
    steps = []
    out = df.apply(pd.to_numeric, errors='coerce')

    # ---- work out what this conversion would do to the data before doing any of it ----
    if kind in ('zscore', 'centered'):
        concerns.append({'code': 'not_recoverable', 'severity': 'blocking',
                         'detail': detected['message'],
                         'consequence': 'the ISAFN models expect log2(TPM+1), so scores computed from '
                                        'these values may be unreliable; supply TPM, FPKM or raw counts '
                                        'if you can'})
    elif kind == 'counts':
        if gene_length is None:
            concerns.append({'code': 'no_gene_lengths', 'severity': 'advisory',
                             'detail': 'these are raw counts and no gene lengths are available, so they '
                                       'can only become CPM',
                             'consequence': 'CPM does not correct for gene length, so a long gene looks '
                                            'more expressed than a short one at the same abundance; pass '
                                            'gene_length= for a real TPM'})
        else:
            lengths_probe = pd.Series(gene_length).reindex(out.index).astype(float)
            n_missing = int((lengths_probe.isna() | (lengths_probe <= 0)).sum())
            if n_missing:
                concerns.append({'code': 'genes_dropped', 'severity': 'advisory',
                                 'detail': f'{n_missing} of {len(lengths_probe)} genes have no known '
                                           f'length',
                                 'consequence': 'those genes are dropped, so the matrix loses rows and '
                                                'every per-sample total is computed without them'})
    elif kind == 'log' and detected['log_base'] is None:
        concerns.append({'code': 'log_base_unknown', 'severity': 'advisory',
                         'detail': 'the values are log-transformed but the base cannot be told from them',
                         'consequence': 'log2 is assumed; if they are natural-log values every number is '
                                        'off by a factor of 1.44, so convert with df / log(2) first'})

    if kind not in ('log', 'counts', 'zscore', 'centered') and rescale:
        col_sum_probe = out.sum(axis=0)
        if not np.allclose(col_sum_probe.dropna(), 1e6, rtol=0.01):
            concerns.append({'code': 'rescaled', 'severity': 'advisory',
                             'detail': 'each sample is rescaled so its values sum to 1e6',
                             'consequence': 'if the matrix holds TPM for only some of the genes, this '
                                            'inflates every value by the proportion that is missing; '
                                            'pass rescale=False to keep the original scale'})

    if confirm is None:
        confirm = _CONFIRM_DEFAULT
    proceed, how = confirm_transformations(concerns, confirm=True if force else confirm, verbose=verbose)
    for c in concerns:
        warnings.warn(f"TMEImmune: {c['detail']} -- {c['consequence']}", UserWarning)
    if not proceed:
        return out, {'detected': detected, 'steps': ['left unchanged (conversion not confirmed)'],
                     'target': 'log2(TPM+1)', 'converted': False, 'gene_coverage': coverage,
                     'concerns': concerns, 'confirmed': how,
                     'tpm_conversion': 'declined'}

    if kind == 'log':
        steps.append('already log-transformed, left as it is')
    elif kind == 'counts':
        if gene_length is not None:
            lengths = pd.Series(gene_length).reindex(out.index).astype(float)
            keep = lengths.notna() & (lengths > 0)
            out = out.loc[keep]
            rate = out.div(lengths[keep] / 1000.0, axis=0)          # reads per kilobase
            if allow_tpm:
                out = rate.div(rate.sum(axis=0), axis=1) * 1e6
                steps.append(f'counts -> TPM using gene lengths ({int(keep.sum())} genes)')
            else:
                # sum-to-1e6 over a panel is meaningless, so stop at reads per kilobase per million
                # mapped reads of the panel itself and say so
                out = rate.div(rate.sum(axis=0), axis=1) * rate.sum(axis=0).mean()
                steps.append(f'counts -> length-normalised reads ({int(keep.sum())} genes); not '
                             f'rescaled to 1e6 because the input is a targeted panel')
        elif allow_tpm:
            out = out.div(out.sum(axis=0), axis=1) * 1e6
            steps.append('counts -> CPM (no gene lengths given, so length is not corrected for)')
        else:
            out = out.div(out.sum(axis=0), axis=1) * out.sum(axis=0).mean()
            steps.append('counts -> library-size-normalised reads; not rescaled to 1e6 because the '
                         'input is a targeted panel and no gene lengths were given')
        out = np.log2(out + 1)
        steps.append('log2(x+1)')
    else:                                                            # tpm_or_cpm, fpkm_or_tpm, unknown
        col_sum = out.sum(axis=0)
        if np.allclose(col_sum, 1e6, rtol=0.01):
            steps.append('already sums to 1e6 per sample')
        elif rescale:
            out = out.div(col_sum, axis=1) * 1e6
            steps.append('rescaled each sample to sum to 1e6 (TPM)')
        else:
            steps.append('left on its own scale (rescale=False)')
        out = np.log2(out.clip(lower=0) + 1)
        steps.append('log2(x+1)')

    report = {'detected': detected, 'steps': steps, 'target': 'log2(TPM+1)', 'converted': True,
              'gene_coverage': coverage, 'concerns': concerns, 'confirmed': how,
              'tpm_conversion': 'applied' if allow_tpm else
              'prohibited (targeted panel); original scale preserved'}
    if verbose:
        print(f"---------- input detected as {kind}: {detected['message']} ----------")
        for st in steps:
            print(f"---------- {st} ----------")
    return out, report


def harmonize(df = None, path = None, gene_length = 'auto', convert_ids = True, aggregate = None,
              verbose = True, force = False, rescale = True, mapping = None, allow_tpm = 'auto',
              confirm = None):
    """
    Read (optionally) an expression matrix, put its genes on HGNC symbols and convert it to log2(TPM + 1).

    df / path: the matrix itself, or a file read with read_data (genes as rows, samples as columns)
    gene_length: 'auto' (packaged annotation), a Series of lengths, or None
    convert_ids: map Ensembl / Entrez ids and outdated symbols onto current symbols (see gene_id)
    aggregate: how to merge rows landing on the same symbol; by default 'sum' for counts, TPM and FPKM
               and 'max' for log-scale data
    mapping: an identifier -> symbol mapping of your own, taking precedence over the packaged
             annotation (see gene_id.to_symbol); a dict, Series, two-column dataframe or csv path
    confirm: ask before applying a transformation that changes the shape of the data (see to_log2tpm);
             None follows the session-wide setting from set_confirm()
    Returns (dataframe, report). The report records what the identifiers and the values were detected as
    and every step applied, so the whole conversion can be quoted in a methods section.
    """
    if df is None:
        if path is None:
            raise ValueError("provide df or path")
        df = read_data(path)
    first_col = df.iloc[:, 0]
    has_letters = first_col.apply(lambda x: any(char.isalpha() for char in str(x)))
    if has_letters.all():                                            # gene ids in the first column
        df = df.set_index(df.columns[0])

    detected = detect_expression_type(df)
    id_report = None
    if convert_ids:
        from TMEImmune import gene_id
        id_kind = gene_id.detect_id_type(df.index)['type']
        if aggregate is None:
            aggregate = 'sum' if detected['type'] in ('counts', 'tpm_or_cpm', 'fpkm_or_tpm') else 'max'
        if id_kind != 'symbol' or gene_id.annotation_available() or mapping is not None:
            df, id_report = gene_id.to_symbol(df, aggregate=aggregate, verbose=verbose,
                                              mapping=mapping)

    out, report = to_log2tpm(df, gene_length=gene_length, detected=detected, verbose=verbose,
                             force=force, rescale=rescale, allow_tpm=allow_tpm, confirm=confirm)
    report['identifiers'] = id_report
    return out, report


def is_log2_transformed(df):
    """
    Guess whether a gene expression matrix is already log-transformed (heuristic used by GEO2R).
    Data are treated as NOT log-transformed when the 99th percentile is above 100, or when the
    value range is larger than 50 and the lower quartile is positive. Matrices containing negative
    values are treated as already transformed (e.g. log-scale or centered data).
    """
    values = df.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return True
    if (values < 0).any():
        return True
    qx = np.quantile(values, [0, 0.25, 0.5, 0.75, 0.99, 1.0])
    not_log = (qx[4] > 100) or ((qx[5] - qx[0] > 50) and (qx[1] > 0))
    return not not_log


def log2_transform(df, log2 = None, verbose = True):
    """
    Apply log2(x + 1) when the data are not log-transformed.
    log2: None to detect automatically, True to always transform, False to never transform.
    """
    if log2 is None:
        log2 = not is_log2_transformed(df)
    if log2:
        df = np.log2(df.clip(lower = 0) + 1)
        if verbose:
            print("---------- data has been log2(x+1) transformed ----------")
    return df


def normalization(df = None, path = None, method = None, minmax = False, zscore = False, batch = None, batch_col = None, log2 = None): 
    """
    Perform normalization and batch effect correction to the input. The input must be a pandas dataframe or a valid path.
    Method: choose method from TMM, CPM, median ratio for read count data. Default to None for normalized gene expression.
    minmax: perform min-max normalization, set to False by default
    zscore: perform z-score normalization, set to False by default
    batch: a pandas dataframe having batch information for the gene expression data to perform batch effect correction, with row index as sample ID
    batch_col: column name of the batch information, must be a string
    log2: whether to apply log2(x+1) after read count normalization / batch correction and before z-score or
          min-max normalization. None (default) detects it automatically: data that are not log-transformed
          are log2(x+1) transformed; True always transforms; False never transforms.
    """

    # read data from path if path exists
    if path is not None:
        df = read_data(path)
    else:
        if df is None:
            raise ValueError("Invalid input pandas dataframe and path")
        elif not isinstance(df, pd.DataFrame):
            raise TypeError("The input should be a pandas dataframe")

    # test whether the first column is genes
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

    # fill NA's with 0 count
    df1 = df1.fillna(0)

    # read count normalization
    if method is not None:

        if method == 'TMM':
            # calculate TMM
            norm = TMM().fit(df1.T)
            gene_norm = norm.transform(df1.T)
            gene_norm = pd.DataFrame(gene_norm).T
            gene_norm.columns = df1.columns
            gene_norm.index = df1.index
            print("---------- data has been TMM normalized ----------")
        elif method == 'CPM':
            column_sums = df1.sum(axis=0)
            # calculate CPM
            gene_norm = (df1 / column_sums) * 1e6
            gene_norm.index = df1.index
            print("---------- data has been CPM normalized ----------")
        elif method == 'median ratio':
            # compute geometric means of each gene
            geo_means = np.exp(np.log(df1.replace(0, np.nan)).mean(axis=1))
            ratios = df1.div(geo_means, axis=0)
            # calculate the median of the ratios for each sample
            medratio = ratios.median(axis=0)
            medratio1 = medratio.replace(0, 1e-06)
            gene_norm = df1.div(medratio1, axis=1)
            gene_norm.index = df1.index
            print("---------- data has been median ratio normalized ----------")
        else:
            raise TypeError("Normalization method not supported")
        
    df_norm = gene_norm if 'gene_norm' in locals() else df1
    # remove incorrectly formatted rows
    df_norm = df_norm[~pd.to_datetime(df_norm.index, errors='coerce').notna()]

    if batch is not None:
        if not isinstance(batch_col, str):
            raise TypeError("batch_col must be a string")
        # replace negative expression values to the absolute values
        df_norm = df_norm.abs()
        warnings.warn("Negative gene expression exists in the dataframe", category=UserWarning)

        # reorder batch by the order of the gene expression matrix columns
        common_ID = set(df_norm.columns) & set(batch.index)
        df_norm = df_norm[list(common_ID)]
        common_batch = batch.loc[list(common_ID),:]
        batch_ordered = common_batch.loc[df_norm.columns].reset_index()

        # remove batch with only one sample and raise warning
        unique_batch = batch_ordered[batch_col].value_counts()[batch_ordered[batch_col].value_counts() == 1].index
        
        if not unique_batch.empty:
            warnings.warn(f"The following batches with only one sample will be removed: {unique_batch}")
            sample_id = batch[batch[batch_col].isin(unique_batch)].index
            single_batch_colname = list(set(sample_id) & set(df_norm.columns))
            batch_ordered = batch_ordered[~batch_ordered[batch_col].isin(unique_batch)]
            df_norm = df_norm.drop(columns = single_batch_colname)

        df_norm = pycombat_seq(df_norm, batch_ordered[batch_col])
        print("---------- batch effect removed ----------")


    # log2(x+1) transformation if the data are not log-transformed yet
    df_norm = log2_transform(df_norm, log2 = log2)

    # z-score normalization
    if zscore:
        scalar = StandardScaler()
        df_norm = pd.DataFrame(scalar.fit_transform(df_norm.T).T,  # Transpose to scale across rows (genes)
                                  index=df_norm.index, columns=df_norm.columns)
        print("---------- data has been z-score normalized ----------")

    # min-max normalization
    if minmax:
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))  # Adjust range as needed
        df_norm = pd.DataFrame(min_max_scaler.fit_transform(df_norm.T).T,  # Transpose to scale across rows (genes)
                                        index=df_norm.index, columns=df_norm.columns)
        print("---------- data has been min-max normalized ----------")
            
    return df_norm


# ---------------------------------------------------------------------------------------------
# clinical harmonisation
# ---------------------------------------------------------------------------------------------

# column names that mean the same thing in different cohorts (matched case- and separator-insensitively)
CLINICAL_SYNONYMS = {
    'sample': ['sample_id', 'sampleid', 'sample', 'patient', 'patient_id', 'patientid', 'id',
               'tumor_sample_barcode', 'case_id', 'subject', 'subject_id', 'bcr_patient_barcode'],
    'response': ['response', 'resp', 'responder', 'responder_flag', 'bestoverallresponse',
                 'best_overall_response', 'recist', 'benefit', 'clinical_benefit', 'orr', 'bor'],
    'sex': ['sex', 'gender', 'sex_at_birth', 'reported_sex', 'patient_sex'],
    'met': ['metastatic', 'metastatic_disease', 'met_status', 'metastasis', 'met', 'm_stage',
            'metastasis_status', 'stage_m'],
    'drug': ['drug', 'treatment', 'therapy', 'agent', 'regimen', 'treatment_name', 'ici'],
    'cancer': ['cancer', 'cancer_type', 'cancertype', 'indication', 'tumor_type', 'disease',
               'histology', 'primary_diagnosis'],
    'age': ['age', 'patient_age', 'age_at_treatment', 'age_at_diagnosis', 'years'],
}

# response label -> 1 (responder) / 0 (non-responder). CR and PR count as response, SD and PD do not.
RESPONSE_MAP = {
    'cr': 1, 'pr': 1, 'complete response': 1, 'partial response': 1, 'crpr': 1,
    'sd': 0, 'pd': 0, 'stable disease': 0, 'progressive disease': 0, 'sdpd': 0, 'pdsd': 0,
    'r': 1, 'nr': 0, 'responder': 1, 'non-responder': 0, 'nonresponder': 0, 'non responder': 0,
    'response': 1, 'no response': 0, 'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0,
    'benefit': 1, 'no benefit': 0, 'dcb': 1, 'ndb': 0,
}

# free-text cancer type -> the codes ISAFN was trained with
CANCER_SYNONYMS = {
    'melanoma': 'melanoma', 'skcm': 'melanoma', 'cutaneous melanoma': 'melanoma',
    'metastatic melanoma': 'melanoma',
    'bladder': 'bladder', 'blca': 'bladder', 'urothelial': 'bladder',
    'urothelial carcinoma': 'bladder', 'bladder cancer': 'bladder', 'mibc': 'bladder',
    'ccrcc': 'ccrcc', 'kirc': 'ccrcc', 'renal': 'ccrcc', 'renal cell carcinoma': 'ccrcc',
    'clear cell renal cell carcinoma': 'ccrcc', 'rcc': 'ccrcc', 'kidney': 'ccrcc',
}

# drug names ISAFN knows; anything else stays as written and is encoded as 'unknown' downstream
DRUG_SYNONYMS = {
    'pembrolizumab': 'Pembrolizumab', 'keytruda': 'Pembrolizumab', 'mk-3475': 'Pembrolizumab',
    'nivolumab': 'Nivolumab', 'opdivo': 'Nivolumab', 'bms-936558': 'Nivolumab',
    'atezolizumab': 'Atezolizumab', 'tecentriq': 'Atezolizumab', 'mpdl3280a': 'Atezolizumab',
    'ipilimumab + pembrolizumab': 'Ipilimumab + Pembrolizumab',
    'ipilimumab + nivolumab': 'Ipilimumab + Nivolumab',
    'nivolumab + ipilimumab': 'Ipilimumab + Nivolumab',
    'ipi + nivo': 'Ipilimumab + Nivolumab', 'ipi+nivo': 'Ipilimumab + Nivolumab',
    'bevacizumab+atezolizumab': 'Bevacizumab+Atezolizumab',
    'bevacizumab + atezolizumab': 'Bevacizumab+Atezolizumab',
}

_SEX_SYNONYMS = {'m': 'Male', 'male': 'Male', '0': 'Male', 'man': 'Male',
                 'f': 'Female', 'female': 'Female', '1': 'Female', 'woman': 'Female'}

_MET_YES = {'1', '1.0', 'yes', 'y', 'true', 't', 'm1', 'met', 'metastatic', 'metastasis',
            'distant', 'stage iv', 'iv'}
_MET_NO = {'0', '0.0', 'no', 'n', 'false', 'f', 'm0', 'nonmetastatic', 'non-metastatic',
           'non metastatic', 'primary', 'local', 'localised', 'localized'}


def _norm_key(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).strip().lower())


def _find_column(columns, role, given = None):
    """Pick the column of `columns` holding `role`, or None. An explicit name always wins."""
    if given is not None:
        return given if given in columns else None
    wanted = [_norm_key(c) for c in CLINICAL_SYNONYMS[role]]
    by_key = {}
    for c in columns:
        by_key.setdefault(_norm_key(c), c)
    for w in wanted:                                   # exact match first, in synonym order
        if w in by_key:
            return by_key[w]
    for w in wanted:                                   # then a containing match, longest name last
        for k, c in by_key.items():
            if w in k:
                return c
    return None


def harmonize_clinical(clin, sample_col = None, response_col = None, sex_col = None, met_col = None,
                       drug_col = None, cancer_col = None, age_col = None, cohort = None,
                       verbose = True):
    """
    Put a cohort's clinical table on the column names and encodings the rest of the package expects.

    Cohorts name and encode the same variables differently -- sample_id / SampleID / patient,
    Responder / CR-PR-SD-PD / 1-0, Male / M, Yes / 1 / metastatic. This finds each variable by name
    and rewrites its values, without touching the columns it does not recognise.

    clin: the cohort's clinical table (a dataframe, or a path read with read_data)
    *_col: give a column name explicitly when the automatic match is wrong or missing
    cohort: cohort label written into a 'source' column, for provenance after cohorts are merged
    Output: (harmonised dataframe indexed by sample, report dict recording every column and value
             mapping applied, plus anything that could not be mapped).

    The harmonised columns are: resp (1 responder / 0 non-responder, CR and PR count as responders),
    sex ('Male' / 'Female'), met1 (1 / 0), drug (ISAFN's drug names), cancer_type (ISAFN's codes:
    melanoma, bladder, ccrcc, unknown), age, and source. Original columns are kept as they were.
    """
    if not isinstance(clin, pd.DataFrame):
        clin = read_data(clin)
    out = clin.copy()
    cols = list(out.columns)
    report = {'n_samples': len(out), 'columns': {}, 'values': {}, 'unmapped': {}}

    # ---- sample identifier becomes the index ----
    scol = _find_column(cols, 'sample', sample_col)
    if scol is None and out.index.name is not None:
        scol = out.index.name
    elif scol is not None:
        out = out.set_index(scol)
    out.index = pd.Index([str(i).strip() for i in out.index], name='sample')
    report['columns']['sample'] = scol
    if out.index.duplicated().any():
        dup = int(out.index.duplicated().sum())
        warnings.warn(f"TMEImmune: {dup} duplicate sample identifiers in the clinical table", UserWarning)

    remaining = list(out.columns)

    def _map_column(role, given, target, mapper):
        col = _find_column(remaining, role, given)
        report['columns'][role] = col
        if col is None:
            return
        raw = out[col]
        new = mapper(raw)
        bad = sorted({str(v).strip() for v, n in zip(raw, new) if pd.notna(v) and pd.isna(n)})
        if bad:
            report['unmapped'][role] = bad
            warnings.warn(f"TMEImmune: {role} value(s) {bad[:6]} in column '{col}' were not recognised "
                          f"and are left as missing", UserWarning)
        out[target] = new
        seen = {}
        for v, n in zip(raw, new):
            if pd.notna(v):
                seen[str(v).strip()] = None if pd.isna(n) else n
        report['values'][role] = seen

    _map_column('response', response_col, 'resp',
                lambda s: s.map(lambda x: RESPONSE_MAP.get(str(x).strip().lower(), np.nan)
                                if pd.notna(x) else np.nan).astype(float))
    _map_column('sex', sex_col, 'sex',
                lambda s: s.map(lambda x: _SEX_SYNONYMS.get(str(x).strip().lower(), np.nan)
                                if pd.notna(x) else np.nan))
    _map_column('met', met_col, 'met1',
                lambda s: s.map(lambda x: (1.0 if str(x).strip().lower() in _MET_YES else
                                           0.0 if str(x).strip().lower() in _MET_NO else np.nan)
                                if pd.notna(x) else np.nan).astype(float))
    _map_column('cancer', cancer_col, 'cancer_type',
                lambda s: s.map(lambda x: CANCER_SYNONYMS.get(str(x).strip().lower(), np.nan)
                                if pd.notna(x) else np.nan))
    # an unrecognised drug is kept verbatim: ISAFN encodes it as 'unknown' and warns there
    dcol = _find_column(remaining, 'drug', drug_col)
    report['columns']['drug'] = dcol
    if dcol is not None:
        out['drug'] = out[dcol].map(
            lambda x: DRUG_SYNONYMS.get(str(x).strip().lower(), str(x).strip())
            if pd.notna(x) else np.nan)
        report['values']['drug'] = {str(v).strip(): DRUG_SYNONYMS.get(str(v).strip().lower(),
                                                                      str(v).strip())
                                    for v in out[dcol].dropna().unique()}
        known = set(DRUG_SYNONYMS.values())
        unknown = sorted({v for v in report['values']['drug'].values() if v not in known})
        if unknown:
            report['unmapped']['drug'] = unknown
    acol = _find_column(remaining, 'age', age_col)
    report['columns']['age'] = acol
    if acol is not None:
        out['age'] = pd.to_numeric(out[acol], errors='coerce')

    if cohort is not None:
        out['source'] = str(cohort)
        report['cohort'] = str(cohort)

    if verbose:
        found = {k: v for k, v in report['columns'].items() if v is not None}
        print(f"---------- clinical: {len(out)} samples; matched {found} ----------")
        if 'response' in report['values']:
            print(f"---------- response encoded as {report['values']['response']} ----------")
        for role, bad in report['unmapped'].items():
            print(f"---------- {role}: not recognised -> {bad[:6]} ----------")
    return out, report


# ---------------------------------------------------------------------------------------------
# cross-cohort merging and batch correction
# ---------------------------------------------------------------------------------------------

BATCH_METHODS = ('none', 'combat', 'minmax_global', 'minmax_cohort', 'zscore_cohort')


def combat(df, batch, covariates = None, method = 'auto', verbose = True):
    """
    Remove cohort / batch effects with ComBat (inmoose).

    df: expression matrix, genes as rows and samples as columns
    batch: batch label per sample -- a Series indexed like df.columns, or a list in column order
    covariates: optional dataframe of biological variables to protect (e.g. response, sex), indexed
                like df.columns. ComBat keeps the variance it explains instead of removing it
    method: 'seq' (pycombat_seq, the negative-binomial model for raw integer counts), 'norm'
            (pycombat_norm, for continuous data such as log2(TPM+1)), or 'auto' to pick by input
    Output: (corrected dataframe with the same shape and labels, report dict)

    Genes with missing values are left uncorrected, since ComBat cannot fit them; they are listed in
    the report. Correcting a matrix that a model later min-max aligns per cohort will partly undo the
    correction, so use one or the other rather than both.
    """
    batch = pd.Series(batch, index=df.columns) if not isinstance(batch, pd.Series) else \
        batch.reindex(df.columns)
    if batch.isna().any():
        raise ValueError("every sample needs a batch label")
    if batch.nunique() < 2:
        if verbose:
            print("---------- only one batch; nothing to correct ----------")
        return df.copy(), {'method': 'none', 'reason': 'a single batch', 'n_genes_corrected': 0}

    values = df.apply(pd.to_numeric, errors='coerce')
    complete = values.notna().all(axis=1) & (values.std(axis=1) > 0)
    skipped = list(values.index[~complete])
    if method == 'auto':
        v = values[complete].to_numpy(dtype=float)
        integral = v.size > 0 and np.all(v >= 0) and np.allclose(v, np.round(v))
        method = 'seq' if integral else 'norm'

    if method == 'seq':
        from inmoose.pycombat import pycombat_seq as _fn
    elif method == 'norm':
        try:
            from inmoose.pycombat import pycombat_norm as _fn
        except ImportError:                       # older inmoose exposed it under the previous name
            from inmoose.pycombat import pycombat as _fn
    else:
        raise ValueError(f"method must be 'seq', 'norm' or 'auto', not '{method}'")

    covar = None
    if covariates is not None:
        covar = pd.DataFrame(covariates).reindex(df.columns)
        covar = covar.loc[:, covar.nunique(dropna=False) > 1]
        if covar.shape[1] == 0:
            covar = None

    corrected = _fn(values[complete], batch.to_numpy(), covar_mod=covar)
    corrected = pd.DataFrame(np.asarray(corrected), index=values.index[complete], columns=df.columns)
    out = values.copy()
    out.loc[complete] = corrected

    report = {'method': f'combat_{method}', 'n_batches': int(batch.nunique()),
              'n_genes_corrected': int(complete.sum()), 'n_genes_skipped': len(skipped),
              'skipped_genes': skipped[:50],
              'covariates': list(covar.columns) if covar is not None else []}
    if verbose:
        print(f"---------- ComBat ({method}): {report['n_genes_corrected']} genes corrected across "
              f"{report['n_batches']} batches, {report['n_genes_skipped']} skipped "
              f"(missing or constant) ----------")
    if skipped and verbose:
        print(f"---------- skipped: {skipped[:8]}{' ...' if len(skipped) > 8 else ''} ----------")
    return out, report


def correct_batch(df, batch, method = 'combat', covariates = None, verbose = True):
    """
    Apply one cross-cohort correction to a genes x samples matrix, chosen by name.

    method: 'none'            leave the data alone (cohort effects stay in)
            'combat'          empirical-Bayes correction (see combat)
            'minmax_global'   scale every cohort onto the overall min/max of each gene, which is the
                              alignment ISAFN applies internally
            'minmax_cohort'   scale every cohort's genes to [0, 1] within that cohort
            'zscore_cohort'   standardise every gene within each cohort
    Output: (corrected dataframe, report dict)
    """
    if method not in BATCH_METHODS:
        raise ValueError(f"method must be one of {BATCH_METHODS}, not '{method}'")
    batch = pd.Series(batch, index=df.columns) if not isinstance(batch, pd.Series) else \
        batch.reindex(df.columns)
    if method == 'none':
        return df.copy(), {'method': 'none', 'n_batches': int(batch.nunique())}
    if method == 'combat':
        return combat(df, batch, covariates=covariates, verbose=verbose)

    values = df.apply(pd.to_numeric, errors='coerce')
    out = values.copy()
    g_min, g_max = values.min(axis=1), values.max(axis=1)
    g_range = (g_max - g_min).replace(0, 1)
    for label in batch.dropna().unique():
        cols = batch.index[batch == label]
        block = values[cols]
        if method == 'zscore_cohort':
            sd = block.std(axis=1).replace(0, 1)
            out[cols] = block.sub(block.mean(axis=1), axis=0).div(sd, axis=0)
        else:
            c_min, c_max = block.min(axis=1), block.max(axis=1)
            c_range = (c_max - c_min).replace(0, 1)
            scaled = block.sub(c_min, axis=0).div(c_range, axis=0)          # -> [0, 1] within cohort
            out[cols] = scaled if method == 'minmax_cohort' else \
                scaled.mul(g_range, axis=0).add(g_min, axis=0)              # -> the overall range
    report = {'method': method, 'n_batches': int(batch.nunique()), 'n_genes_corrected': len(out)}
    if verbose:
        print(f"---------- {method}: {len(out)} genes aligned across {report['n_batches']} "
              f"cohorts ----------")
    return out, report


def merge_cohorts(expression, clinical = None, mutation = None, how = 'union', mapping = None,
                  gene_length = 'auto', batch_method = 'none', protect = ('resp',), verbose = True,
                  confirm = None):
    """
    Harmonise several cohorts and stack them into one matrix, keeping track of where each sample came from.

    Every cohort is harmonised on its own first -- identifiers to HGNC symbols, values to log2(TPM+1)
    where the gene coverage allows it -- because the unit and the identifier scheme differ per cohort.
    Only then are the cohorts stacked, so nothing is converted using another cohort's scale.

    expression: {cohort name: expression matrix or file path}, genes as rows and samples as columns
    clinical: {cohort name: clinical table or path}, harmonised with harmonize_clinical
    mutation: {cohort name: long-format mutation table or path}; cohorts may be missing from this dict
    how: 'union' keeps every gene seen in any cohort, leaving NaN where a cohort did not measure it;
         'intersection' keeps only genes present in all cohorts
    mapping: identifier -> symbol table for identifiers the packaged annotation does not know; either
             one table for all cohorts or {cohort name: table}
    batch_method: cross-cohort correction applied after stacking (see correct_batch)
    confirm: ask before a transformation that changes the shape of a cohort's data (see to_log2tpm);
             each cohort is asked about separately, naming that cohort
    protect: clinical columns whose variance ComBat should preserve
    Output: (expression, clinical, mutation, report). Expression is genes x samples; clinical is
            indexed by sample with a 'source' column naming the cohort; mutation is samples x genes,
            or None when no cohort supplied any. The report records, per cohort, what was detected and
            applied, and a per-gene coverage table.
    """
    from TMEImmune import ISAFN
    names = list(expression)
    report = {'cohorts': {}, 'order': names, 'how': how}
    expr_parts, clin_parts, mut_parts, gene_sets = {}, {}, {}, {}

    for name in names:
        mp = mapping.get(name) if isinstance(mapping, dict) and name in mapping else mapping
        df = expression[name]
        df = read_data(df) if isinstance(df, str) else df.copy()
        ex, ex_rep = harmonize(df=df, mapping=mp, gene_length=gene_length, verbose=False,
                               confirm=confirm)
        if ex.columns.duplicated().any():
            ex = ex.loc[:, ~ex.columns.duplicated(keep='first')]
        expr_parts[name] = ex
        gene_sets[name] = set(ex.index)
        entry = {'expression': ex_rep, 'n_samples': ex.shape[1], 'n_genes': ex.shape[0]}

        if clinical is not None and name in clinical:
            cl, cl_rep = harmonize_clinical(clinical[name], cohort=name, verbose=False)
            clin_parts[name] = cl
            entry['clinical'] = cl_rep
            shared = ex.columns.intersection(cl.index)
            entry['samples_with_clinical'] = int(len(shared))
            if len(shared) == 0:
                warnings.warn(f"TMEImmune: cohort '{name}' has no sample in common between its "
                              f"expression columns and its clinical index", UserWarning)

        if mutation is not None and name in mutation and mutation[name] is not None:
            mu = mutation[name]
            mu = read_data(mu) if isinstance(mu, str) else mu
            mu = mu.reset_index() if mu.index.name else mu
            scol, gcol = mu.columns[0], mu.columns[1]
            mat = ISAFN.mutation_to_matrix(mu, sample_col=scol, gene_col=gcol)
            if mp is not None or gene_id_needed(mat.columns):
                conv, id_rep = _mutation_ids_to_symbol(mat, mp)
                mat = conv
                entry['mutation_identifiers'] = id_rep
            mut_parts[name] = mat
            entry['mutation'] = {'n_samples': mat.shape[0], 'n_genes': mat.shape[1],
                                 'sample_col': scol, 'gene_col': gcol}
        report['cohorts'][name] = entry

    # ---- stack ----
    if how == 'intersection':
        genes = [g for g in expr_parts[names[0]].index if all(g in gene_sets[n] for n in names)]
    elif how == 'union':
        genes = list(dict.fromkeys([g for n in names for g in expr_parts[n].index]))
    else:
        raise ValueError("how must be 'union' or 'intersection'")

    expr = pd.concat([expr_parts[n].reindex(index=genes) for n in names], axis=1)
    coverage = pd.DataFrame({n: [g in gene_sets[n] for g in genes] for n in names}, index=genes)
    coverage['n_cohorts'] = coverage[names].sum(axis=1)
    report['gene_coverage_table'] = coverage
    report['n_genes'] = len(genes)
    report['genes_missing_somewhere'] = list(coverage.index[coverage['n_cohorts'] < len(names)])
    report['n_missing_values'] = int(expr.isna().sum().sum())

    source = pd.Series({s: n for n in names for s in expr_parts[n].columns}).reindex(expr.columns)
    clin = None
    if clin_parts:
        clin = pd.concat([clin_parts[n] for n in names if n in clin_parts], axis=0)
        clin = clin.reindex(expr.columns)
        clin['source'] = source
    mut = None
    if mut_parts:
        mut = pd.concat([mut_parts[n] for n in names if n in mut_parts], axis=0).fillna(0)
        mut = mut.reindex(index=expr.columns).fillna(0)
        # a cohort that supplied no mutation file gets all-zero rows, which the models cannot tell
        # apart from "sequenced, no mutation found" -- so say exactly which samples those are
        no_data = [n for n in names if n not in mut_parts]
        without = [s for n in no_data for s in expr_parts[n].columns]
        report['mutation_cohorts'] = [n for n in names if n in mut_parts]
        report['cohorts_without_mutation_data'] = no_data
        report['n_samples_without_mutation_data'] = len(without)
        if no_data:
            warnings.warn(
                f"TMEImmune: cohort(s) {no_data} supplied no mutation data, so all {len(without)} of "
                f"their samples carry an all-zero mutation profile, which a model cannot distinguish "
                f"from a sample that was sequenced and had no mutation. Either drop these cohorts from "
                f"the fusion model or score them with the expression model only.", UserWarning)

    # ---- optional cross-cohort correction ----
    covar = None
    if clin is not None and protect:
        keep = [c for c in protect if c in clin.columns]
        covar = clin[keep] if keep else None
    expr, batch_report = correct_batch(expr, source, method=batch_method, covariates=covar,
                                       verbose=verbose)
    report['batch_correction'] = batch_report

    if verbose:
        per = ', '.join(f"{n}: {expr_parts[n].shape[1]}x{expr_parts[n].shape[0]}" for n in names)
        print(f"---------- merged {len(names)} cohorts ({per}) -> {expr.shape[0]} genes x "
              f"{expr.shape[1]} samples ----------")
        print(f"---------- {how}: {len(report['genes_missing_somewhere'])} genes missing from at "
              f"least one cohort, {report['n_missing_values']} missing values ----------")
    return expr, clin, mut, report


def gene_id_needed(ids):
    """True when these gene identifiers are not already symbols."""
    from TMEImmune import gene_id
    return gene_id.detect_id_type(ids)['type'] != 'symbol'


def _mutation_ids_to_symbol(mat, mapping):
    """Convert the gene columns of a samples x genes mutation matrix to symbols."""
    from TMEImmune import gene_id
    conv, rep = gene_id.to_symbol(mat.T, mapping=mapping, aggregate='sum', verbose=False)
    return conv.T, rep
