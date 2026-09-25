"""
Gene identifier harmonisation: map Ensembl gene ids, Entrez ids, RefSeq ids or outdated symbols onto
current HGNC symbols, and look up gene lengths for the counts -> TPM conversion.

Everything here reads one packaged table, data/gene_annotation.npz, built by
tools/build_gene_annotation.py from a GENCODE annotation (and, optionally, the HGNC complete set).
Without that file the detection still works, but conversion raises a clear error.

    from TMEImmune import gene_id
    gene_id.detect_id_type(df.index)              # what kind of identifiers are these?
    df2, report = gene_id.to_symbol(df)           # index -> HGNC symbols
    lengths = gene_id.gene_lengths()              # symbol -> union exon length, for counts -> TPM
"""
import os
import re
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

_ANNOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gene_annotation.npz")

_ENSEMBL_RE = re.compile(r'^ENS[A-Z]*G\d+(\.\d+)?$', re.I)
_REFSEQ_RE = re.compile(r'^[NX][MRP]_\d+(\.\d+)?$', re.I)
_ENTREZ_RE = re.compile(r'^\d+$')


def annotation_available():
    """Whether the packaged gene annotation table is present."""
    return os.path.exists(_ANNOT_PATH)


@lru_cache(maxsize=1)
def load_annotation():
    """
    Load the packaged annotation table (cached).

    Output: dict with
        genes: dataframe with symbol, ensembl, entrez, length (union of exons, in bases)
        alias: {old or alias symbol: current symbol}
        source: how and when the table was built
    """
    if not annotation_available():
        raise FileNotFoundError(
            "TMEImmune: data/gene_annotation.npz is missing. Build it once with\n"
            "    python tools/build_gene_annotation.py --gtf gencode.vXX.basic.annotation.gtf.gz\n"
            "                                          [--hgnc hgnc_complete_set.txt]\n"
            "see the docstring of that script for the download links.")
    with np.load(_ANNOT_PATH, allow_pickle=False) as z:
        genes = pd.DataFrame({'symbol': z['symbol'], 'ensembl': z['ensembl'],
                              'entrez': z['entrez'], 'length': z['length']})
        alias = dict(zip(z['alias_from'], z['alias_to'])) if 'alias_from' in z else {}
        source = str(z['source'][0]) if 'source' in z else 'unknown'
    return {'genes': genes, 'alias': alias, 'source': source}


def _clean(ids):
    return pd.Index([str(i).strip() for i in ids])


def detect_id_type(ids, sample = 2000):
    """
    Guess what kind of gene identifiers these are.

    ids: the index of an expression matrix (or any iterable of identifiers)
    Output: dict with type ('symbol', 'ensembl', 'entrez', 'refseq', 'mixed' or 'unknown'),
            the fraction of identifiers of each kind, and a few examples.
    """
    ids = _clean(ids)
    probe = ids[:sample] if len(ids) > sample else ids
    n = max(len(probe), 1)
    frac = {
        'ensembl': float(np.mean([bool(_ENSEMBL_RE.match(i)) for i in probe])),
        'entrez': float(np.mean([bool(_ENTREZ_RE.match(i)) for i in probe])),
        'refseq': float(np.mean([bool(_REFSEQ_RE.match(i)) for i in probe])),
    }
    frac['symbol'] = float(np.mean([bool(re.match(r'^[A-Za-z][A-Za-z0-9\-\.@_]*$', i)) and
                                    not _ENSEMBL_RE.match(i) and not _REFSEQ_RE.match(i) for i in probe]))
    best = max(frac, key=frac.get)
    kind = best if frac[best] >= 0.7 else ('mixed' if sum(v > 0.2 for v in frac.values()) > 1 else 'unknown')
    return {'type': kind, 'fractions': frac, 'examples': list(probe[:5])}


def read_mapping(mapping):
    """
    Normalise a user-supplied identifier -> symbol mapping into a Series.

    mapping: a dict, a Series (identifier index, symbol values), a two-column dataframe, or the path
             of a two-column csv. Either column order is accepted: whichever column looks more like
             HGNC symbols becomes the target, so both "symbol,ensembl_id" and "ensembl_id,symbol"
             files work.
    """
    if isinstance(mapping, pd.Series):
        out = mapping
    elif isinstance(mapping, dict):
        out = pd.Series(mapping)
    else:
        tab = mapping if isinstance(mapping, pd.DataFrame) else pd.read_csv(mapping, dtype=str)
        if tab.shape[1] < 2:
            raise ValueError("a mapping table needs at least two columns (identifier and symbol)")
        a, b = tab.columns[0], tab.columns[1]
        # the symbol column is the one whose values look least like Ensembl/Entrez/RefSeq accessions
        sym_a = detect_id_type(tab[a].dropna().astype(str))['type'] == 'symbol'
        sym_b = detect_id_type(tab[b].dropna().astype(str))['type'] == 'symbol'
        key, val = (b, a) if (sym_a and not sym_b) else (a, b)
        out = tab.dropna(subset=[key]).set_index(key)[val]
    out.index = pd.Index([str(i).strip() for i in out.index])
    out = out.dropna().astype(str).str.strip()
    return out[~out.index.duplicated(keep='first')]


def to_symbol(df, id_type = 'auto', aggregate = 'max', keep_unmapped = None, verbose = True,
              mapping = None):
    """
    Convert the index of an expression matrix to current HGNC symbols.

    df: expression matrix, gene identifiers as index and samples as columns
    id_type: 'auto' (default), 'symbol', 'ensembl', 'entrez' or 'refseq'
    aggregate: how to combine rows that end up on the same symbol -- 'max' (the row with the highest
               mean, safest for log-scale data), 'sum' (right for counts, TPM and FPKM), 'mean',
               or 'first'
    keep_unmapped: keep identifiers that could not be mapped, under their original name. The default
                   keeps them when the input is already symbols (a symbol missing from the annotation
                   table is usually just newer than the table, not invalid) and drops them otherwise
    mapping: an identifier -> symbol mapping of your own, which takes precedence over the packaged
             annotation table. A dict, a Series, a two-column dataframe, or the path of a two-column
             csv. Use it for identifiers the packaged table does not know, such as an in-house or
             synthetic ID scheme, or an assembly older than the packaged annotation. Identifiers the
             mapping does not cover still fall back to the packaged table
    Output: (converted dataframe, report dict with counts of what was mapped, renamed and dropped)

    Ensembl ids keep only the part before the version suffix. Symbols that are outdated or aliases are
    renamed to the current symbol when the HGNC table was included in the annotation.
    """
    ids = _clean(df.index)
    detected = detect_id_type(ids)
    kind = detected['type'] if id_type == 'auto' else id_type

    user_map = read_mapping(mapping) if mapping is not None else None
    if user_map is not None and keep_unmapped is None and not annotation_available():
        keep_unmapped = False
    if keep_unmapped is None:
        keep_unmapped = (kind == 'symbol')

    out = df.copy()
    out.index = ids
    report = {'detected': detected, 'id_type': kind, 'n_input': len(ids)}

    if user_map is not None and not annotation_available():
        # a user mapping is enough on its own; the packaged table is only consulted when it exists
        new = user_map.reindex(ids)
        report['from_user_mapping'] = int(new.notna().sum())
        return _relabel(out, ids, new, keep_unmapped, aggregate, report, kind, verbose)

    if kind == 'symbol' and not annotation_available():
        report.update({'mapped': len(ids), 'renamed': 0, 'unmapped': 0,
                       'note': 'identifiers already look like symbols; no annotation table needed'})
        if verbose:
            print(f"---------- identifiers detected as gene symbols ({len(ids)} genes) ----------")
        return out, report

    ann = load_annotation()
    genes = ann['genes']

    if kind == 'ensembl':
        key = pd.Index([i.split('.')[0].upper() for i in ids])
        lookup = genes[genes['ensembl'] != ''].drop_duplicates('ensembl').set_index('ensembl')['symbol']
    elif kind == 'entrez':
        key = pd.Index([i.split('.')[0] for i in ids])
        lookup = genes[genes['entrez'] != ''].drop_duplicates('entrez').set_index('entrez')['symbol']
    elif kind == 'symbol':
        key = ids
        current = set(genes['symbol'])
        lookup = pd.Series({s: s for s in current})
        alias = {k: v for k, v in ann['alias'].items() if k not in current}
        lookup = pd.concat([lookup, pd.Series(alias)])
        lookup = lookup[~lookup.index.duplicated(keep='first')]
    elif kind == 'refseq':
        raise ValueError("RefSeq identifiers are not in the annotation table; convert them to Ensembl "
                         "or Entrez ids first")
    else:
        raise ValueError(f"cannot convert identifiers of type '{kind}'; pass id_type= explicitly")

    new = lookup.reindex(key)
    if user_map is not None:
        # the user's mapping wins wherever it has an entry; the packaged table fills the rest
        override = user_map.reindex(ids)
        report['from_user_mapping'] = int(override.notna().sum())
        new = pd.Series(np.where(override.notna(), override.to_numpy(), new.to_numpy()),
                        index=new.index)
    report['annotation'] = ann['source']
    return _relabel(out, ids, new, keep_unmapped, aggregate, report, kind, verbose)


def _relabel(out, ids, new, keep_unmapped, aggregate, report, kind, verbose):
    """Apply an identifier -> symbol mapping to the rows of `out`, merging rows that collide."""
    mapped = new.notna().to_numpy()
    renamed = int(np.sum(mapped & (new.fillna('').to_numpy() != ids.to_numpy())))
    if not keep_unmapped:
        out = out[mapped]
        out.index = new[mapped].to_numpy()
    else:
        out.index = np.where(mapped, new.fillna('').to_numpy(), ids.to_numpy())

    duplicated = int(out.index.duplicated().sum())
    if duplicated:
        first_seen = list(dict.fromkeys(out.index))              # keep the original gene order
        if aggregate == 'max':                                   # keep the row with the highest mean
            order = np.argsort(-out.mean(axis=1).to_numpy(), kind='stable')
            out = out.iloc[order]
            out = out[~out.index.duplicated(keep='first')]
        elif aggregate == 'first':
            out = out[~out.index.duplicated(keep='first')]
        else:
            out = out.groupby(level=0).agg(aggregate)
        out = out.loc[first_seen]

    report.update({'mapped': int(mapped.sum()), 'renamed': renamed,
                   'unmapped': int((~mapped).sum()), 'duplicates_merged': duplicated,
                   'aggregate': aggregate, 'n_output': len(out)})
    if not keep_unmapped and mapped.mean() < 0.5:
        warnings.warn(
            f"TMEImmune: only {report['mapped']} of {len(ids)} identifiers (detected as {kind}) could be "
            f"mapped to gene symbols, so {report['unmapped']} rows were dropped. If these are in-house "
            f"or non-standard identifiers, pass mapping= with your own identifier -> symbol table.",
            UserWarning)
    if verbose:
        print(f"---------- identifiers detected as {kind}: {report['mapped']} of {len(ids)} mapped to "
              f"symbols ({renamed} renamed, {report['unmapped']} unmapped, "
              f"{duplicated} duplicate symbols merged by '{aggregate}') ----------")
    return out, report


def gene_lengths(symbols = None):
    """
    Gene lengths in bases (union of the exons), indexed by HGNC symbol; for counts -> TPM.
    symbols: optional list to restrict to, in which case missing genes come back as NaN.
    """
    genes = load_annotation()['genes']
    lengths = genes.dropna(subset=['length']).drop_duplicates('symbol').set_index('symbol')['length']
    return lengths if symbols is None else lengths.reindex(pd.Index(symbols))
