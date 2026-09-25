"""
Build the packaged gene annotation table, TMEImmune/data/gene_annotation.npz.

The table maps Ensembl gene ids, Entrez ids and outdated / alias symbols onto current HGNC symbols,
and holds the union-exon length of every gene, which is what turns raw counts into TPM.

It is built from whichever of these sources you have; all are optional, and they are merged:

  --gtf    GENCODE annotation, gives Ensembl id, symbol and gene length
           https://www.gencodegenes.org/human/  ->  "Basic gene annotation", CHR regions,
           e.g. gencode.v47.basic.annotation.gtf.gz (about 35 MB, no need to unzip)

  --hgnc   HGNC complete set, gives current symbols, previous symbols, aliases, Entrez and Ensembl ids
           https://www.genenames.org/download/archive/  ->  hgnc_complete_set.txt (about 15 MB)

  --symbol2entrez  a two-column csv (symbol, entrez_id), e.g. one exported from an earlier project

Example:
    python tools/build_gene_annotation.py --gtf gencode.v47.basic.annotation.gtf.gz \\
                                          --hgnc hgnc_complete_set.txt \\
                                          --symbol2entrez data/symbol2entrez.csv
"""
import argparse
import gzip
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "TMEImmune", "data", "gene_annotation.npz")


def parse_gtf(path):
    """Ensembl id, symbol and union-exon length per gene from a GENCODE GTF."""
    opener = gzip.open if str(path).endswith(".gz") else open
    gene_id, gene_name, starts, ends = [], [], [], []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "exon":
                continue
            attr = f[8]
            gid = attr.split('gene_id "', 1)[1].split('"', 1)[0] if 'gene_id "' in attr else ""
            gname = attr.split('gene_name "', 1)[1].split('"', 1)[0] if 'gene_name "' in attr else ""
            gene_id.append(gid.split(".")[0])
            gene_name.append(gname)
            starts.append(int(f[3]))
            ends.append(int(f[4]))
    exons = pd.DataFrame({"ensembl": gene_id, "symbol": gene_name, "start": starts, "end": ends})
    print(f"  {len(exons)} exons, {exons['ensembl'].nunique()} genes")

    # union of the exons of each gene, so overlapping exons of different transcripts are not counted twice
    exons = exons.sort_values(["ensembl", "start", "end"])
    lengths, sym = {}, {}
    for (gid, gname), grp in exons.groupby(["ensembl", "symbol"], sort=False):
        total, cur_start, cur_end = 0, None, None
        for s, e in zip(grp["start"].to_numpy(), grp["end"].to_numpy()):
            if cur_end is None:
                cur_start, cur_end = s, e
            elif s <= cur_end + 1:
                cur_end = max(cur_end, e)
            else:
                total += cur_end - cur_start + 1
                cur_start, cur_end = s, e
        total += (cur_end - cur_start + 1) if cur_end is not None else 0
        lengths[gid] = lengths.get(gid, 0) + total if gid in lengths else total
        sym[gid] = gname
    out = pd.DataFrame({"ensembl": list(lengths), "symbol": [sym[g] for g in lengths],
                        "length": [lengths[g] for g in lengths]})
    return out


def parse_hgnc(path):
    """Current symbols with their Entrez / Ensembl ids, plus previous and alias symbols."""
    hgnc = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    cols = {c.lower(): c for c in hgnc.columns}
    sym = hgnc[cols["symbol"]]
    genes = pd.DataFrame({
        "symbol": sym,
        "entrez": hgnc[cols.get("entrez_id", cols["symbol"])].fillna("") if "entrez_id" in cols else "",
        "ensembl": hgnc[cols.get("ensembl_gene_id", cols["symbol"])].fillna("") if "ensembl_gene_id" in cols else "",
    })
    alias = {}
    for col in ("prev_symbol", "alias_symbol"):
        if col in cols:
            for current, others in zip(sym, hgnc[cols[col]].fillna("")):
                for other in str(others).split("|"):
                    other = other.strip()
                    if other and other not in alias:
                        alias[other] = current
    print(f"  {len(genes)} HGNC genes, {len(alias)} previous/alias symbols")
    return genes, alias


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtf", help="GENCODE annotation GTF (.gtf or .gtf.gz)")
    ap.add_argument("--hgnc", help="HGNC complete set (tab separated)")
    ap.add_argument("--symbol2entrez", help="csv with columns symbol, entrez_id")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if not any([args.gtf, args.hgnc, args.symbol2entrez]):
        ap.error("give at least one of --gtf, --hgnc, --symbol2entrez")

    frames, alias, sources = [], {}, []
    if args.gtf:
        print(f"reading {args.gtf}")
        g = parse_gtf(args.gtf)
        g["entrez"] = ""
        frames.append(g[["symbol", "ensembl", "entrez", "length"]])
        sources.append(os.path.basename(args.gtf))
    if args.hgnc:
        print(f"reading {args.hgnc}")
        g, alias = parse_hgnc(args.hgnc)
        g["length"] = np.nan
        frames.append(g[["symbol", "ensembl", "entrez", "length"]])
        sources.append(os.path.basename(args.hgnc))
    if args.symbol2entrez:
        print(f"reading {args.symbol2entrez}")
        s2e = pd.read_csv(args.symbol2entrez)
        s2e = s2e.dropna(subset=["symbol"])
        s2e["entrez"] = s2e["entrez_id"].apply(lambda x: "" if pd.isna(x) else str(int(float(x))))
        s2e["ensembl"], s2e["length"] = "", np.nan
        frames.append(s2e[["symbol", "ensembl", "entrez", "length"]])
        sources.append(os.path.basename(args.symbol2entrez))
        print(f"  {len(s2e)} rows, {(s2e['entrez'] != '').sum()} with an Entrez id")

    genes = pd.concat(frames, ignore_index=True)
    genes["symbol"] = genes["symbol"].astype(str).str.strip()
    genes = genes[genes["symbol"] != ""]
    # one row per symbol, filling each field from the first source that has it, so a symbol present in
    # several files keeps the GENCODE length AND the HGNC Entrez id instead of only one source's row
    for col in ("ensembl", "entrez"):
        genes[col] = genes[col].astype(str).str.strip().replace("", np.nan)
    genes = (genes.groupby("symbol", sort=True)[["ensembl", "entrez", "length"]]
             .first()                                   # groupby first skips missing values per column
             .reset_index())

    genes["ensembl"] = genes["ensembl"].fillna("")
    genes["entrez"] = genes["entrez"].fillna("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        symbol=genes["symbol"].to_numpy(dtype="U"),
        ensembl=genes["ensembl"].to_numpy(dtype="U"),
        entrez=genes["entrez"].to_numpy(dtype="U"),
        length=genes["length"].to_numpy(dtype=float),
        alias_from=np.array(list(alias), dtype="U"),
        alias_to=np.array([alias[k] for k in alias], dtype="U"),
        source=np.array([" + ".join(sources)], dtype="U"),
    )
    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1024 ** 2:.2f} MB)")
    print(f"  {len(genes)} symbols, {(genes['ensembl'] != '').sum()} with Ensembl id, "
          f"{(genes['entrez'] != '').sum()} with Entrez id, {genes['length'].notna().sum()} with a length, "
          f"{len(alias)} alias/previous symbols")


if __name__ == "__main__":
    main()
