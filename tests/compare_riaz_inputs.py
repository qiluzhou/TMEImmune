"""
Compare the ISAFN model inputs built by the package with the tensors built in validation.ipynb.

1) In validation.ipynb, after cell 43 (riaz_test_fusion / _male / _female are built), run:

    import pandas as pd
    for name, data, gene_df, mut_df, clin in [
            ('merged', riaz_test_fusion, riaz_gene_test, riaz_mutation_test, riaz_clin),
            ('male', riaz_test_fusion_male, riaz_gene_test_male, riaz_mutation_test_male, riaz_clin_male),
            ('female', riaz_test_fusion_female, riaz_gene_test_female, riaz_mutation_test_female, riaz_clin_female)]:
        # prepare_test_data keeps the samples in the order of the gene dataframe
        idx = gene_df.index[gene_df.index.isin(clin.index) & gene_df.index.isin(mut_df.index)]
        pd.DataFrame(data[3].numpy(), index=idx).to_csv(f'nb_gene_input_{name}.csv')   # gene tensor
        pd.DataFrame(data[0].numpy(), index=idx).to_csv(f'nb_mut_input_{name}.csv')    # mutation tensor

   and copy the six nb_*.csv files into data/riaz_isafn/.

2) Run from the project root:  python tests/compare_riaz_inputs.py
   It reports, per model and per input column, where the notebook and package inputs differ.
"""
import os
import numpy as np
import pandas as pd

d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "riaz_isafn")

for name in ["merged", "male", "female"]:
    for kind in ["gene", "mut"]:
        pkg = pd.read_csv(os.path.join(d, f"pkg_{kind}_input_{name}.csv"), index_col=0)
        nb_path = os.path.join(d, f"nb_{kind}_input_{name}.csv")
        if not os.path.exists(nb_path):
            print(f"[{name}/{kind}] {nb_path} not found, skipped")
            continue
        nb = pd.read_csv(nb_path, index_col=0)
        nb.columns = pkg.columns                                    # same column order by construction
        common = pkg.index.intersection(nb.index)
        print(f"\n===== {name} / {kind}: {len(common)} common samples "
              f"(package {len(pkg)}, notebook {len(nb)}) =====")
        diff = (pkg.loc[common] - nb.loc[common]).abs()
        per_col = diff.max().sort_values(ascending=False)
        n_bad = int((per_col > 1e-4).sum())
        print(f"columns differing (> 1e-4): {n_bad} of {len(per_col)}")
        if n_bad:
            print(per_col[per_col > 1e-4].head(15).round(4).to_string())
            meta = [c for c in pkg.columns[-7:] if per_col.get(c, 0) > 1e-4]
            for c in meta:                                          # covariates: show value counts side by side
                print(f"  {c}: package {pkg.loc[common, c].round(4).value_counts().to_dict()} | "
                      f"notebook {nb.loc[common, c].round(4).value_counts().to_dict()}")
