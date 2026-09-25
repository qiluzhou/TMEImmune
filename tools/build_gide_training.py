"""
Build the trimmed NetBio training table shipped with the package.

NetBio trains on the Gide cohort. Of the three raw files, the 90 MB expression matrix is only used
for its gene names -- the model itself is fitted on the pathway ssGSEA table -- so the package ships
gene names + pathway table + responses instead, in one compressed npz (about 1 MB).

Run from the project root, with the raw files still in TMEImmune/data/Gide/:

    python tools/build_gide_training.py
"""
import os
import numpy as np

from TMEImmune import nb_utilities as nbu

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "TMEImmune", "data", "Gide", "gide_training.npz")

samples, edf, epdf, responses = nbu.parse_reactomeExpression_and_immunotherapyResponse("Gide", raw=True)
np.savez_compressed(
    out,
    genes=edf['genes'].astype(str).to_numpy(dtype='U'),
    pathways=epdf['pathway'].astype(str).to_numpy(dtype='U'),
    samples=np.asarray(samples, dtype='U'),
    pathway_expression=epdf.iloc[:, 1:].to_numpy(dtype=float),
    responses=np.asarray(responses),
)
print(f"wrote {out} ({os.path.getsize(out) / 1024 ** 2:.2f} MB) "
      f"- {len(edf)} genes, {epdf.shape[0]} pathways, {len(samples)} samples")
