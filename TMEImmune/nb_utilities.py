import numpy as np 
import pandas as pd
import os
import warnings
import hashlib
from functools import lru_cache
#import gseapy as gp
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from TMEImmune import data_processing
from joblib import Parallel, delayed


class nb_pathway():
	def __init__(self):
		self.gene_set = data_processing.load_data("gene_sets_full.json")
    
	def reactome_geneset(self):
		return self.gene_set
	

class netbio_data:
	def __init__(self, gene, clin, response, ssgsea = None):
		self.gene = gene
		self.clin = clin[~clin[response].isna()]
		self._ssgsea = ssgsea
		self.response = response
		# keep the order of the expression matrix: the scores are returned in this order, and an
		# unordered set() here used to shuffle samples differently on every run
		clin_ids = set(self.clin.index)
		self.common_id = [c for c in gene.columns if c in clin_ids]

	def get_gene(self, gene_id):
		gene_columns = [gene_id] + self.common_id
		return self.gene[gene_columns]
	
	def get_clin(self):
		clin = self.clin.loc[self.common_id,:]
		if clin[self.response].isin(["R", "NR"]).all():
			clin_resp = clin[self.response].apply(lambda x: 1 if x == "R" else 0)
		elif clin[self.response].isin([0,1]).all():
			clin_resp = clin[self.response]
		else:
			raise ValueError("Unsupported response type")
		return clin_resp
	
	def get_ssgsea(self):
		ssgsea_result = self._ssgsea
		ssgsea_gene = self.gene.iloc[:,1:]
		ssgsea_gene.index = self.gene.iloc[:,0]
		if self._ssgsea is None:
			ssgsea_result = ssgsea(ssgsea_gene)

		ssgsea_col = ['pathway'] + self.common_id
		ssgsea_result = ssgsea_result[ssgsea_col]

		return ssgsea_result


## pathway expression and immunotherapy response
@lru_cache(maxsize=4)
def parse_reactomeExpression_and_immunotherapyResponse(dataset, Prat_cancer_type='MELANOMA', raw=False):
	"""Training cohort (Gide) gene names, pathway ssGSEA table and responses.

	By default this reads the trimmed table shipped with the package (data/Gide/gide_training.npz):
	NetBio uses the training expression matrix only for its gene names -- the model is fitted on the
	pathway table -- so the 90 MB text file does not have to be parsed. raw=True forces the original
	files (used by tools/build_gide_training.py to regenerate the trimmed table).
	The result is cached, so it is built once per session rather than once per NetBio call.
	"""
	if not raw:
		trimmed = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Gide", "gide_training.npz")
		if os.path.exists(trimmed):
			with np.load(trimmed, allow_pickle=False) as z:
				samples = list(z['samples'])
				edf = pd.DataFrame({'genes': z['genes']})
				epdf = pd.DataFrame(z['pathway_expression'], columns=samples)
				epdf.insert(0, 'pathway', z['pathways'])
				return samples, edf, epdf, z['responses']
		warnings.warn("NetBio: trimmed training table not found, parsing the raw Gide files "
					  "(run tools/build_gide_training.py to build it)", UserWarning)

	edf = data_processing.load_data('Gide/expression_mRNA.norm3.txt')
	epdf = data_processing.load_data('Gide/pathway_expression_ssgsea.txt')
	pdf = data_processing.load_data('Gide/patient_df.txt')
	
	# features, labels
	exp_dic, responses = defaultdict(list), []
	e_samples = []
	for sample, response in zip(pdf['Patient'].tolist(), pdf['Response'].tolist()):
		# labels
		binary_response = response
		if response == 'NR':
			binary_response = 0
		if response == 'R':
			binary_response = 1
		# features
		for e_sample in epdf.columns:
			tmp = []

			if (sample in e_sample) or (e_sample in sample):
				e_samples.append(e_sample)
				responses.append(binary_response)	

	edf = pd.DataFrame(data=edf, columns=np.append(['gene_id'], e_samples))
	edf = edf.rename(columns = {"gene_id": "genes"})
	epdf = pd.DataFrame(data=epdf, columns=np.append(['pathway'], e_samples))
	responses = np.array(responses)
	return e_samples, edf, epdf, responses


def expression_StandardScaler(exp_df):
	'''
	Standardize each row (gene or pathway) of an id + samples dataframe across the samples.
	Vectorized version of the original loop; rows with zero variance are returned as zeros,
	as scikit-learn's StandardScaler does.
	'''
	col1 = exp_df.columns[0]
	values = exp_df.iloc[:, 1:].to_numpy(dtype=float)
	mean = values.mean(axis=1, keepdims=True)
	std = values.std(axis=1, keepdims=True)
	std[std == 0] = 1.0
	scaled = (values - mean) / std
	output = pd.DataFrame(scaled, columns=exp_df.columns[1:], index=exp_df.index)
	output.insert(0, col1, exp_df[col1].to_numpy())
	return output


_RANK_CACHE = {}
_RANK_CACHE_SIZE = 2


def _rank_matrix(df):
	"""
	Rank the expression matrix once and return everything the enrichment score needs:
	W = rank ** (1/4) (the ssGSEA weights) and rev = N + 1 - position in the descending ranking.
	The result is cached, so the scores that share an expression matrix (ESTIMATE, ISTME, NetBio)
	rank it only once.
	"""
	values = df.to_numpy(dtype=float, copy=False)
	key = (values.shape, hashlib.blake2b(np.ascontiguousarray(values).tobytes(), digest_size=16).hexdigest())
	if key in _RANK_CACHE:
		return _RANK_CACHE[key]

	ranked = pd.DataFrame(values).rank(axis=0, method="average")
	ranks = np.abs(ranked.to_numpy())
	N = ranks.shape[0]
	# rev = N + 1 - position in the descending ranking. Genes with equal expression share the average
	# position of their group, so the score does not depend on the order of the rows (ssgsea_reference
	# leaves tied genes in whatever order pandas' sort returns).
	rev = N + 1.0 - (-ranked).rank(axis=0, method="average").to_numpy()
	out = (ranks ** 0.25, rev, N)
	if len(_RANK_CACHE) >= _RANK_CACHE_SIZE:
		_RANK_CACHE.pop(next(iter(_RANK_CACHE)))
	_RANK_CACHE[key] = out
	return out


def clear_ssgsea_cache():
	"""Empty the cached ranking of expression matrices."""
	_RANK_CACHE.clear()


def ssgsea(df, geneset = None, score = "NetBio", verbose = True):
	"""
	single sample GSEA. Same enrichment score as ssgsea_reference, computed in closed form:
	for a gene set H in a sample whose ssGSEA weights are w and whose descending-rank positions are p,

	    ES = sum_{g in H} w_g * rev_g / sum_{g in H} w_g  -  (N(N+1)/2 - sum_{g in H} rev_g) / (N - |H|)

	with rev_g = N + 1 - p_g. This costs O(|H|) per gene set and sample instead of one cumulative sum
	over all genes, and the ranking itself is computed once per expression matrix and cached.

	df: expression matrix, gene symbols as index and samples as columns
	geneset: {name: genes} (dict or dataframe of gene columns); default the NetBio reactome pathways
	score: 'NetBio', 'ESTIMATE' or 'ISTME' -- selects the normalization of the enrichment scores
	"""
	if geneset is None:
		gene_set_dict = {pw: genes for pw, genes in nb_pathway().reactome_geneset().items() if genes}
	else:
		gene_set_dict = geneset

	pos_map = {}
	for i, g in enumerate(df.index):
		pos_map.setdefault(g, i)

	W, rev, N = _rank_matrix(df.apply(pd.to_numeric))
	T = N * (N + 1.0) / 2.0
	sigs = list(gene_set_dict.keys())
	ES = np.zeros((len(sigs), W.shape[1]))

	if verbose:
		print("-------- Begin ssgsea --------")
	for j, sig in enumerate(sigs):
		genes = gene_set_dict[sig]
		if isinstance(genes, pd.Series):
			genes = genes.dropna().tolist()
		hits = np.fromiter({pos_map[g] for g in genes if g in pos_map}, dtype=int)
		k = len(hits)
		if k == 0 or k >= N:
			continue
		w_hit = W[hits]                       # |H| x samples
		sum_w = w_hit.sum(axis=0)
		sum_wr = (w_hit * rev[hits]).sum(axis=0)
		sum_r = rev[hits].sum(axis=0)
		with np.errstate(divide='ignore', invalid='ignore'):
			es = sum_wr / sum_w - (T - sum_r) / (N - k)
		ES[j] = np.where(sum_w > 0, es, 0.0)
	if verbose:
		print("-------- Complete computing enrichment score --------")

	ES_df = pd.DataFrame(ES, index=sigs, columns=df.columns).dropna(how='all')

	if score == "ESTIMATE":
		return ES_df.T
	elif score == "ISTME":
		nes_df = ES_df.apply(lambda row: row / (row.max() - row.min()) if row.max() - row.min() != 0 else row, axis = 1)
		return nes_df.T

	es_range = (ES_df.min().min(skipna=True), ES_df.max().max(skipna=True))
	if pd.isna(es_range[0]) or pd.isna(es_range[1]) or not np.isfinite(es_range[0]) or not np.isfinite(es_range[1]):
		raise ValueError("Normalizing factor contains NAs or infinite values")
	nes_df = ES_df / (es_range[1] - es_range[0])
	return nes_df.reset_index().rename(columns = {"index": "pathway"})


def ssgsea_reference(df, geneset = None, score = "NetBio"):
	"""Original per-sample ssGSEA implementation, kept as a reference for testing the fast version."""

	if geneset is None:
		# get netbio reactome pathways and corresponding genesets
		gene_set_dict = nb_pathway().reactome_geneset()
		# Remove pathways where the gene list is empty
		gene_set_dict = {pathway: genes for pathway, genes in gene_set_dict.items() if genes}
	else:
		gene_set_dict = geneset

	gene_set_dict = {sig: set(genes) & set(df.index) for sig, genes in gene_set_dict.items()}

	df1 = df.apply(pd.to_numeric)
	df_ranked = df1.rank(axis = 0, method = "average")
	df_ranked = df_ranked.apply(abs)
	num_signatures = len(gene_set_dict)
	num_samples = df_ranked.shape[1]

	sigs = list(gene_set_dict.keys())

    #for sample in range(num_samples):  # Loop through samples
	def compute_sample_es(sample):
		ordered_genes = df_ranked.iloc[:, sample].sort_values(ascending=False)
		ordered_genes1 = ordered_genes.pow(1. / 4)

		gene_lookup = {gene: idx for idx, gene in enumerate(ordered_genes.index)}
		sample_ES = np.zeros(num_signatures)

		for j, sig in enumerate(sigs):  # Iterate over signatures
			hit_genes = gene_set_dict[sig] & set(gene_lookup.keys())  # Find intersection of genes

			if not hit_genes:
				sample_ES[j] = 0
				continue 

			hit_indices = np.array([gene_lookup[gene] for gene in hit_genes], dtype = int)  # Get indices for these genes
			hit_ind = np.zeros(len(ordered_genes), dtype=bool)
			hit_ind[hit_indices] = True
			no_hit_ind = ~hit_ind
			hit_exp = ordered_genes1[hit_ind]

			if np.sum(hit_exp) > 0:
				no_hit_penalty = np.cumsum(no_hit_ind / np.sum(no_hit_ind))
				hit_reward = np.cumsum((hit_ind * ordered_genes1) / np.sum(hit_exp))
				sample_ES[j] = np.sum(hit_reward - no_hit_penalty)

		return sample_ES
	print("-------- Begin ssgsea, might take some time --------")
	results = Parallel(n_jobs=-1)(delayed(compute_sample_es)(sample) for sample in range(num_samples))
	print("-------- Complete computing enrichment score --------")
    # Convert results to a NumPy array
	ES_vector = np.array(results).T  # Transpose to match (pathways x samples) shape
        # Convert back to DataFrame
	ES_df = pd.DataFrame(ES_vector, index=sigs, columns=df.columns)
	ES_df = ES_df.replace(r'^\s*$', np.nan, regex=True)
	ES_df = ES_df.replace("nan", np.nan)
	ES_df = ES_df.dropna(how='all')

	if score == "ESTIMATE":
		return ES_df.T
	elif score == "ISTME":
		# Normalize ES_vector for each row
		nes_df = ES_df.apply(lambda row: row / (row.max() - row.min()) if row.max() - row.min() != 0 else row, axis = 1)
		return nes_df.T
		
	print("-------- Normalizing enrichment score --------")

	any_na = ES_df.isna().any().any()
	if any_na:
		es_range = (ES_df.min().min(skipna=True), ES_df.max().max(skipna=True))
	else:
		es_range = (ES_df.min().min(), ES_df.max().max())

	# check if the range is valid
	if pd.isna(es_range[0]) or pd.isna(es_range[1]) or not np.isfinite(es_range[0]) or not np.isfinite(es_range[1]):
		raise ValueError("Normalizing factor contains NAs or infinite values")

	nes_df = ES_df / (es_range[1] - es_range[0])

	print("-------- ssgsea complete --------")

	NES_df = nes_df.reset_index().rename(columns = {"index": "pathway"})

	return NES_df
