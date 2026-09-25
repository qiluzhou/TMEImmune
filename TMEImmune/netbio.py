import numpy as np 
import pandas as pd
import hashlib
import warnings
from collections import defaultdict
import scipy.stats as stat
from sklearn.model_selection import GridSearchCV, StratifiedKFold
#import networkx as nx
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from TMEImmune import data_processing
from TMEImmune import nb_utilities as nbu

warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


_TRAIN_CACHE = {}


def clear_netbio_cache():
	"""Empty the cached NetBio training results (pathway selection + fitted model)."""
	_TRAIN_CACHE.clear()


def _fingerprint(*parts):
	h = hashlib.blake2b(digest_size=16)
	for part in parts:
		h.update(repr(part).encode())
	return h.hexdigest()


def _select_proximal_pathways(target, train_genes, reactome, nGene, qval):
	"""Biomarker-proximal Reactome pathways: pathways enriched (hypergeometric) for the top network
	propagated biomarker genes of the treatment target."""
	bdf = data_processing.load_data('nb_biomarker/%s.txt' % target).dropna(subset=['gene_id'])
	train_gene_set = set(train_genes)
	b_genes = []
	for gene in bdf.sort_values(by=['propagate_score'], ascending=False)['gene_id'].tolist():
		if gene in train_gene_set and gene not in b_genes:
			b_genes.append(gene)
			if len(b_genes) >= nGene:
				break
	b_gene_set = set(b_genes)

	M, N = len(train_genes), len(b_gene_set)
	pws = list(reactome.keys())
	n_arr, k_arr = np.empty(len(pws)), np.empty(len(pws))
	for i, pw in enumerate(pws):
		pw_genes = set(reactome[pw]) & train_gene_set
		n_arr[i] = len(pw_genes)
		k_arr[i] = len(pw_genes & b_gene_set)
	pvalues = stat.hypergeom.sf(k_arr - 1, M, n_arr, N)          # vectorized over pathways
	_, qvalues, _, _ = multipletests(pvalues)
	tmp = pd.DataFrame({'pw': pws, 'p': pvalues, 'q': qvalues}).sort_values(by=['q'])
	return tmp.loc[tmp['q'] <= qval, 'pw'].tolist()


def _train_netbio(target, train_edf, train_epdf, train_responses, train_geneid, reactome,
				  nGene, qval, penalty, n_jobs = None):
	"""Select the proximal pathways and fit the NetBio logistic model. Cached: the training data are fixed
	(Gide by default), so repeated scoring of cohorts with the same gene/pathway coverage refits nothing."""
	train_genes = train_edf[train_geneid].tolist()
	key = _fingerprint(target, nGene, qval, penalty, sorted(train_genes),
					   sorted(train_epdf['pathway'].tolist()), np.asarray(train_responses).tolist())
	if key in _TRAIN_CACHE:
		return _TRAIN_CACHE[key]

	proximal_pathways = _select_proximal_pathways(target, train_genes, reactome, nGene, qval)
	if len(proximal_pathways) == 0:
		raise ValueError(
			f"NetBio: no pathway is significantly proximal to the {target} biomarkers among the "
			f"{len(train_genes)} genes shared by the training data and this cohort, so there is nothing "
			f"to train on. This happens when the input covers only part of the transcriptome (a targeted "
			f"panel): NetBio needs transcriptome-wide expression. Check the input with "
			f"data_processing.assess_gene_coverage.")
	X_train = train_epdf.loc[train_epdf['pathway'].isin(proximal_pathways), :].T.values[1:]

	param_grid = {'penalty': ['l2'], 'max_iter': [1000], 'solver': ['lbfgs'],
				  'C': np.arange(0.1, 1, 0.1), 'class_weight': ['balanced']}
	if penalty == 'none':
		param_grid = {'penalty': [None], 'max_iter': [1000], 'class_weight': ['balanced']}
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	gcv = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=cv, scoring='roc_auc',
					   n_jobs=n_jobs).fit(X_train, train_responses)

	out = (gcv.best_estimator_, proximal_pathways)
	_TRAIN_CACHE[key] = out
	return out



def get_netbio(test_gene, test_clin, test_clinid, test_ssgsea = None, 
			   cohort_targets = {'PD1':['Gide']}, train_gene = None, 
			   train_geneid = "gene_id", train_clin = None, train_clinid = None, train_ssgsea = None, 
			   nGene = 200, qval = 0.01, penalty = "l2", n_jobs = None):

	""" 
	train NetBio model and get NetBio score
	test_gene: gene expression dataset, gene symbol must be the first column of the dataset
	test_clin: clinical dataset having treatment response
	test_clinid: column name of treatment response column
	test_ssgsea: whether to perform ssgsea on the gene expression dataset
	cohort_targets: {treatment: training dataset}, by default use Gide et al.'s data to train model
	train_geneid: column of gene symbol in training dataset
	"""
	if train_gene is None:
		train_dataset = "Gide"
		train_geneid = 'genes'
		target = "PD1"
		train_samples, train_edf, train_epdf, train_responses = nbu.parse_reactomeExpression_and_immunotherapyResponse(train_dataset)

	else:
		target, train_dataset = cohort_targets.items
		train_data = nbu.netbio_data(train_gene, train_clin, train_clinid, train_ssgsea)
		train_edf, train_epdf, train_responses = train_data.get_gene(train_geneid), train_data.get_ssgsea(), train_data.get_clin()

	reactome = data_processing.load_data("c2.all.v7.2.symbols.gmt")

	test_data = nbu.netbio_data(test_gene, test_clin, test_clinid, test_ssgsea)
	test_geneid = test_gene.columns[0]
	test_edf, test_epdf, test_responses = test_data.get_gene(test_geneid), test_data.get_ssgsea(), test_data.get_clin()

	### data cleanup: match genes and pathways between cohorts
	#common_genes, common_pathways = list(set(train_edf[train_geneid].tolist()) & set(test_edf[test_geneid].tolist())), list(set(train_epdf['pathway'].tolist()) & set(test_epdf['pathway'].tolist()))
	common_genes, common_pathways = list(set(train_edf[train_geneid]) & set(test_edf[test_geneid])), list(set(train_epdf['pathway']) & set(test_epdf['pathway']))
	train_edf = train_edf.loc[train_edf[train_geneid].isin(common_genes),:].sort_values(by=train_geneid)
	train_epdf = train_epdf.loc[train_epdf['pathway'].isin(common_pathways),:].sort_values(by='pathway')
	test_edf = test_edf.loc[test_edf[test_geneid].isin(common_genes),:].sort_values(by=test_geneid)
	test_epdf = test_epdf.loc[test_epdf['pathway'].isin(common_pathways),:].sort_values(by = 'pathway')


	### data cleanup: expression standardization
	# only the pathway tables are standardized: the gene-level matrices are used for their gene names
	# (common genes, biomarker overlap), never for their values
	train_epdf = nbu.expression_StandardScaler(train_epdf)
	test_epdf = nbu.expression_StandardScaler(test_epdf)

	model, proximal_pathways = _train_netbio(target, train_edf, train_epdf, train_responses, train_geneid,
											 reactome, nGene, qval, penalty, n_jobs)

	test_dic = {'NetBio': test_epdf.loc[test_epdf['pathway'].isin(proximal_pathways), :]}
	X_test = test_dic['NetBio'].T.values[1:]
	gcv = model

	pred_proba = gcv.predict_proba(X_test)[:,1]
	logit_pred = logit(np.clip(pred_proba, 1e-6, 1 - 1e-6))

	# indexed by sample, so callers cannot pair the scores with the wrong samples
	return pd.Series(logit_pred, index=test_data.common_id, name='NetBio')
