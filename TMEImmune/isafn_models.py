
import torch
import torch.nn as nn
import torch.nn.functional as F


class mut_model(nn.Module):
    """Mutation feature extractor.
    Input layout: [selected mutation genes ..., total_count, mut_count, mut_max, mut_mean, cancer_type, trt, sex]."""
    def __init__(self, mut_dim, hidden_dim, feature_dim, interaction=False, trt_head=False, monosex=None,
                 monocancer=False, monotrt=False):
        super().__init__()
        self.interaction = interaction
        self.trt_head = trt_head
        self.monosex = monosex
        self.cancer = monocancer
        self.monotrt = monotrt
        self.sex_emb = nn.Embedding(3, 4)
        self.linear1 = nn.Linear(7, hidden_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.linear2 = nn.Linear(hidden_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, mutation, batch):
        mut_features = mutation[:, :-7]
        X_bin = (mut_features > 0).float()
        count_feat = torch.log1p(X_bin.sum(dim=1, keepdim=True))
        max_feat = torch.log1p(mut_features.max(dim=1, keepdim=True).values)
        mean_feat = mut_features.mean(dim=1, keepdim=True)
        mut_features_emb = torch.cat([count_feat, max_feat, mean_feat], dim=1)
        mut_features_selected = mutation[:, -7:-3]
        cohort_type = mutation[:, -3].long()

        mutation_interact = torch.cat([mut_features_selected, mut_features_emb], dim=1)
        feature = self.linear1(mutation_interact)
        h = self.gelu(feature)
        feature = self.dropout(h + feature)
        feature = self.linear2(feature)
        feature = self.norm(feature)
        x = self.gelu(feature)
        return x, mutation[:, -1].long(), cohort_type


class gene_model(nn.Module):
    """Gene-expression feature extractor.
    Input layout: [selected genes ..., drug, met, cohort, cancer, trt, sex]."""
    def __init__(self, mut_dim, hidden_dim, feature_dim, interaction=False, trt_head=False, monosex=None,
                 monocancer=False, monotrt=False):
        super().__init__()
        self.interaction = interaction
        self.trt_head = trt_head
        self.monosex = monosex
        self.cancer = monocancer
        self.monotrt = monotrt
        self.cohort_emb = nn.Embedding(8, 4)
        self.sex_emb = nn.Embedding(3, 4)
        self.gate_layer = nn.Linear(4, mut_dim - 6)
        if interaction:
            self.linear1 = nn.Linear(2 * (mut_dim - 6), hidden_dim)
        else:
            self.linear1 = nn.Linear(mut_dim - 6, hidden_dim)
        self.attn_proj = nn.Linear(mut_dim - 6, hidden_dim)
        self.pos_proj = nn.Linear(mut_dim, hidden_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.linear2 = nn.Linear(hidden_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, mutation):
        mut_features = mutation[:, :-6]
        sex = mutation[:, -1].long()
        cohort = mutation[:, -4].long()
        met = mutation[:, -5].long()
        drug = mutation[:, -6].long()
        cohort_vec = self.cohort_emb(cohort)
        if torch.all(sex == 1):
            gate = torch.sigmoid(self.gate_layer(cohort_vec))
        else:
            gate = 1 + 0.5 * torch.tanh(self.gate_layer(cohort_vec))
        feat_x_cohort = mut_features * gate
        if self.interaction:
            feature = self.linear1(torch.cat([mut_features, feat_x_cohort], dim=1))
        else:
            feature = self.linear1(mut_features)
        h = self.gelu(feature)
        feature = self.dropout(h + feature)
        feature = self.linear2(feature)
        feature = self.norm(feature)
        x = self.gelu(feature)
        return x, met, drug


class Classifier_fusion(nn.Module):
    def __init__(self, hidden_dim, feature_dim, dropout=0.3, modality='gene+mut+tme', trt_head=False, n_cancer=1,
                 sex=None):
        super().__init__()
        self.proj = nn.Linear(feature_dim, hidden_dim)
        self.modality = modality
        self.trt_head = trt_head
        self.sex = sex
        self.sex_emb = nn.Embedding(3, 2)
        self.drug_emb = nn.Embedding(7, 4)
        self.met_emb = nn.Embedding(3, 2)
        if modality == 'gene+mut':
            if self.sex is None:
                self.shared_head = nn.Linear(2*feature_dim+8, feature_dim)
                self.adapter_male = nn.Linear(2*feature_dim+8, feature_dim)
                self.adapter_female = nn.Linear(2*feature_dim+8, feature_dim)
                self.linear1 = nn.Linear(2*feature_dim+8, feature_dim) # for concat fusion
                self.norm1 = nn.LayerNorm(2*feature_dim)
            else:
                self.shared_head = nn.Linear(2*feature_dim+6, feature_dim)
                self.adapter_male = nn.Linear(2*feature_dim+6, feature_dim)
                self.adapter_female = nn.Linear(2*feature_dim+6, feature_dim)
                self.linear1 = nn.Linear(2*feature_dim+6, feature_dim) # for concat fusion
                self.norm1 = nn.LayerNorm(2*feature_dim)
        else:
            if self.sex is None:
                self.shared_head = nn.Linear(feature_dim+8, feature_dim)
                self.linear1 = nn.Linear(feature_dim+8, feature_dim) # for concat fusion
                self.norm1 = nn.LayerNorm(feature_dim)        
                self.adapter_male = nn.Linear(feature_dim+8, feature_dim)
                self.adapter_female = nn.Linear(feature_dim+8, feature_dim)                
            else:
                self.shared_head = nn.Linear(feature_dim+6, feature_dim)
                self.linear1 = nn.Linear(feature_dim+6, feature_dim) # for concat fusion
                self.norm1 = nn.LayerNorm(feature_dim)        
                self.adapter_male = nn.Linear(feature_dim+8, feature_dim)
                self.adapter_female = nn.Linear(feature_dim+8, feature_dim)

        self.norm = nn.LayerNorm(feature_dim)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.heads = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(n_cancer)])
        self.linear2 = nn.Linear(feature_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_features_gene, source_features_mut, tme_features, sex, cancer_type, met, drug):
        source_features_gene = F.normalize(source_features_gene, dim=1)
        source_features_mut = F.normalize(source_features_mut, dim=1)
        sex_emb = self.sex_emb(sex)
        met_emb = self.met_emb(met)
        drug_emb = self.drug_emb(drug)

        if self.sex is None:
            if self.modality == 'gene+mut':
                z = torch.cat([source_features_gene, source_features_mut, sex_emb, met_emb, drug_emb], dim=1)
            elif self.modality == 'gene':
                z = torch.cat([source_features_gene, sex_emb, met_emb, drug_emb], dim=1)
            else:
                z = torch.cat([source_features_mut, sex_emb, met_emb, drug_emb], dim=1)
        else:
            if self.modality == 'gene+mut':
                z = torch.cat([source_features_gene, source_features_mut, met_emb, drug_emb], dim=1)
            elif self.modality == 'gene':
                z = torch.cat([source_features_gene, met_emb, drug_emb], dim=1)
            else:
                z = torch.cat([source_features_mut, met_emb, drug_emb], dim=1)
        z = F.normalize(z, dim=1)

        if self.sex is None:
            shared_z = self.shared_head(z)
            delta_z = torch.zeros_like(shared_z)
            mask_male = sex == 0
            mask_female = sex == 1
            if mask_male.any():
                delta_z[mask_male] = self.adapter_male(z[mask_male])
            if mask_female.any():
                delta_z[mask_female] = self.adapter_female(z[mask_female])
            z = shared_z + 0.5 * delta_z
        else:
            z = self.linear1(z)
        h = self.gelu(z)
        z = self.dropout(h + z)
        return self.linear2(z).squeeze(-1)
