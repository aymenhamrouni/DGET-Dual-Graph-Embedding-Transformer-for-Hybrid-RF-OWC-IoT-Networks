# utils_DGET.py: Helper functions and model classes for DGET training/inference
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv
from tqdm import tqdm
from itertools import combinations
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve, auc
)

# ---- Utility functions from DGET.py ----
def find_indices_with_values(pairs, values):
    indices = [index for index, pair in enumerate(pairs) if pair[0] in values or pair[1] in values]
    return indices

def generate_all_combinations_indices(n):
    pairs = [(i, j) for i in range(n + 1) for j in range(n + 1) if i != j]
    all_combinations = list(combinations(range(n + 1), 2))
    result_indices = []
    for combo in all_combinations:
        result_indices.append(find_indices_with_values(pairs, set(combo)))
    return result_indices

def postProcessing(target, size, Int):
    final = []
    nSwitch = 0
    a = len(target)
    target = target.reshape(a//(size*(size-1)), (size*(size-1)))
    for f in range(a//(size*(size-1))):
        batch = target[f]
        for S in Int:
            theSum = batch[S]
            org = batch
            if (np.count_nonzero(theSum==0)+np.count_nonzero(theSum==1)+np.count_nonzero(theSum==3)+np.count_nonzero(theSum==5))==8:
                if np.count_nonzero(theSum==6)!=0:
                    batch[org==6]=5
                    org[org==6]=5
                    nSwitch+=1
                    theSum=batch[S]
                elif np.count_nonzero(theSum==7)!=0:
                    batch[org==7]=5
                    org[org==7]=5
                    nSwitch+=1
                    theSum=batch[S]
            if (np.count_nonzero(theSum==0)+np.count_nonzero(theSum==1)+np.count_nonzero(theSum==3)+np.count_nonzero(theSum==5))==8:
                if np.count_nonzero(theSum==2)!=0:
                    batch[org==2]=1
                    org[org==2]=1
                    nSwitch+=1
                    theSum=batch[S]
            if (np.count_nonzero(theSum==0)+np.count_nonzero(theSum==1)+np.count_nonzero(theSum==3)+np.count_nonzero(theSum==5))==8:
                if np.count_nonzero(theSum==4)!=0:
                    batch[org==4]=3
                    org[org==4]=3
                    nSwitch+=1
                    theSum=batch[S]
        final.append(org)
    return torch.tensor(np.stack(final)).reshape(a), nSwitch/(a)

def compute_class_weights(dataset):
    labels = []
    for data in dataset:
        for item in data:
            labels.append(item.y.view(-1).numpy())
    labels = np.concatenate(labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    weights = total_samples / (len(class_counts) * class_counts)
    weights = np.sqrt(weights)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    weights = weights / (np.sum(weights)+1e-8) * len(class_counts)
    return weights

def create_batched_dataloaders(train_known, train_recorded, test_known, test_recorded, batch_size=32):
    class GraphDataset(torch.utils.data.Dataset):
        def __init__(self, known_data, recorded_data):
            self.known_data = known_data
            self.recorded_data = recorded_data
        def __len__(self):
            return len(self.known_data)
        def __getitem__(self, idx):
            return self.known_data[idx], self.recorded_data[idx]
    train_dataset = GraphDataset(train_known, train_recorded)
    test_dataset = GraphDataset(test_known, test_recorded)
    def collate_fn(batch):
        known_batch = []
        recorded_batch = []
        for known_instance, recorded_instance in batch:
            known_batch.append(known_instance)
            recorded_batch.append(recorded_instance)
        return known_batch, recorded_batch
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader

def normalize_edge_weights(weights):
    min_val = np.min(weights)
    max_val = np.max(weights)
    return (weights - min_val) / (max_val - min_val + 1e-6)  # Avoid div0

def normalize_edge_features(features):
    scaler = RobustScaler()
    return scaler.fit_transform(features)

def normalize_node_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# --- MAIN DGET ARCHITECTURE ---
class WeightedSAGEConv(SAGEConv):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.lin_l = nn.Linear(in_channels, out_channels)
        self.edge_weight_scale = nn.Parameter(torch.tensor(1.0))
        self._edge_weight = None
    def message(self, x_j, edge_weight=None):
        if edge_weight is not None:
            edge_weight = self.edge_weight_scale * (edge_weight ** 2)
            return x_j * edge_weight.unsqueeze(-1)
        return x_j
    def forward(self, x, edge_index, edge_weight=None):
        self._edge_weight = edge_weight
        return super().forward(x, edge_index)
    def propagate(self, edge_index, size=None, **kwargs):
        if self._edge_weight is not None:
            kwargs['edge_weight'] = self._edge_weight
        return super().propagate(edge_index, size=size, **kwargs)

class TransformerGNN(nn.Module):
    def __init__(self, node_features, num_classes, d_model=32, nhead=8, dropout=0.2, edge_features_dim=None):
        super().__init__()
        self.d_model = d_model
        # Temporal encoder
        self.temporal_encoder = nn.LSTM(1, d_model, batch_first=True, bidirectional=True)
        self.temporal_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gat1 = GATConv(node_features, d_model, heads=8, dropout=dropout, edge_dim=1)
        self.gat2 = GATConv(d_model * 8, d_model, heads=1, dropout=dropout, edge_dim=1)
        self.gat_norm1 = nn.LayerNorm(d_model * 8)
        self.gat_norm2 = nn.LayerNorm(d_model)
        self.gat_dropout = nn.Dropout(dropout)
        self.sage1 = WeightedSAGEConv(d_model, d_model)
        self.sage2 = WeightedSAGEConv(d_model, d_model)
        self.sage_norm1 = nn.LayerNorm(d_model)
        self.sage_norm2 = nn.LayerNorm(d_model)
        self.sage_dropout = nn.Dropout(dropout)
        ef_dim = edge_features_dim if edge_features_dim is not None else 6
        self.edge_processor = nn.Sequential(
            nn.Linear(ef_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.edge_classifier = nn.Sequential(
            nn.Linear(d_model * 2 + 1, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def process_snapshot(self, x, edge_index, edge_attr, edgeweight):
        x_temporal, _ = self.temporal_encoder(x.unsqueeze(0))
        x_temporal = x_temporal.squeeze(0)
        x_temporal = self.temporal_proj(x_temporal)
        edge_attr_processed = self.edge_processor(edge_attr).squeeze(-1) * edgeweight.view(-1)
        x_gat1 = self.gat1(x_temporal, edge_index, edge_attr=edge_attr_processed)
        x_gat1 = self.gat_norm1(x_gat1)
        x_gat1 = self.gelu(x_gat1)
        x_gat1 = self.gat_dropout(x_gat1)
        x_gat2 = self.gat2(x_gat1, edge_index, edge_attr=edge_attr_processed)
        x_gat2 = self.gat_norm2(x_gat2)
        x_gat2 = self.gelu(x_gat2)
        x_gat2 = self.gat_dropout(x_gat2)
        x_sage1 = self.sage1(x_gat2, edge_index,edge_weight=edge_attr_processed)
        x_sage1 = self.sage_norm1(x_sage1)
        x_sage1 = self.gelu(x_sage1)
        x_sage1 = self.sage_dropout(x_sage1)
        x_sage2 = self.sage2(x_sage1, edge_index,edge_weight=edge_attr_processed)
        x_sage2 = self.sage_norm2(x_sage2)
        x_sage2 = self.gelu(x_sage2)
        x_sage2 = self.sage_dropout(x_sage2)
        return x_gat2, x_sage2, edge_attr_processed
    def forward(self, known_graph, labeled_graph=None):
        batch_size = len(known_graph)
        num_snapshots = len(known_graph[0])
        all_x, all_edge_index, all_edge_attr, all_edge_weight, all_y = [], [], [], [], []
        for instance in known_graph:
            for snapshot in instance:
                all_x.append(snapshot.x)
                all_edge_index.append(snapshot.edge_index)
                all_edge_attr.append(snapshot.edge_attr)
                all_edge_weight.append(snapshot.edge_weight)
                all_y.append(snapshot.y)
        all_x = torch.stack(all_x)
        all_edge_index = torch.stack(all_edge_index)
        all_edge_attr = torch.stack(all_edge_attr)
        all_edge_weight = torch.stack(all_edge_weight)
        all_y = torch.stack(all_y)
        all_gat_features, all_sage_features, all_edge_attr_processed = self.process_snapshot(
            all_x.view(-1, all_x.size(-1)),
            all_edge_index.view(2, -1),
            all_edge_attr.view(-1, all_edge_attr.size(-1)),
            all_edge_weight.view(-1, all_edge_attr.size(-1))
        )
        all_gat_features = all_gat_features.view(batch_size, num_snapshots, -1, self.d_model)
        all_sage_features = all_sage_features.view(batch_size, num_snapshots, -1, self.d_model)
        all_edge_attr_processed = all_edge_attr_processed.view(batch_size, num_snapshots, -1)
        all_scores = []
        known_embeddings = []
        enhanced_known_embeddings = []
        labeled_embeddings = []
        consistency_loss = 0
        for b in range(batch_size):
            instance_scores = []
            instance_gat_embeddings = []
            instance_sage_embeddings = []
            for s in range(num_snapshots):
                gat_features = all_gat_features[b, s]
                sage_features = all_sage_features[b, s]
                edge_attr = all_edge_attr_processed[b, s]
                edge_index = all_edge_index[b * num_snapshots + s]
                row, col = edge_index
                gat_row = gat_features[row]
                gat_col = gat_features[col]
                sage_row = sage_features[row]
                sage_col = sage_features[col]
                gat_edge_embeddings = torch.cat([gat_row, gat_col], dim=-1)
                instance_gat_embeddings.append(gat_edge_embeddings)
                sage_edge_embeddings = torch.cat([sage_row, sage_col], dim=-1)
                instance_sage_embeddings.append(sage_edge_embeddings)
                edge_features = torch.cat([sage_row, sage_col, edge_attr.unsqueeze(-1)], dim=-1)
                scores = self.edge_classifier(edge_features)
                instance_scores.append(scores)
            instance_gat_embeddings = torch.stack(instance_gat_embeddings)
            instance_sage_embeddings = torch.stack(instance_sage_embeddings)
            instance_scores = torch.cat(instance_scores, dim=0)
            known_embeddings.append(instance_gat_embeddings)
            enhanced_known_embeddings.append(instance_sage_embeddings)
            all_scores.append(instance_scores)
        known_embeddings = torch.stack(known_embeddings)
        enhanced_known_embeddings = torch.stack(enhanced_known_embeddings)
        all_scores = torch.cat(all_scores, dim=0)
        if labeled_graph is not None:
            for b, (known_instance, labeled_instance) in enumerate(zip(known_graph, labeled_graph)):
                for s, (known_snapshot, labeled_snapshot) in enumerate(zip(known_instance, labeled_instance)):
                    labeled_x = labeled_snapshot.x
                    labeled_edge_index = labeled_snapshot.edge_index
                    labeled_edge_attr = labeled_snapshot.edge_attr
                    labeled__edgeweight = labeled_snapshot.edge_weight
                    _, labeled_sage_features, _ = self.process_snapshot(
                        labeled_x,
                        labeled_edge_index,
                        labeled_edge_attr, labeled__edgeweight
                    )
                    row, col = labeled_edge_index
                    labeled_row = labeled_sage_features[row]
                    labeled_col = labeled_sage_features[col]
                    labeled_edge_embeddings = torch.cat([labeled_row, labeled_col], dim=-1)
                    labeled_embeddings.append(labeled_edge_embeddings)
                    known_emb = enhanced_known_embeddings[b, s]
                    consistency_loss += F.mse_loss(
                        known_emb * all_edge_weight,
                        labeled_edge_embeddings * labeled__edgeweight
                    )
            labeled_embeddings = torch.stack(labeled_embeddings)
        else:
            labeled_embeddings = None
        return F.log_softmax(all_scores, dim=-1), consistency_loss, known_embeddings, enhanced_known_embeddings, labeled_embeddings, all_edge_weight.view(-1)
