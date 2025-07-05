import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add
from torch_geometric.nn import GATv2Conv, GCNConv
import time

class GATNet(nn.Module):
    def __init__(self, num_features, edge_num_features, head, embedding_dim):
        super(GATNet, self).__init__()
        self.gat1 = GATv2Conv(num_features, embedding_dim, heads=head, edge_dim=edge_num_features)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x_tuple = self.gat1(x, edge_index, edge_attr, return_attention_weights=True)
        x, attention_weights = x_tuple
        x = F.relu(x)
        
        return x, attention_weights  # Output is node embeddings

class EvoEmbedding(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, out_dim, dropout_p=0.35):

        super(EvoEmbedding, self).__init__()


        self.linear1 = torch.nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.relu1 = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(dropout_p)
        # self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relu2 = torch.nn.LeakyReLU()
        self.dropout2 = torch.nn.Dropout(dropout_p)
        # self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relu3 = torch.nn.LeakyReLU()
        self.dropout3 = torch.nn.Dropout(dropout_p)
        # self.bn3 = torch.nn.BatchNorm1d(hidden_dim)

        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim//2, bias=False)
        self.relu4 = torch.nn.LeakyReLU()
        self.dropout4 = torch.nn.Dropout(dropout_p)
        # self.bn4 = torch.nn.BatchNorm1d(hidden_dim//2)

        self.linear_out = torch.nn.Linear(hidden_dim // 2, out_dim, bias=False)

    def forward(self, vX):
        lin1 = self.linear1(vX)
        # lin1 = self.bn1(lin1)
        h_relu1 = self.relu1(lin1)
        drop1 = self.dropout1(h_relu1)

        lin2 = self.linear2(drop1)
        # lin2 = self.bn2(lin2)
        h_relu2 = self.relu2(lin2)
        drop2 = self.dropout2(h_relu2)

        lin3 = self.linear3(drop2)
        # lin3 = self.bn3(lin3)
        h_relu3 = self.relu3(lin3)
        drop3 = self.dropout3(h_relu3)

        lin4 = self.linear4(drop3)
        # lin4 = self.bn4(lin4)
        h_relu4 = self.relu4(lin4)
        drop4 = self.dropout4(h_relu4)

        out = self.linear_out(drop4).squeeze()
        return out

class FlowGenerator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, dropout_p=0.35):

        super(FlowGenerator, self).__init__()


        self.linear1 = torch.nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.relu1 = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(dropout_p)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relu2 = torch.nn.LeakyReLU()
        self.dropout2 = torch.nn.Dropout(dropout_p)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relu3 = torch.nn.LeakyReLU()
        self.dropout3 = torch.nn.Dropout(dropout_p)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)

        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim//2, bias=False)
        self.relu4 = torch.nn.LeakyReLU()
        self.dropout4 = torch.nn.Dropout(dropout_p)
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim//2)


        self.linear_out = torch.nn.Linear(hidden_dim // 2, 1, bias=False)

    def forward(self, vX):
        lin1 = self.linear1(vX)
        lin1 = self.bn1(lin1)
        h_relu1 = self.relu1(lin1)
        drop1 = self.dropout1(h_relu1)

        lin2 = self.linear2(drop1)
        lin2 = self.bn2(lin2)
        h_relu2 = self.relu2(lin2)
        drop2 = self.dropout2(h_relu2)

        lin3 = self.linear3(drop2)
        lin3 = self.bn3(lin3)
        h_relu3 = self.relu3(lin3)
        drop3 = self.dropout3(h_relu3)

        lin4 = self.linear4(drop3)
        lin4 = self.bn4(lin4)
        h_relu4 = self.relu4(lin4)
        drop4 = self.dropout4(h_relu4)

        out = self.linear_out(drop4)

        return out.squeeze()

class Model(nn.Module):
    def __init__(self, 
                 node_num_features, 
                 flow_num_features, 
                 head, 
                 loc_embedding_dim, 
                 evo_emb_hidden_dim, 
                 evo_emb_dim,
                 flow_generator_hidden_dim, 
                 dropout_p=0.35):
        super(Model, self).__init__()
        # CCL
        self.gat_net_snap1 = GATNet(node_num_features, flow_num_features, head, loc_embedding_dim)
        self.gat_net_snap2 = GATNet(node_num_features, flow_num_features, head, loc_embedding_dim)

        # SED
        self.evo_embedding = EvoEmbedding(loc_embedding_dim*4*head+1, evo_emb_hidden_dim, evo_emb_dim, dropout_p=dropout_p)
        self.flow_generator = FlowGenerator(evo_emb_dim + 5, flow_generator_hidden_dim, dropout_p=dropout_p)

    def forward(self, data, od_nodes):
        # Generate Node Embeddings

        # gat_start_time = time.time()
        node_embeddings_snap1, attention_weights_snap1 = self.gat_net_snap1(data.graph_snap1.cuda())
        node_embeddings_snap2, attention_weights_snap2 = self.gat_net_snap2(data.graph_snap2.cuda())

        dist_m = data.dis_m

        # Prepare Edge Features for Classification
        all_edge_features = []

        # Extract origin and destination nodes as separate tensors
        o_nodes = od_nodes[:, 0].long()
        d_nodes = od_nodes[:, 1].long()

        # Gather embeddings for all origin and destination nodes for both snapshots
        o_embeddings_snap1 = node_embeddings_snap1[o_nodes]  # Shape: [num_edges, embedding_dim]
        d_embeddings_snap1 = node_embeddings_snap1[d_nodes]
        o_embeddings_snap2 = node_embeddings_snap2[o_nodes]
        d_embeddings_snap2 = node_embeddings_snap2[d_nodes]

        o_embeddings_diff = o_embeddings_snap2 - o_embeddings_snap1
        d_embeddings_diff = d_embeddings_snap2 - d_embeddings_snap1

        # Compute distances in a vectorized way and convert to a tensor
        distances = torch.tensor([dist_m[o, d] for o, d in zip(o_nodes, d_nodes)], dtype=torch.float).cuda()
        distances = distances.view(-1, 1)  # Reshape to match dimensions for concatenation

        # Concatenate all features in a single operation
        all_edge_features = torch.cat([o_embeddings_snap1, 
                                       d_embeddings_snap1, 
                                       o_embeddings_snap2, 
                                       d_embeddings_snap2, 
                                       distances], dim=1)


        all_evo_embedding = self.evo_embedding(all_edge_features)

        o_visit = torch.tensor(data.node_visit[:, 0], dtype=torch.float).cuda()
        d_visit = torch.tensor(data.node_visit[:, 0], dtype=torch.float).cuda()
        o_visit = o_visit[od_nodes[:, 0].cpu().long()].unsqueeze(1)
        d_visit = d_visit[od_nodes[:, 1].cpu().long()].unsqueeze(1)

        o_visit_next = torch.tensor(data.node_visit[:, 1], dtype=torch.float).cuda()
        d_visit_next = torch.tensor(data.node_visit[:, 1], dtype=torch.float).cuda()
        o_visit_next = o_visit_next[od_nodes[:, 0].cpu().long()].unsqueeze(1)
        d_visit_next = d_visit_next[od_nodes[:, 1].cpu().long()].unsqueeze(1)
        
        od_dist = torch.tensor([dist_m[o, d] for o, d in zip(o_nodes, d_nodes)], dtype=torch.float).cuda()
        od_dist = od_dist.view(-1, 1)

        cat_visit_evo_emb = torch.cat((all_evo_embedding, 
                                       o_visit, 
                                       d_visit, 
                                       o_visit_next,
                                       d_visit_next,
                                       od_dist), dim=1)
        inferred_edge = self.flow_generator(cat_visit_evo_emb)

        return inferred_edge, attention_weights_snap1, attention_weights_snap2, all_evo_embedding
    