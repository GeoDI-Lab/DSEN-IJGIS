import os
import random
import multiprocessing as mp

import numpy as np
import pandas as pd
# import geopandas as gpd

import torch
import torch.nn as nn
import torch.utils.data as data
from torch_geometric.data import Data
from tqdm import tqdm

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, folder_name, seed=22222, normalize=2):#, weather= True

        network_data_path = 'data/'

        self.random_seed = seed
        self.folder_name = folder_name

        self.graph_snap1, self.graph_snap2 = self._set_graph(network_data_path)

        self.dis_m = pd.read_csv(network_data_path + 'dist_matrix_cbg_index.csv', index_col=0).values
        
        self.train_flow, self.valid_flow, self.test_flow = self._get_flow(network_data_path)

    
    def _set_graph(self, network_data_path):

        node_snap1 = np.load(network_data_path + 'node_feature_snap1_norm.npy')
        node_snap2 = np.load(network_data_path + 'node_feature_snap2_norm.npy')
        
        x_snap1 = torch.tensor(node_snap1, dtype=torch.float)
        x_snap2 = torch.tensor(node_snap2, dtype=torch.float)

        edge_indices = np.load(network_data_path+'edge_index_dist_adj.npy') 
        edge_weights = np.load(network_data_path+'edge_weight_dist_adj_dis.npy') 


        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        return Data(x=x_snap1, edge_index=edge_index, edge_attr=edge_attr), Data(x=x_snap2, edge_index=edge_index, edge_attr=edge_attr)
    
        
    def _get_flow(self, network_data_path):

        # read already splited data train 0.6, valid 0.2, test 0.2
        train_flow = np.load(network_data_path+'Random_622/'+self.folder_name+'/train_nonzero_flow.npy')
        val_flow = np.load(network_data_path+'Random_622/'+self.folder_name+'/val_nonzero_flow.npy')
        test_flow = np.load(network_data_path+'Random_622/'+self.folder_name+'/test_nonzero_flow.npy')

        all_flow = np.vstack((train_flow, val_flow, test_flow))

        train_flow = torch.tensor(train_flow, dtype=torch.float)
        val_flow = torch.tensor(val_flow, dtype=torch.float)
        test_flow = torch.tensor(test_flow, dtype=torch.float)

        all_flow = torch.tensor(all_flow, dtype=torch.float)

        return train_flow, val_flow, test_flow

    def get_batches(self, inputs, targets, batch_size):
        length = len(inputs)
        start_idx = 0
        # print(len(inputs))
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            # print(excerpt)
            X = inputs[start_idx:end_idx]
            Y = targets[start_idx:end_idx]
            yield X, Y
            start_idx += batch_size

