import os
os.environ["TORCH_USE_CUDA_DSA"] = "0"
import argparse
import math
import time
import random
import warnings
from tqdm import tqdm
import logging

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import pearsonr, entropy
from scipy.spatial.distance import jensenshannon
from sklearn import metrics


from models import DSEN
from utils import Data_utility
from Optim import Optim

def make_dir(path, dic_name):
    path = os.path.join(path, dic_name)
    is_dir_exist = os.path.exists(path)
    if is_dir_exist:
        print("----Dic existed----")
    else:
        os.mkdir(path)
        print("----Dic created successfully----")
    return path

def find_xy_flow(flow_df,TCMA_cbg):
    TCMA_cbg_center=TCMA_cbg.copy()
    flow_df['start_lat']=None
    flow_df['start_lon']=None
    flow_df['end_lat']=None
    flow_df['end_lon']=None

    for index, row in tqdm(TCMA_cbg_center.iterrows()):
        start_lat=row['y']
        start_lon=row['x']
        end_lat=row['y']
        end_lon=row['x']
        flow_df.loc[flow_df.o_cbg==row['CensusBlockGroup'], 'start_lat']=start_lat
        flow_df.loc[flow_df.o_cbg==row['CensusBlockGroup'], 'start_lon']=start_lon
        flow_df.loc[flow_df.d_cbg==row['CensusBlockGroup'], 'end_lat']=end_lat
        flow_df.loc[flow_df.d_cbg==row['CensusBlockGroup'], 'end_lon']=end_lon

    return flow_df


def js_divergence(p, q):
    # Normalize the arrays to make sure they are probability distributions
    p = np.array(p) / np.sum(p)
    q = np.array(q) / np.sum(q)
    
    # Calculate JS Divergence using scipy
    return jensenshannon(p, q) ** 2

def common_part_of_commuters(generated_flow_flat, real_flow_flat, numerator_only=False):
    """
    The implementation of common_part_of_commuters
    :param generated_flow:
    :param real_flow:
    :param numerator_only:
    :return: The value of CPC
    """
    nonzero_indices = np.nonzero(real_flow_flat)

    filtered_real_flow = real_flow_flat[nonzero_indices]
    filtered_generated_flow = generated_flow_flat[nonzero_indices]

    # Ensure non-negative values for real flow (though in this case it already is)
    filtered_real_flow = np.maximum(filtered_real_flow, 0)

    # 4. Calculate CPC (Common Part of Commuters)
    numerator = 2 * np.sum(np.minimum(filtered_real_flow, filtered_generated_flow))
    denominator = np.sum(filtered_real_flow + filtered_generated_flow)
    cpc = numerator / denominator if denominator != 0 else 0.0

    return cpc


def NRMSE(generated_flow, real_flow):
    MSE = 0.0
    RMSE = 0.0
    NRMSE = 0.0
    for i in range(len((generated_flow))):

        select_indice = np.where(real_flow[i]!=0)[0]

        predictions = generated_flow[i][select_indice].copy()
        actuals = real_flow[i][select_indice].copy()

        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        norm_factor = np.max(actuals) - np.min(actuals)

        if norm_factor == 0:
            NRMSE += 0
        else:
            NRMSE += rmse / norm_factor
    
    return NRMSE/len(generated_flow)

def evaluate(data, 
             X, 
             Y, 
             model, 
             test_save_path,
             batch_size=64 , 
             mode="valid"):
    """
    Evaluating or testing function
    :param data:
    :param X:
    :param Y:
    :param model:
    :param batch_size:
    :return:
    """
    model.eval()

    generated_flow = []
    real_flow = []
    # print(Y.shape)
    for x, y in data.get_batches(X, Y, batch_size):

        x = x.clone().detach().cuda()
        y = y.clone().detach().cuda()
        out, attention_weights_snap1, attention_weights_snap2, evo_embedding = model(data, x)
            
        generated_flow.append(out.cpu().detach().numpy())
        real_flow.append(y.cpu().numpy())
        # print(out)

    generated_flow = np.concatenate(generated_flow)
    real_flow = np.concatenate(real_flow)

    cpc = common_part_of_commuters(generated_flow, real_flow)
    corr,_ = pearsonr(generated_flow, real_flow)
    nrmse = np.sqrt(np.mean((generated_flow - real_flow) ** 2))
    mape = np.mean(np.abs((real_flow - generated_flow) / real_flow)) * 100
    js = js_divergence(generated_flow, real_flow)

    if mode == "test":

        o_index_list = X[:,0]
        d_index_list = X[:,1]

        test_results = pd.DataFrame.from_dict({'o':o_index_list, 'd':d_index_list, 
                                'generated_flow': generated_flow, 'Actual_flow': real_flow})
        test_results.to_csv(test_save_path+'12345/results/generated_flow_test_set.csv', index=False)


    return cpc, corr, nrmse, mape, js

def train(data, 
          X, 
          Y, 
          model, 
          criterion_mse,
          lambda_reg,
          optim, 
          epoch,
          batch_size
          ):
    """
    Training function
    :param data:
    :param X:
    :param Y:
    :param model:
    :param criterion:
    :param optim:
    :param batch_size:
    :return:
    """
    model.train()

    total_loss = 0.0

    # binary_accur = []
    # binary_f1 = []
    generated_flow = []
    real_flow = []
    evo_embedding = []

    last_attention_weights_snap1 = None
    last_attention_weights_snap2 = None
    last_edge_index_snap1 = None
    last_edge_index_snap2 = None

    # print(Y.shape)
    for x, y in data.get_batches(X, Y, batch_size):
        model.zero_grad()
        batch_loss = 0.0
        x = x.clone().detach().cuda()
        y = y.clone().detach().cuda()
        out, attention_weights_snap1, attention_weights_snap2, Batch_evo_embedding = model(data, x)

        edge_index_snap1, attention_weights_snap1 = attention_weights_snap1
        edge_index_snap2, attention_weights_snap2 = attention_weights_snap2

        last_attention_weights_snap1 = attention_weights_snap1.cpu().detach().numpy()
        last_attention_weights_snap2 = attention_weights_snap2.cpu().detach().numpy()
        last_edge_index_snap1 = edge_index_snap1.cpu().detach().numpy()
        last_edge_index_snap2 = edge_index_snap2.cpu().detach().numpy()
        Batch_evo_embedding = Batch_evo_embedding.cpu().detach().numpy()

        batch_loss = criterion_mse(out, y)
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        batch_loss += lambda_reg * l2_norm

        generated_flow.append(out.cpu().detach().numpy())
        real_flow.append(y.cpu().numpy())
        evo_embedding.append(Batch_evo_embedding)
        
        # print('Total loss: ',batch_loss)
        batch_loss.backward()
        grad_norm = optim.step()
        total_loss += batch_loss.cpu().data.numpy()


    generated_flow = np.concatenate(generated_flow)
    real_flow = np.concatenate(real_flow)
    evo_embedding = np.concatenate(evo_embedding)
    # print(evo_embedding.shape)
    # exit()

    print("Generated Flow:", generated_flow)
    print("Real Flow:", real_flow)

    cpc = common_part_of_commuters(generated_flow, real_flow)
    corr,_ = pearsonr(generated_flow, real_flow)
    nrmse = np.sqrt(np.mean((generated_flow - real_flow) ** 2))
    mape = np.mean(np.abs((real_flow - generated_flow) / real_flow)) * 100
    js = js_divergence(generated_flow, real_flow)

    return total_loss, cpc, corr, nrmse, mape, js, last_attention_weights_snap1, last_attention_weights_snap2, last_edge_index_snap1, last_edge_index_snap2, evo_embedding


parser = argparse.ArgumentParser(description='DSEN on Minneapolis')
parser.add_argument('--model', type=str, default='DSEN', help='The model of DSEN')

### hyper-parameters
parser.add_argument('--epochs', type=int, default=1500, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('--seed', type=int, default=12345, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help="The Index of GPU where we want to run the code")
parser.add_argument('--cuda', type=str, default=True)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip', type=float, default=10., help='gradient_visual clipping')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--test_model_name', type=str, default='full_DSEN', help='the name of tested model')


args = parser.parse_args()

path = "/"
path_result = make_dir(path, str(args.seed))
path_log = make_dir(path_result, args.test_model_name+"log")
path_model = make_dir(path_result, "pkl")
path_att_weight = make_dir(path_result, "att_weight")
path_evo_feature = make_dir(path_result, "evo_feature")
path_result = make_dir(path_result, "results")
logging.basicConfig(filename=path_result+'_process.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

print("----Spliting the training and testing data----")
Data = Data_utility('42') # random seed of selecting train, test, validate

# The setting of GPU
args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Prints the whole tensor
torch.set_printoptions(profile='full')


print("----Building models----")
model = eval(args.model).Model(node_num_features = 22, 
                               flow_num_features = 1, 
                               head = 1, 
                               loc_embedding_dim = 24, 
                               evo_emb_dim = 64,
                               evo_emb_hidden_dim = 128, 
                               flow_generator_hidden_dim = 128, 
                               dropout_p= 0.3)


if args.cuda:
    model.cuda()
 
nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('----number of parameters: %d----' % nParams)

criterion_mse = nn.MSELoss()

optim = Optim(model.parameters(), args.optim, args.lr, args.clip)

best_valid_cpc = 0.0
lambda_reg = 0.0

try:
    logging.info('----Traning begin----')
    last_update = 1 
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss, train_cpc, train_corr, train_nrmse, train_mape, train_js, att_w_snap1, att_w_snap2, edge_id_snap1, edge_id_snap2, evo_embed_train = train(Data, 
                                                  Data.train_flow[:, [0,1]], 
                                                  Data.train_flow[:,2], 
                                                  model,
                                                  criterion_mse,
                                                  lambda_reg,
                                                  optim, 
                                                  epoch, 
                                                  args.batch_size)
        
        valid_cpc, valid_corr, valid_nrmse, valid_mape, valid_js = evaluate(Data, 
                                                        Data.valid_flow[:, [0,1]], 
                                                        Data.valid_flow[:, 2], 
                                                        model, 
                                                        path,
                                                        args.batch_size, 
                                                        "valid")

        
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.4f} | train cpc  {:5.4f} | train corr {:5.4f} | train nrmse {:5.4f} | train mape {:5.4f} | train js {:5.4f} | valid cpc {:5.4f} | valid corr {:5.4f} | valid nrmse {:5.4f} | valid mape {:5.4f} | valid js {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), train_loss, train_cpc, train_corr, train_nrmse, train_mape, train_js, valid_cpc, valid_corr, valid_nrmse, valid_mape, valid_js))

        if best_valid_cpc < valid_cpc and valid_cpc < 1:

            # np.save(path+str(args.seed)+'/att_weight/attention_weights_snap1_'+str(epoch), att_w_snap1)
            # np.save(path+str(args.seed)+'/att_weight/attention_weights_snap2_'+str(epoch), att_w_snap2)
            # np.save(path+str(args.seed)+'/att_weight/edge_id_snap1_'+str(epoch), edge_id_snap1)
            # np.save(path+str(args.seed)+'/att_weight/edge_id_snap2_'+str(epoch), edge_id_snap2)

            # np.save(path+str(args.seed)+'/evo_feature/evo_feature_'+str(epoch), evo_embed_train)
            
            logging.info("----epoch:{}, save the model----".format(epoch))
            with open(os.path.join(path_model, args.test_model_name+"model.pkl"), 'wb') as f:
                torch.save(model, f)
            with open(os.path.join(path_log, args.test_model_name+"log.txt"), "a") as file:
                file.write("epoch:{}, save the model.\r\n".format(epoch))
            best_valid_cpc = valid_cpc
            last_update = epoch

        if epoch - last_update == 200:
            with open(os.path.join(path_model, args.test_model_name+"model_last_epoch.pkl"), 'wb') as f:
                torch.save(model, f)
            break

except KeyboardInterrupt:
    logging.info('-' * 90)
    logging.info('----Exiting from training early----')
    if os.path.isfile(os.path.join(path_model, args.test_model_name+"model_early_stop.pkl"))==False:
        with open(os.path.join(path_model, args.test_model_name+"model_early_stop.pkl"), 'wb') as f:
                torch.save(model, f)

logging.info("----Testing begin----")
if os.path.isfile(os.path.join(path_model, args.test_model_name+"model.pkl")):
    with open(os.path.join(path_model, args.test_model_name+"model.pkl"), 'rb') as f:
        model = torch.load(f)
        test_cpc, test_corr, test_nrmse, test_mape, test_js= evaluate(Data, 
                                                   Data.test_flow[:, [0,1]], 
                                                   Data.test_flow[:, 2], 
                                                   model, 
                                                   path,
                                                   args.batch_size, 
                                                   "test")
        logging.info(args)
elif os.path.isfile(os.path.join(path_model, args.test_model_name+"model_last_epoch.pkl")):
    with open(os.path.join(path_model, args.test_model_name+"model_last_epoch.pkl"), 'rb') as f:
        model = torch.load(f)
        test_cpc, test_corr, test_nrmse, test_mape, test_js= evaluate(Data, 
                                                   Data.test_flow[:, [0,1]], 
                                                   Data.test_flow[:,2], 
                                                   model, 
                                                   path,
                                                   args.batch_size, 
                                                   "test")
        logging.info(args)
else:
    with open(os.path.join(path_model, args.test_model_name+"model_early_stop.pkl"), 'rb') as f:
        model = torch.load(f)
        test_cpc, test_corr, test_nrmse, test_mape, test_js= evaluate(Data, 
                                                   Data.test_flow[:, [0,1]], 
                                                   Data.test_flow[:,2], 
                                                   model, 
                                                   path,
                                                   args.batch_size, 
                                                   "test")
        logging.info(args)

logging.info("test cpc {:5.4f} | test corr {:5.4f} | test nrmse {:5.4f} | test mape {:5.4f} | test js {:5.4f}".format(test_cpc, test_corr, test_nrmse, test_mape, test_js))



