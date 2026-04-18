#this script, we build with the part of the model for the general training 
#neighbourhood consensus one and the combined one.

#here, we first train with the relation and temporal model, followed by the neighbourhood
#consensus and then the joint training.

import model_o
import model_r
import model_t

import utils_n
import numpy as np
from tqdm import *
import torch
import torch.nn.functional as F
import pandas as pd
import time
import tensorflow as tf
from config_ import cfg

from seu_tkg import sinkhorn, cal_sims
from CSLS_test_4 import *

print(torch.cuda.is_available())
print(torch.cuda.current_device())

import os
file_path = os.getcwd() + '/dataset/'
#filename = 'YAGO-WIKI180K/'
#filename = 'ICEWS05-15/'
filename = 'YAGO-WIKI50K/'

#get with the args
cfg = cfg()
cfg.get_args()
cfgs = cfg.update_train_configs()
#set_seed(cfgs.random_seed)

ts = time.time()

#load with data and more
train_pair, dev_pair, adj_matrix, adj_features, rel_features,\
 time_features, time_features_int, radj, time_int_dict_i = \
 utils_n.load_data(
    file_path + filename, train_ratio=2000)

print(train_pair.shape, dev_pair.shape)

if (0, 0) in time_int_dict_i:
    print(time_int_dict_i[(0, 0)])

adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
time_matrix, time_val = np.stack(time_features.nonzero(), axis=1), time_features.data
time_int_matrix, time_int_val = np.stack(time_features_int.nonzero(), axis=1),\
 time_features_int.data

node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
time_size = time_features.shape[1]
time_int_size = time_features_int.shape[1]
ent_size = 0

triple_size = len(adj_matrix)  # not triple size, but number of diff(h, t)
eval_epoch = 3
node_hidden = 50
rel_hidden = 50
#time_hidden = int(node_hidden / 2)
time_hidden = 50
batch_size = 512
dropout_rate = 0.3
lr = 0.005
gamma = 1
depth = 2
device = 'cuda:0'
#print(rel_size)

training_time = 0.
grid_search_time = 0.
time_encode_time = 0.

inputs = [adj_matrix, train_pair]

#after the general loading, we train with the relation and temporal model in the following:
model_rel = model_r.OverAll(node_size=node_size, node_hidden=node_hidden, 
                        rel_size=rel_size, rel_hidden=rel_hidden,
                        rel_matrix=rel_matrix, rel_val=rel_val, \
                        ent_matrix=ent_matrix, ent_val=ent_val, args=cfgs,
                        triple_size=triple_size, dropout_rate=dropout_rate,
                        depth=depth, device=device)
model_rel = model_rel.to(device)
#load with the relation model.
print('finish rel')

opt = torch.optim.RMSprop(model_rel.parameters(), lr=lr, weight_decay=0)
print('model constructed')

epoch_r = 25

start = time.time()
tic = time.time()
for i in trange(epoch_r):
    np.random.shuffle(train_pair)
    for pairs in [train_pair[i * batch_size:(i + 1) * batch_size] for i in
                    range(len(train_pair) // batch_size + 1)]:
        inputs = [adj_matrix, pairs]
        output_ent, loss_ent = model_rel(inputs)
        #loss_ent = align_loss(pairs, output_ent, node_size)
        print(loss_ent)
        loss_ent.backward(retain_graph=True)
        #loss_tem.backward()
        opt.step()
        opt.zero_grad()

te = time.time()
print(f'rel train time {te - ts}')

epoch_t = 60

ts = time.time()
model_time = \
    model_t.OverAll(node_size=node_size, node_hidden=node_hidden, time_hidden=time_hidden,
                rel_size=rel_size, rel_hidden=rel_hidden,
                time_size=time_size, time_int_size=time_int_size,
                time_matrix=time_matrix, time_val=time_val,
                time_int_matrix=time_int_matrix, time_int_val=time_int_val, args=cfgs,
                triple_size=triple_size, dropout_rate=dropout_rate,
                depth=depth, device=device)
model_time = model_time.to(device)
print('finish time')

opt_t = torch.optim.RMSprop(model_time.parameters(), lr=lr, weight_decay=0)
print('model constructed')

for i in trange(epoch_t):
    np.random.shuffle(train_pair)
    for pairs in [train_pair[i * batch_size:(i + 1) * batch_size] for i in
                    range(len(train_pair) // batch_size + 1)]:
        inputs = [adj_matrix, pairs]
        output_time, loss_time = model_time(inputs)
        #loss_ent = align_loss(pairs, output_ent, node_size)
        print(loss_time)
        loss_time.backward(retain_graph=True)
        #loss_tem.backward()
        opt_t.step()
        opt_t.zero_grad()

te = time.time()
print(f'time train time {te - ts}')
#load_path = '/home/jiayun/Desktop/faiss_test_rel/model_save/'

import gc
del opt, opt_t, output_ent, loss_ent, output_time, loss_time

gc.collect()
torch.cuda.empty_cache()

# Optional: aggressively delete lingering CUDA tensors
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.is_cuda:
            del obj
    except:
        pass

gc.collect()
torch.cuda.empty_cache()

#then form with the general neighbourhood consensus.
#then we engineer the input and the model of the neighbourhood consensus part.
# indices_dic_2 = dict(zip(np.arange(187987, 187987 + 187977, 1), np.arange(0, 187977, 1)))
# indices_dic_1 = dict(zip(np.arange(0, 187987, 1), np.arange(0, 187987, 1)))

all_pair = np.vstack((train_pair, dev_pair))
print(all_pair.shape)
indices_dic_2 = dict(zip(all_pair[:, 1], np.arange(0, len(all_pair), 1)))
indices_dic_1 = dict(zip(all_pair[:, 0], np.arange(0, len(all_pair), 1)))

def load_triples_(file_name):
    """load the  triples for the specific file"""
    triples = []
    for line in open(file_name, 'r'):
        params = line.split()
        if len(params) == 5:
            head = params[0]
            rel = params[1]
            tail = params[2]
            ts = params[3]
            te = params[4]
        else:
            head = params[0]
            rel = params[1]
            tail = params[2]
            ts = params[3]
            te = 0 
        triples.append([head, rel, tail, ts, te])
    return triples

def convert_triple(triples):
    """convert the triple representation from string to int"""
    rec = []
    for triple in triples:
        head, rel, tail, ts, te = int(triple[0]), \
        int(triple[1]), int(triple[2]), int(triple[3]), int(triple[4])
        rec.append([head, rel, tail, ts, te])
    return rec

triples_1 = load_triples_(file_path + filename + 'triples_1')
triples_2 = load_triples_(file_path + filename + 'triples_2')
new_triples_1 = convert_triple(triples_1)
new_triples_2 = convert_triple(triples_2)

def gen_edge_index(triples, dict=None):
    """generate with the edge indices for the KG"""
    rec = []
    for triple in triples:
        head, rel, tail, ts, te = triple
        if dict is None:
            rec.append([head, tail])
        else:
            if head not in dict or tail not in dict:
                continue
            rec.append([dict[head], dict[tail]])
    edge_tensor = torch.LongTensor(rec)
    edge_tensor = torch.vstack((edge_tensor[:, 0], edge_tensor[:, 1]))
    return edge_tensor

# edge_tensor_1 = gen_edge_index(new_triples_1)
# edge_tensor_2 = gen_edge_index(new_triples_2, indices_dic_2)
# edge_tensor_1 = edge_tensor_1.to(device)
# edge_tensor_2 = edge_tensor_2.to(device)

edge_tensor_1 = gen_edge_index(new_triples_1, indices_dic_1)
edge_tensor_2 = gen_edge_index(new_triples_2, indices_dic_2)
edge_tensor_1 = edge_tensor_1.to(device)
edge_tensor_2 = edge_tensor_2.to(device)

# def reform_pair(pairs, dict):
#     """this function reform with the training pairs or testing pairs"""
#     rec = []
#     for pair in pairs:
#         rec.append([pair[0], dict[pair[1]]])
#     new_pair = torch.LongTensor(rec)
#     new_pair = torch.vstack((new_pair[:, 0], new_pair[:, 1]))
#     return new_pair

def reform_pair(pairs, dict_1, dict_2):
    """this function reform with the training pairs or testing pairs"""
    rec = []
    for pair in pairs:
        rec.append([dict_1[pair[0]], dict_2[pair[1]]])
    new_pair = torch.LongTensor(rec)
    new_pair = torch.vstack((new_pair[:, 0], new_pair[:, 1]))
    return new_pair

train_pair_n = reform_pair(train_pair, indices_dic_1, indices_dic_2)
dev_pair_n = reform_pair(dev_pair, indices_dic_1, indices_dic_2)

# train_pair_n = reform_pair(train_pair, indices_dic_2)
# dev_pair_n = reform_pair(dev_pair, indices_dic_2)
train_pair_n = train_pair_n.to(device)
dev_pair_n = dev_pair_n.to(device)
print(train_pair_n.shape, dev_pair_n.shape)

from models import RelCNN
from dgl_dual import *

import os.path as osp
class Args:
    def __init__(self):
        self.category = 'zh_en'  # Replace 'default_category' with your value
        self.dim = 300
        self.rnd_dim = 32
        self.num_layers = 3
        self.num_steps = 5
        self.k = 15
    
model_rel.set_att_gate(att=False, gate=True)
model_rel.return_loss = False
model_time.return_loss = False
# Instantiate the arguments
args = Args()

psi_1 = model_time

psi_2 = model_rel

psi_3 = RelCNN(args.rnd_dim, args.rnd_dim, args.num_layers, batch_norm=False,
            cat=True, lin=True, dropout=0.0)

model = DGMC(psi_1, psi_2, psi_3, num_steps=None, k=args.k).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# N_s = 187987
N_s = len(all_pair)

print('Optimize initial feature matching...')
model.num_steps = 0
#ts = time.time()

#the overall parts of training.
for epoch in tqdm(range(1, 91)):
    if epoch == 31:
        print('Refine correspondence matrix...')
        model.num_steps = args.num_steps
        model.detach = True

    #loss = train()
    model.train()
    select_index = np.arange(2000, 187987, 1)
    np.random.shuffle(select_index)
    select_index = torch.LongTensor(select_index).to(device)

    selection_index = \
    torch.hstack((train_pair_n[0, :], select_index))

    optimizer.zero_grad()

    _, S_L = model(inputs, indices_dic_1, indices_dic_2, None, edge_tensor_1, \
                None, None, None, edge_tensor_2, None, None, selection_index, y=train_pair_n)
    
    loss = model.loss(S_L, train_pair_n)

    loss.backward()
    optimizer.step()
    print(loss)

#then we form with the general model and train with overall part.
model_over = model_o.OverAll(node_size=node_size, node_hidden=node_hidden, time_hidden=time_hidden,
                        rel_size=rel_size, rel_hidden=rel_hidden,
                        time_size=time_size, time_int_size=time_int_size,
                        rel_matrix=rel_matrix, rel_val=rel_val, \
                        ent_matrix=ent_matrix, ent_val=ent_val,
                        time_matrix=time_matrix, time_val=time_val, out_dim=4,
                        time_int_matrix=time_int_matrix, time_int_val=time_int_val, args=cfgs,
                        triple_size=triple_size, dropout_rate=dropout_rate,
                        depth=depth, device=device)

model_over = model_over.to(device)

#here, we build with the model state dict and load into the general model.
param_rel = model_rel.state_dict()
param_time = model_time.state_dict()
param_overall = model_over.state_dict()

for key, value in param_rel.items():
    if key in param_overall:
        param_overall[key] = value

for key, value in param_time.items():
    if key in param_overall and key != 'ent_emb' and key != 'rel_emb':
        param_overall[key] = value

#then we load the overall part of model.
model_over.load_state_dict(param_overall)

model_over.set_att_gate(att=False, gate=True)

print('begin')

opt = torch.optim.RMSprop(model_over.parameters(), lr=lr, weight_decay=0)
print('model constructed')

epoch = 5
#model_over.use_mlp = True

start = time.time()
tic = time.time()
for i in trange(epoch):
    if i == 2 and filename[-5:-1] == '180K':
        model_over.use_mlp = True
    np.random.shuffle(train_pair)
    for pairs in [train_pair[i * batch_size:(i + 1) * batch_size] for i in
                    range(len(train_pair) // batch_size + 1)]:
        inputs = [adj_matrix, pairs]
        output_ent, loss_ent = model_over(inputs)
        #loss_ent = align_loss(pairs, output_ent, node_size)
        print(loss_ent)
        loss_ent.backward(retain_graph=True)
        #loss_tem.backward()
        opt.step()
        opt.zero_grad()

#the test, requires memory for 180K dataset.
# model_over.eval()
# with torch.no_grad():
#     output, loss = model_over(inputs)
#     output = output.cpu().numpy()
#     output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-5)
#     output = tf.convert_to_tensor(output)
#     sim = cal_sims(dev_pair, output)
#     score_s = sinkhorn(sim)
#     print(multi_thread_cal_(score_s.numpy(), 20, [1, 5, 10]))








