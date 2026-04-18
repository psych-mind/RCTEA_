#import model_o
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
filename = 'ICEWS05-15/'
#filename = 'YAGO-WIKI50K/'

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

all_pair = np.vstack((train_pair, dev_pair))
all_pair_dic = dict(zip(all_pair[:, 0], all_pair[:, 1]))
all_pair_dic_i = dict(zip(all_pair[:, 1], all_pair[:, 0]))

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

def convert_result(result_dic, result_dic_i, dev_pair):
    """convert the result for the real index"""
    rec = {}
    rec_i = {}
    for index, ranks in result_dic.items():
        index_c = dev_pair[index][0]
        ranks_c = dev_pair[ranks][:, 1]
        rec[index_c] = ranks_c
    for index, ranks in result_dic_i.items():
        index_c = dev_pair[index][1]
        ranks_c = dev_pair[ranks][:, 0]
        rec_i[index_c] = ranks_c
    return rec, rec_i

def gen_acc_cor\
(rel_result_dic, rel_result_dic_i, time_result_dic, time_result_dic_i, all_pair_dic):
    """get the bi-directional result and result aligns with temporal part"""
    rec_set = set()
    rec_acc_set = set()
    for index_y, index_list in rel_result_dic.items():
        #the ranked first entity.
        index_w = index_list[0]
        #3 constraints for the testment.
        if time_result_dic[index_y][0] == index_w and index_y == rel_result_dic_i[index_w][0] and\
        index_y == time_result_dic_i[index_w][0]:
            rec_set.add(index_y)
        #whether it is correct.
        if time_result_dic[index_y][0] == index_w and index_w == all_pair_dic[index_y]:
            rec_acc_set.add(index_y)
    return rec_set, rec_acc_set

def gen_new_seeds(train_pair, select_ent_set, result_dic, select_num=2000):
    """generate the new seeds pairs for complementing the training set"""
    rec = []
    for ent in select_ent_set:
        ent_c = result_dic[ent][0]
        rec.append((ent, ent_c))
    rec = np.array(rec)
    
    #choose random number of the samples
    np.random.seed(42)
    choice_array = np.random.choice(np.arange(0, len(rec), 1, dtype=np.int64), select_num, replace=False)
    rec_ = rec[choice_array]
    return np.vstack((train_pair, rec_)), rec

#the dual aspect of similarity matrices.
model_rel.eval()
with torch.no_grad():
    output, loss = model_rel(inputs)
    output = output.cpu().numpy()
    output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-5)
    output = tf.convert_to_tensor(output)
    sim_r = cal_sims(dev_pair, output)
    score_rs = sinkhorn(sim_r)

model_time.eval()
with torch.no_grad():
    output, loss = model_time(inputs)
    output = output.cpu().numpy()
    output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-5)
    output = tf.convert_to_tensor(output)
    sim_t = cal_sims(dev_pair, output)
    score_ts = sinkhorn(sim_t)

#here, we would like to test further of the sim matrix.
#and specifically, here, we use the iteration manner to form with the dict.
def gen_index_p(score):
    """generate the argmax for the positive order"""
    index_dict = {}
    for i in tqdm(range(len(score))):
        index_m = np.argmax(score[i])
        index_dict[i] = np.array([index_m])
    return index_dict

def gen_index_pi(score):
    """generate the argmax for the positive order"""
    index_dict = {}
    for i in tqdm(range(len(score))):
        index_m = np.argmax(score[:, i])
        index_dict[i] = np.array([index_m])
    return index_dict

#the muti-thread version if needed.
from CSLS_test_g import *
from CSLS_test_gi import *
# rel_result_dic = multi_thread_g(score_rs.numpy(), 25)
# rel_result_dic_i = multi_thread_gi(score_rs.numpy(), 25)

# time_result_dic = multi_thread_g(score_ts.numpy(), 25)
# time_result_dic_i = multi_thread_gi(score_ts.numpy(), 25)

rel_result_dic = gen_index_p(score_rs)
rel_result_dic_i = gen_index_pi(score_rs)
time_result_dic = gen_index_p(score_ts)
time_result_dic_i = gen_index_pi(score_ts)

rel_result_dic, rel_result_dic_i =\
    convert_result(rel_result_dic, rel_result_dic_i, dev_pair)

time_result_dic, time_result_dic_i = \
convert_result(time_result_dic, time_result_dic_i, dev_pair)

rec_set, rec_acc_set = \
gen_acc_cor(rel_result_dic, rel_result_dic_i, \
time_result_dic, time_result_dic_i, all_pair_dic)
        
#get the seeds accuracy.
print(1 - len(rec_set - rec_acc_set) / len(rec_set), len(rec_set))

#generate a selected number of new seeds.
train_pair_new, rec_new = gen_new_seeds(train_pair, rec_set, rel_result_dic, select_num=2000)
print(len(train_pair_new))





