from collections import Counter
import tqdm

import numpy as np
import scipy.sparse as sp
import scipy
#import tensorflow as tf
import os
import multiprocessing
from torch import Tensor
import torch

from collections import Counter
import torch

import numpy as np
from scipy.special import softmax

def view2(x):
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))

def view3(x: Tensor) -> Tensor:
    if x.dim() == 3:
        return x
    return x.view(1, x.size(0), -1)

def view_back(M):
    return view3(M) if M.dim() == 2 else view2(M)

def cosine_sim(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def load_alignment_pair(file_name):
    alignment_pair = []
    for line in open(file_name, 'r'):
        e1, e2 = line.split()
        alignment_pair.append((int(e1), int(e2)))
    return alignment_pair

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([])
    time_int = set([])
    time = set([])
    for line in open(file_name, 'r'):
        para = line.split()
        if len(para) == 5:
            head, r, tail, ts, te = [int(item) for item in para]
            if ts != 0 and te == 0:
                te = ts
            if te != 0 and ts == 0:
                ts = te
            t_int = (ts, te)
            entity.add(head)
            entity.add(tail)
            rel.add(r)
            time_int.add(t_int)
            time.add(ts)
            time.add(te)
            triples.append([head, r, tail, ts, te])
        else:
            head, r, tail, ts = [int(item) for item in para]
            te = 0
            if ts != 0 and te == 0:
                te = ts
            if te != 0 and ts == 0:
                ts = te
            t_int = (ts, te)
            entity.add(head)
            entity.add(tail)
            rel.add(r)
            time_int.add(t_int)
            time.add(ts)
            time.add(te)
            triples.append([head, r, tail, ts, te])
    return entity, rel, triples, time_int, time

#gen temporal appearance dict.
def gen_temporal_app(triples):
    """generate the temporal appearance dict specific entity {ent: {tem: times}}"""
    rec = {}
    rec_int = {}
    rec_tem_count = {}
    rec_tem_int_count = {}
    rec_tem_unique = {}
    for triple in triples:
        head, rel, tail, ts, te = triple
        if head not in rec:
            rec[head] = []
            rec_int[head] = []
            rec[head].append(ts)
            rec[head].append(te)
            rec_int[head].append((ts, te))
        else:
            rec[head].append(ts)
            rec[head].append(te)
            rec_int[head].append((ts, te))
        if tail not in rec:
            rec[tail] = []
            rec_int[tail] = []
            rec[tail].append(ts)
            rec[tail].append(te)
            rec_int[tail].append((ts, te))
        else:
            rec[tail].append(ts)
            rec[tail].append(te)
            rec_int[tail].append((ts, te))
    for ent, tem_list in rec.items():
        rec_tem_count[ent] = Counter(tem_list)
    for ent, tem_list in rec_int.items():
        rec_tem_int_count[ent] = Counter(tem_list)
    for ent, count_dict in rec_tem_count.items():
        rec_tem_unique[ent] = len(count_dict)
    return rec, rec_tem_count, rec_tem_int_count, rec_tem_unique

def get_matrix(triples, entity, rel, time, time_int, ent_tem_time_count,\
                ent_tem_int_time_count, tem_valid_set, tem_int_valid_set):
    ent_size = max(entity) + 1 + 1
    #here, for the relation, we add with one extra one.
    rel_size = max(rel) + 1 + 1
    time_size = max(time) + 1
    time_int_size = len(time_int)
    print(ent_size, rel_size, time_size, time_int_size)
    #time_int_dict = dict(zip(np.arange(time_size), time_int))
    time_int_dict_i = dict(zip(time_int, np.arange(time_int_size)))
    if (0, 0) in time_int_dict_i:
        zero_index = time_int_dict_i[(0, 0)]
    
    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    radj = []
    #in relation feature
    rel_in = np.zeros((ent_size, rel_size))
    #out relation feature
    rel_out = np.zeros((ent_size, rel_size))
    #only one feature for the relation
    #rel_f = np.zeros((ent_size, rel_size))
    
    time_link = np.zeros((ent_size, time_size))
    time_int_link = np.zeros((ent_size, time_int_size))
    #time_link = np.zeros((ent_size, time_size))  # new adding
    #time_link_ = np.zeros((ent_size, time_size))
    #adj features with the self entry
    for i in range(max(entity) + 2):
        adj_features[i, i] = 1

    #generate the temproal feature.
    for i in range(len(ent_tem_time_count)):
        if i not in ent_tem_time_count:
            time_link[i][0] = 1
            continue
        tem_times_rec = ent_tem_time_count[i]
        #total number of term
        #time_sum = sum(tem_times_rec.values())
        for j in tem_times_rec:
            #number of a specific temporal
            time_j = tem_times_rec[j]
            tf = np.log(1 + time_j)
            #time_link[i][j] = tf
            time_link[i][j] = tf if j in tem_valid_set else 0.0
    
    #generate the temporal interval feature.
    for i in range(len(ent_tem_int_time_count)):
        if i not in ent_tem_int_time_count:
            time_int_link[i][zero_index] = 1
            continue
        tem_times_rec = ent_tem_int_time_count[i]
        for j in tem_times_rec:
            #number of a specific temproal interval.
            time_j = tem_times_rec[j]
            tf = np.log(1 + time_j)
            #time_int_link[i][time_int_dict_i[j]] = tf
            time_int_link[i][time_int_dict_i[j]] = tf if j in tem_int_valid_set else 0.0

    # 先进行判断，说明数据集中要么都是时间点，要么都是区间，后续可能需要改
    if len(triples[0]) < 5:
        for h, r, t, tau in triples:
            adj_matrix[h, t] = 1;
            adj_matrix[t, h] = 1
            adj_features[h, t] = 1;
            adj_features[t, h] = 1
            radj.append([h, t, r, tau]);
            radj.append([t, h, r+rel_size, tau])
            time_link[h][tau] += 1;
            time_link[t][tau] += 1
            rel_in[h][r] += 1;
            rel_out[t][r] += 1
    else:
        for h, r, t, ts, te in tqdm.tqdm(triples):
            time_index = time_int_dict_i[(ts, te)]
            adj_matrix[h, t] = 1;
            adj_matrix[t, h] = 1
            adj_features[h, t] += 1;
            adj_features[t, h] += 1
            adj_features[h, -1] += 1
            adj_features[t, -1] += 1
            radj.append([h, t, r, time_index]);
            radj.append([t, h, r+rel_size, time_index])
            rel_in[h][r] += 1
            rel_out[t][r] += 1
            #here, for each available triple, we add with one more dimension for rel
            rel_in[h][max(rel) + 1] += 1
            rel_out[t][max(rel) + 1] += 1

    radj.append([max(entity) + 1, max(entity) + 1, 0, 0])

    adj_matrix[max(entity)+1, max(entity)+1] = 1
    #here, is the log feature normalize with the temporal part.
    time_features = time_link
    time_int_features = time_int_link
    time_features =\
     time_features / (np.sum(time_features, axis=-1).reshape(-1, 1) + 0.01)
    time_int_features =\
     time_int_features / (np.sum(time_int_features, axis=-1).reshape(-1, 1) + 0.01)

    #time_features = sp.lil_matrix(time_features)
    time_features = sp.coo_matrix(time_features)
    time_int_features = sp.coo_matrix(time_int_features)    

    #rel_feature with the inverse relation
    rel_features = np.concatenate([rel_in, rel_out], axis=1)
    #here, we do with log norm size of the rel feature.
    rel_features =  np.log(1 + rel_features)
    #the log norm.
    rel_features = rel_features / (np.sum(rel_features, axis=-1).reshape(-1, 1) + 0.01)
    rel_features = sp.coo_matrix(rel_features)

    #the normalized adj matrix with self entry
    # Convert to COO for easy access to data
    adj_coo = adj_features.tocoo()

    # Apply log(1 + x) only to the data values.
    new_data = np.log1p(adj_coo.data)
    # Create a new sparse matrix with the transformed data
    adj_features = sp.coo_matrix((new_data, (adj_coo.row, adj_coo.col)), shape=adj_coo.shape)

    # Optional: convert back to LIL format
    adj_features = adj_features.tolil()
    
    adj_features = adj_features / (np.sum(adj_features, axis=-1).reshape(-1, 1) + 0.01)
    adj_features = sp.coo_matrix(adj_features)
    #adj_features = normalize_adj(adj_features)
    #rel_features = normalize_adj(sp.lil_matrix(rel_features))
    #the output: the adj_matrix, r_index: (head_tail_index, rel), r_val: value for head_tail
    #t_index: (head_tail_index, time), ent_feature, rel_feature, time_feature.
    return \
        adj_matrix, adj_features, rel_features, time_features, time_int_features, radj, time_int_dict_i

def load_data(lang, train_ratio=0.3, unsup=False):
    entity1, rel1, triples1, time_int1, time1 = load_triples(lang + 'triples_1')
    entity2, rel2, triples2, time_int2, time2 = load_triples(lang + 'triples_2')
    #the temporal processing.
    
    ent_tem_dict, ent_tem_time_count, ent_tem_int_time_count, \
    ent_tem_unique_count = gen_temporal_app(triples1+triples2)
    tem_valid_set = time1 & time2
    tem_int_valid_set = time_int1 & time_int2
    #tem_num_stat = gen_tem_stat(triples1+triples2, tem_valid_set)
    #tem_num_stat = None
    # modified here #

    if lang[-5:-1] == '180K':
        #train_pair = load_alignment_pair(lang + 'sup_pairs')
        train_pair = []
    else:
        train_pair = load_alignment_pair(lang + 'sup_pairs')

    dev_pair = load_alignment_pair(lang + 'ref_pairs')
    all_pair = train_pair + dev_pair
    #if train_ratio < 0.25:
    #train_ratio = int(len(train_pair) * train_ratio)
    dev_pair = all_pair[train_ratio:]
    train_pair = all_pair[:train_ratio]
    print(lang[-5:-1])
    print(len(train_pair))
    #if unsup:
    #    dev_pair = train_pair + dev_pair
    #    train_pair = load_alignment_pair(lang + 'unsup_link')
    
    #for the get_matrix, modify the adj_features and adj matrix part.
    adj_matrix, adj_features, rel_features, time_features, time_int_features, radj, time_int_dict_i= \
        get_matrix(triples1 + triples2, \
                   entity1.union(entity2), rel1.union(rel2), time1.union(time2), time_int1.union(time_int2),\
                      ent_tem_time_count, ent_tem_int_time_count, tem_valid_set, tem_int_valid_set)

    return np.array(train_pair), np.array(dev_pair), \
    adj_matrix, adj_features, rel_features, time_features, time_int_features, radj, time_int_dict_i

