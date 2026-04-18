import multiprocessing

import gc
import os

import numpy as np
import time
from tqdm import *

from scipy.spatial.distance import cdist

g = 1000000000

def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return
    
    
def cal_index(task, score, top_k):
    """get the hits1, hits5, hits10 number for each task"""
    num = [0 for k in top_k]
    #num_l = [0 for k in top_k]
    #num_h = [0 for k in top_k]
    #mrr_n = 0
    #mrr_l = 0
    #mrr_h = 0
#     count = 0 
    mrr = 0
    mean = 0
    for i in tqdm(range(len(task))):
        #print(i)
        #the score index
        ref = task[i]
        
        #print(i)
        #rank = torch.argsort(score[i], descending=True)
        rank = (-score[i]).argsort()
        #score_ = score[i][rank]
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
        
    return num, mean, mrr


def multi_thread_cal_(score, nums_threads, top_k):
    """
    apply the multi-thread machenism to apply the argsort function in torch
    and return the hits1, hits5 and hits10 number respectively.
    """
    #score = np.matmul(Lvec, Rvec.T)
    hits_ = [0 for k in top_k]
    t_mrr = 0
    total_mean = 0
    test_size = len(score)
    tasks = div_list(np.array(range(test_size)), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        #print(1)
        reses.append(pool.apply_async(cal_index, (task, score[task, :], top_k)))
    pool.close()
    pool.join()
    
    for res in reses:
        #hits1_num, hits5_num, hits10_num, mean = res.get()
        hits_num, mean, mrr = res.get()
        hits_ += np.array(hits_num)
        total_mean += mean
        t_mrr += mrr
    acc = np.array(hits_)/test_size * 100
    total_mean /= test_size
    t_mrr /= test_size
    del score
    gc.collect()
    return hits_, acc, total_mean, t_mrr
