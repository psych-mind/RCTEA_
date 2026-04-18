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
    
def cal_index(task, score):
    """get the hits1, hits5, hits10 number for each task"""
    result_dict = {}
    for i in tqdm(range(len(task))):
        ref = task[i]
        #rank = (-score[:, i]).argsort()
        #result_dict[ref] = rank[:100]
        rank = np.argmax(score[:, i])
        result_dict[ref] = np.array([rank])
    return result_dict

def multi_thread_gi(score, nums_threads):
    """
    apply the multi-thread machenism to apply the argsort function in torch
    and return the hits1, hits5 and hits10 number respectively.
    """
    result_dic = {}
    test_size = len(score)
    tasks = div_list(np.array(range(test_size)), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        #print(1)
        reses.append(pool.apply_async(cal_index, (task, score[:, task])))
    pool.close()
    pool.join()
    
    for res in reses:
        #hits1_num, hits5_num, hits10_num, mean = res.get()
        result_dic_p = res.get()
        result_dic.update(result_dic_p)
    del score
    gc.collect()
    return result_dic