#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : chenzhikuo
# @Time :  2019/5/30
# @Filename : ndcg.py
import numpy as np

def DCG(label_list):
    dcgsum = 0
    for i in range(len(label_list)):
        dcg = (2**label_list[i] - 1)/np.log2(i+2)
        dcgsum += dcg
    return dcgsum


#ndcg 计算
def ndcg(label_list,top_n):
    #没有设定topn
    if top_n==None:
        dcg = DCG(label_list)
        ideal_list = sorted(label_list, reverse=True)
        ideal_dcg = DCG(ideal_list)
        if ideal_dcg == 0:
            return 0
        return dcg/ideal_dcg
    #设定top n
    else:
        dcg = DCG(label_list[0:top_n])
        ideal_list = sorted(label_list, reverse=True)
        ideal_dcg = DCG(ideal_list[0:top_n])
        if ideal_dcg == 0:
            return 0
        return dcg/ideal_dcg