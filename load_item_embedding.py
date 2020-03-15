#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : chenzhikuo
# @Time :  2019/11/13
# @Filename : load_item_embedding.py
import pickle
import numpy as np

def load_item2vec(emb_path, id_to_item, item_dim):
    """
    Load item embedding from pre-trained file
    embedding size must match
    """
    id_embedding = {}
    while True:
        try:
            with open(emb_path, 'rb') as f:
                id_embedding = pickle.load(f)
            if id_embedding:
                break
        except EOFError:
            print('video vec file is empty')
            continue

    n_items = len(id_to_item)
    # 产生随机的词向量，若词在训练好的词向量中，则再代替
    # Get item embeddings for each token in the sentence
    item_embeddings = np.random.uniform(-0.25, 0.25, (n_items, item_dim))
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    # Lookup table initialization
    for i in range(n_items):
        item = id_to_item[i]
        if item in id_embedding:
            item_embeddings[i] = id_embedding[item]
    return item_embeddings
