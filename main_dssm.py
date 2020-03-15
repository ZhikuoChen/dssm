# -*- coding: utf-8 -*-
"""
Created on Sun june 26 19:49:24 2018

@author: Administrator
"""
import os
import time
import numpy as np
import datetime
from logger import get_logger
from train_dssm import train, build_estimator, predict

date = datetime.datetime.today()

now_dir = 'datepart=' + date.strftime('%Y%m%d')

log_file_name = os.path.basename(__file__).split('.', 1)[0] + '.log'

# 当日志文件大小小于1M时，则以
if os.path.exists(log_file_name) is False or os.path.getsize(log_file_name) / 1024 / 1024 < 1:
    logger = get_logger(log_file_name, mode='a')
else:
    # 否则删除以前的日志
    logger = get_logger(log_file_name)

LOCAL_TRAIN_DIR = '/dssm_data/train/'
LOCAL_VALID_DIR = '/dssm_data/valid/'
LOCAL_PREDICT_DIR = '/dssm_data/predict/'
LOCAL_DICT_DIR = '/dssm_data/vocab/'
LOCAL_TRAIN_NUM_DIR = '/dssm_data/train_num/'

LOCAL_MODEL_DIR = '/cdssm/model/'
LOCAL_DSSM_RESULT_DIR = '/cdssm/dssm_result/'

if __name__ == '__main__':
    all_start = time.time()
    logger.info(now_dir)
    local_train_dir = LOCAL_TRAIN_DIR + now_dir
    local_dict_dir = LOCAL_DICT_DIR + now_dir
    local_valid_dir = LOCAL_VALID_DIR + now_dir
    local_predict_dir = LOCAL_PREDICT_DIR + now_dir
    local_train_num_dir = LOCAL_TRAIN_NUM_DIR + now_dir

    model_dir = LOCAL_MODEL_DIR + now_dir + time.strftime("%H%M", time.localtime())
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    local_video_emb_file = '/data/id_vec.pickle'
    
    estimator, train_steps = build_estimator(model_dir, LOCAL_DICT_DIR + now_dir, LOCAL_TRAIN_NUM_DIR + now_dir + "/part-00000", LOCAL_DICT_DIR + now_dir + "/word.txt", local_video_emb_file)

    train_start = time.time()
    train(estimator, LOCAL_TRAIN_DIR + now_dir + "/part-r*", LOCAL_VALID_DIR + now_dir + "/part-r*", train_steps)
    elapsed_train = (time.time() - train_start) / 60 / 60
    logger.info("Train model takes {} hours".format(elapsed_train))

    predict_start = time.time()
    local_dssm_result_dir = LOCAL_DSSM_RESULT_DIR + '{}/'.format(now_dir)
    
    best_dir = model_dir + '/best'
    predict_result_dict = predict(estimator, LOCAL_PREDICT_DIR + now_dir + "/part-r*", best_dir, local_dssm_result_dir)
    elapsed_predict = (time.time() - predict_start) / 60 / 60
    logger.info("Predict takes {} hours".format(elapsed_predict))

    elapsed = (time.time() - all_start) / 60 / 60
    logger.info("The total program takes {} hours".format(elapsed))
