# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import math
from load_item_embedding import load_item2vec

def cosine_similarity(a, b):
    with tf.device('/cpu:0'):
        normalize_a = tf.nn.l2_normalize(a, axis=1)
        normalize_b = tf.nn.l2_normalize(b, axis=1)
        # Reduce sum is not deterministic on GPU!
        # https://github.com/tensorflow/tensorflow/issues/3103
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1)
    return cos_similarity

def model_fn(features, labels, mode, params):
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    question = features["question"]
    answer = features["answer"]
    ans_num = features["ans_num"]
    if params["emb_file"]:
        embeddings = load_item2vec(params["emb_file"], params["id_items"], params["word_dim"])
    else:
        # Get word embeddings for each token in the sentence
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,shape=[params["vocab_size"], params['word_dim']],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(params['word_dim'])))
    question_emb = tf.cast(tf.nn.embedding_lookup(embeddings, question), tf.float32)  # shape:(batch, sentence_len, embedding_size)
    answer_emb = tf.cast(tf.nn.embedding_lookup(embeddings, answer), tf.float32)  # shape:(batch, sentence_len, embedding_size)
    #embedding形状为batch_size*self.num_steps*self.word_dim 
    # apply dropout before feed to cnn layer
    conv_que = tf.layers.conv1d(question_emb, filters=params['num_filters'], kernel_size=params['filter_size'], padding='same', name='conv_que')
    conv_ans = tf.layers.conv1d(answer_emb, filters=params['num_filters'], kernel_size=params['filter_size'], padding='same', name='conv_ans')
    conv_que = tf.layers.dropout(conv_que, params['dropout'], training=training)
    conv_ans = tf.layers.dropout(conv_ans, params['dropout'], training=training)
    pool_que = tf.layers.max_pooling1d(conv_que, pool_size=params['length'], strides=params['length'], name='max_pool_que')
    pool_ans = tf.layers.max_pooling1d(conv_ans, pool_size=params['length'], strides=params['length'], name='max_pool_ans')
    flat_que = tf.reshape(pool_que, shape=[-1, params['num_filters']])
    flat_ans = tf.reshape(pool_ans, shape=[-1, params['num_filters']])    
    hidden_sizes = params['hidden_sizes'].split(',')
    num_hidden_layers = len(hidden_sizes)

    if num_hidden_layers == 1:
        with tf.variable_scope("full-connect-%s" % hidden_sizes[0]):
            full_connect_w = tf.get_variable("full_connect_w", shape=[params['num_filters'], int(hidden_sizes[0])], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(params['word_dim'])))
            full_connect_b = tf.get_variable("full_connect_b", shape=[int(hidden_sizes[0])], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
            flat_que = tf.matmul(flat_que, full_connect_w) + full_connect_b
            flat_ans = tf.matmul(flat_ans, full_connect_w) + full_connect_b
    elif num_hidden_layers == 2:
        with tf.variable_scope("first-%s" % hidden_sizes[0]):
            W = tf.get_variable("W", shape=[params['num_filters'], int(hidden_sizes[0])],
                                 dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
            b = tf.get_variable("b", shape=[int(hidden_sizes[0])], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
            flat_que = tf.matmul(flat_que, W) + b
            flat_que = tf.nn.relu(flat_que)
            flat_ans = tf.matmul(flat_ans, W) + b
            flat_ans = tf.nn.relu(flat_ans)
        with tf.variable_scope("last-%s" % hidden_sizes[-1]):
            flat_que = tf.layers.dense(flat_que, units=int(hidden_sizes[-1]), activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
            flat_ans = tf.layers.dense(flat_ans, units=int(hidden_sizes[-1]), activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
            flat_que = tf.layers.dropout(flat_que, params['dropout'], training=training)
            flat_ans = tf.layers.dropout(flat_ans, params['dropout'], training=training)
    out_ = cosine_similarity(flat_que, flat_ans)
    out_ = tf.reshape(out_, [-1, params['neg_num'] + 1])
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=params['learning_rate'], shape=[], dtype=tf.float32)
    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(learning_rate,global_step, params['train_steps'], end_learning_rate=0.0, 
                                              power=1.0,cycle=False)
    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if params['num_warmup_steps']:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(params['num_warmup_steps'], dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = params['learning_rate'] * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
    tf.summary.scalar('lr', learning_rate)
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    if mode == tf.estimator.ModeKeys.PREDICT:
        vocabs = tf.contrib.lookup.index_to_string_table_from_file(params['vocab'])
        que_ans = vocabs.lookup(tf.to_int64(features["que_ans"]))
        predictions = {
            'que_ans': que_ans,
            'output_rank': out_,
            'ans_num': ans_num
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    else:
        # Loss
        # cos_sim的shape为[batch_size, params['neg_num'] + 1]
        cos_sim = out_ * 10

        mask_weight = tf.cast(tf.sequence_mask(ans_num, params['neg_num'] + 1), tf.float32)
        mask_weight = tf.reshape(mask_weight, [-1, params['neg_num'] + 1])
        # cos_sim的shape为[batch_size, params['neg_num'] + 1]  
        cos_sim = tf.multiply(cos_sim, mask_weight)
        # Compute the posterior probability
        posterior_prob = tf.nn.softmax(cos_sim, axis=1)
        # Soft-max of position document
        pos_prob = tf.slice(posterior_prob, [0, 0], [-1, 1])  #取第一列的数据

        with tf.variable_scope('loss'):
            loss = -tf.reduce_sum(tf.log(pos_prob))
            
        train_vars = tf.trainable_variables()
        grads = tf.gradients(loss, train_vars)

        # This is how the model was pre-trained.
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        train_op = optimizer.apply_gradients(zip(grads, train_vars), global_step=global_step)
        
        if mode == tf.estimator.ModeKeys.EVAL:
            # Metrics
            metrics = {
            'AUC': tf.metrics.auc(pos_prob, pos_prob),
            }
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        
        elif mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        		 

        
