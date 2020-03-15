# -*- coding=utf-8 -*-
import json
import logging
import os
import tensorflow as tf
from dssm import model_fn
from best_checkpoint_copier import BestCheckpointCopier
from logger import get_logger
from ndcg import ndcg

flags = tf.app.flags
# configurations for training
flags.DEFINE_integer("batch_size", 4096, "batch size")
flags.DEFINE_integer("shuffle_buffer_size", 20000, "dataset shuffle buffer size")  # 只影响取数据的随机性
flags.DEFINE_integer("num_parallel_calls", 4, "Num of cpu cores")
flags.DEFINE_integer("num_parallel_readers", 8, "Number of files read at the same time")
flags.DEFINE_integer("length", 1, "length of sequence")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate")

# configurations for the model
flags.DEFINE_float("dropout", 0.2, "Dropout rate")  # 以0.2的概率drop out
flags.DEFINE_integer("num_epoches", 4, "num of epoches")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for.")
flags.DEFINE_integer("num_filters", 64, "Number of convolution filters")
flags.DEFINE_integer("filter_size", 3, "Convolution kernel size")
flags.DEFINE_string("hidden_sizes", '60,40', "hidden sizes")
flags.DEFINE_integer("neg_num", 10, "negtive num")
flags.DEFINE_integer("word_dim", 100, "Embedding size for characters")

FLAGS = tf.app.flags.FLAGS
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.learning_rate > 0, "learning rate must larger than zero"

log_file_name = os.path.basename(__file__).split('.', 1)[0] + '.log'

os.environ["CUDA_VISIBLE_DEVICES"] = "6" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示warning 和 error
tf.logging.set_verbosity(logging.INFO)
# Save params
# 当日志文件大小小于5M时，则以追加模式写
if os.path.exists(log_file_name) is False or os.path.getsize(log_file_name) / 1024 / 1024 < 5:
    logger = get_logger(log_file_name, mode='a')
else:
    # 否则删除以前的日志
    logger = get_logger(log_file_name)


def write_content(data_list, target_dir):
    FILE_STEP = 100000
    rows = len(data_list)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for j in range(0, rows, FILE_STEP):
        if len(str(j // FILE_STEP)) == 1:
            target_file = '00000' + str(j // FILE_STEP) + '_0'
        elif len(str(j // FILE_STEP)) == 2:
            target_file = '0000' + str(j // FILE_STEP) + '_0'
        elif len(str(j // FILE_STEP)) == 3:
            target_file = '000' + str(j // FILE_STEP) + '_0'
        write_path = os.path.join(target_dir, target_file)
        with open(write_path, 'w', encoding='utf8') as f_w:
            for content in data_list[j:j + FILE_STEP]:
                f_w.write(content + '\n')

def input_fn(filenames, shuffle_buffer_size):
    def parser(record):
        keys_to_features = {
            "question": tf.FixedLenFeature([FLAGS.neg_num + 1], tf.int64),
            "answer": tf.FixedLenFeature([FLAGS.neg_num + 1], tf.int64),
            "ans_num": tf.FixedLenFeature([1], tf.int64)}
        parsed = tf.parse_single_example(record, keys_to_features)
        # print("question:{}, answer:{},  ans_num:{}\n".format(parsed['question'], parsed['answer'], parsed['ans_num']))
        concated = tf.concat([parsed['question'], parsed['answer']], axis=0)
        return {"question": parsed['question'], "answer": parsed['answer'], "ans_num": parsed["ans_num"],
                'que_ans': concated}

    # Load txt file, one example per line
    files = tf.data.Dataset.list_files(filenames)  # A dataset of all files matching a pattern.
    dataset = files.apply(
            tf.data.experimental.parallel_interleave(lambda filename: tf.data.TFRecordDataset(filename, compression_type = "GZIP"), cycle_length=FLAGS.num_parallel_readers))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(FLAGS.num_epoches)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=parser, batch_size=FLAGS.batch_size,
                                                               num_parallel_calls=FLAGS.num_parallel_calls))
    return dataset


def build_estimator(model_dir, input_vocab_dir, num_train_file, output_vocab_file, item_emb_file):
    id_items = {}
    list_file = os.listdir(input_vocab_dir)
    for file in list_file:
        if file.endswith('.json'):
            path = os.path.join(input_vocab_dir, file)
            with open(path, 'r', encoding='utf8') as f_dict:
                for line in f_dict:
                    line = line.strip()
                    json1 = json.loads(line)
                    id_items[int(json1['num'])] = json1['item_id']

    vocab_size = len(id_items)
    logger.info("There are {} videos in data sets".format(vocab_size))
    with open(num_train_file, 'r', encoding='utf8') as f_num:
        for line in f_num:
            train_num = line.strip()
    logger.info("The number of training sets is {}".format(train_num))
    train_steps = int(int(train_num) / FLAGS.batch_size * FLAGS.num_epoches)
    logger.info("Train step size is {}".format(train_steps))

    steps_check = int(train_steps/5)
    sorted_id_items = sorted(id_items.items(), key=lambda x: x[0])  # 对字典按键从小到大排序
    # Get paths for vocabularies and dataset
    with open(output_vocab_file, 'w', encoding='utf8') as f:
        for item in sorted_id_items:
            f.write(item[1] + '\n')
    session_config = tf.ConfigProto(log_device_placement=True)
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=steps_check, session_config=session_config,
                                        keep_checkpoint_max=3)
    num_warmup_steps = int(train_steps * FLAGS.warmup_proportion)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=run_config,
        params={
            'vocab_size': vocab_size,
            'num_filters': FLAGS.num_filters,
            'filter_size': FLAGS.filter_size,
            'learning_rate': FLAGS.learning_rate,
            'dropout': FLAGS.dropout,
            'word_dim': FLAGS.word_dim,
            'hidden_sizes': FLAGS.hidden_sizes,
            'neg_num': FLAGS.neg_num,
            'length': FLAGS.length,
            'vocab': output_vocab_file,
            'num_warmup_steps': num_warmup_steps,
            'train_steps': train_steps,
            'emb_file': item_emb_file,
            'id_items': id_items
        })

    return estimator, train_steps

def train(estimator, train_path, validate_path, train_steps):
    input_fn_for_train = lambda: input_fn(train_path, FLAGS.shuffle_buffer_size)
    # max_steps是决定训练结束的参数
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn_for_train, max_steps=train_steps)
    input_fn_for_eval = lambda: input_fn(validate_path, 0)
    # throttle_secs值越大，相邻两次评估的时间越长
    best_copier = BestCheckpointCopier(name='best',  # directory within model directory to copy checkpoints to
                                       checkpoints_to_keep=1,  # number of checkpoints to keep
                                       score_metric='loss',  # metric to use to determine "best"
                                       compare_fn=lambda x, y: x.score < y.score,
                                       sort_reverse=True)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=20, exporters=best_copier)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    logger.info("after train and evaluate")

def predict(estimator, test_path, best_dir, target_dir):
    input_fn_for_test = lambda: input_fn(test_path, 0)
    output_results = estimator.predict(input_fn_for_test, checkpoint_path=tf.train.latest_checkpoint(best_dir))
    predict_result_dict = {}
    true_result_dict = {}
    for output_result in output_results:
        for i in range(output_result['ans_num'][0]):
            question = output_result['que_ans'][i].decode('utf8')
            answer = output_result['que_ans'][i + FLAGS.neg_num + 1].decode('utf8')
            if question not in true_result_dict:
                true_result_dict[question] = {}
            if i == 0:
                true_result_dict[question][answer] = 1
            else:
                true_result_dict[question][answer] = 0

            if question not in predict_result_dict:
                predict_result_dict[question] = {}
            predict_result_dict[question][answer] = float(output_result['output_rank'][i])

    logger.info(best_dir)
    
    total_item_result = []
    for master_key in predict_result_dict:
        complete_string = master_key + '\x01'
        temp_string = ''
        # 按照值的大小进行排序
        sorted_final_sim_dict = sorted(predict_result_dict[master_key].items(), reverse=True, key=lambda x: x[1])
        for item in sorted_final_sim_dict:
            if item[0] in complete_string: # 从相似item中排除掉其本身
                continue
            temp_string += item[0] + '_' + str(item[1]) + ','
        if temp_string != '':
            complete_string += temp_string.rstrip(',')
            total_item_result.append(complete_string)

    write_content(total_item_result, target_dir) # 写入dssm的相似结果

    total_ndcg = 0
    for key in predict_result_dict:
        label_list = []
        sorted_answer_score = sorted(predict_result_dict[key].items(), reverse=True, key=lambda x: x[1])[:5]  # 取前5个预测值最大的结果
        for item in sorted_answer_score:
            assert item[0] in true_result_dict[key]
            label_list.append(true_result_dict[key][item[0]])
        total_ndcg += ndcg(label_list, top_n=5)
    logger.info('Average ndcg@5 is {}'.format(total_ndcg / len(predict_result_dict)))
    return predict_result_dict
