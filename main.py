# coding:utf-8
import numpy as np
import tensorflow as tf
# from tensorflow.python import debug as tfdbg
import argparse
import time
import sys
import json
import random
from Optimizer import OptimizerList
from Evaluator import Evaluators
from DataManager import DataManager
from lstm_att_con import AttentionLstm as Model
from datetime import datetime
import matplotlib.pyplot as plt


## 主程序

def train(model, train_data, epoch_num, batch_size, batch_n, sess):
    """
    训练
    :param model:模型
    :param train_data:训练数据
    :param epoch_num: 迭代次数
    :param batch_size: batch大小
    :param batch_n: batch数量
    :param sess: 会话域
    :return: 损失
    """
    st_time = time.time()
    loss_sum = np.array([0.0, 0.0, 0.0])
    total_nodes = 0
    for batch in range(batch_n):
        print("---%d" % (batch))
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(train_data))
        batch_loss, batch_total_nodes = do_train(model, train_data[start:end], sess)
        loss_sum += batch_loss
        total_nodes += batch_total_nodes

    return loss_sum[0], loss_sum[2]


def do_train(model, train_data, sess):
    """

    :param model:
    :param train_data:
    :param sess:
    :return:
    """
    eps0 = 1e-8
    batch_loss = np.array([0.0, 0.0, 0.0])
    total_nodes = 0
    update = {}

    # for _, grad in model.grad.items():
    #     update[grad] = tf.assign(grad , np.asarray(np.zeros(grad.get_shape().as_list()), \
    #                               dtype=np.float32))
    _ = sess.run(model.update)
    for item in train_data:
        sequences, target, tar_scalar, solution = item['seqs'], item['target'], item['target_index'], item['solution']
        batch_loss += np.array(model.func_train(sess, sequences, tar_scalar, solution))
        print(batch_loss)
        total_nodes += len(solution)
        print(total_nodes)
    # for _, grad in model.grad.items():
    #     update[grad] = tf.assign(grad , grad / float(len(train_data)))
    # _ = sess.run(model.update2, feed_dict={model.len_train_data: np.expand_dims( float(len(train_data)), axis=0)})
    # sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    _ = sess.run(model.train_op, feed_dict={model.len_train_data: np.expand_dims(float(len(train_data)), axis=0)})
    return batch_loss, total_nodes


def test(sess, model, test_data, grained):
    """
    测试
    :param sess: 会话域
    :param model: 模型
    :param test_data: 测试数据
    :param grained:类别个数
    :return: 平均损失,准确率
    """
    evaluator = Evaluators[grained]()
    keys = evaluator.keys()

    def cross(solution, pred):
        return -np.tensordot(solution, np.log(pred)[0], axes=([0, 1], [0, 1]))

    loss = .0
    total_nodes = 0
    correct = {key: np.array([0]) for key in keys}
    wrong = {key: np.array([0]) for key in keys}
    preds = []
    for item in test_data:
        # print '---' * 15
        sequences, target, tar_scalar, solution = item['seqs'], item['target'], item['target_index'], item['solution']
        pred = model.func_test(sess, sequences, tar_scalar)
        # print solution
        # print pred
        loss += cross(solution, pred)
        total_nodes += len(solution)
        result = evaluator.accumulate(solution[-1:], pred[0][-1:])
        preds.append(pred)
    acc = evaluator.statistic()
    return loss / total_nodes, acc


if __name__ == '__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='lstm')
    parser.add_argument('--seed', type=int, default=int(1000 * time.time()))
    parser.add_argument('--dim_hidden', type=int, default=100)  # 默认300
    parser.add_argument('--dim_gram', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--fast', type=int, choices=[0, 1], default=0)
    parser.add_argument('--screen', type=int, choices=[0, 1], default=0)
    parser.add_argument('--grained', type=int, default=3)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=50)
    args, _ = parser.parse_known_args(argv)  ##1

    random.seed(args.seed)
    data = DataManager(args.dataset)  ##数据载入
    wordlist = data.gen_word()
    train_data, dev_data, test_data = data.gen_data(args.grained)
    model = Model(wordlist, argv, len(data.dict_target))
    batch_n = (len(train_data) - 1) / args.batch_size + 1
    # optimizer = OptimizerList[args.optimizer](model.params, args.lr, args.lr_word_vector)
    details = {'loss': [], 'loss_train': [], 'loss_dev': [], 'loss_test': [], \
               'acc_train': [], 'acc_dev': [], 'acc_test': [], 'loss_l2': []}

    print("\r\n" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "###########begin training######");
    print(datetime.now())
    x = []
    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []
    config = tf.ConfigProto(allow_soft_placement=True)  # 恒定使用gpu
    with tf.Session(config=config).as_default() as sess:
        # init
        #       sess = tfdbg.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        for e in range(args.epoch):
            print("******************** Epoch：%d ********************" % e)
            x.append(e)
            print(datetime.now())

            random.shuffle(train_data)
            now = {}
            now['loss'], now['loss_l2'] = train(model, train_data, e, args.batch_size, int(batch_n), sess)

            print('-' * 20 + 'train' + '-' * 20)
            now['loss_train'], now['acc_train'] = test(sess, model, train_data, args.grained)
            print('acc_train：', now['acc_train']['three-way'])
            print('loss_train：', now['loss_train'])

            print('-' * 20 + 'dev' + '-' * 20)
            now['loss_dev'], now['acc_dev'] = test(sess, model, dev_data, args.grained)
            print('acc_dev：', now['acc_dev']['three-way'])
            print('loss_dev：', now['loss_dev'])

            print('-' * 20 + 'test' + '-' * 20)
            now['loss_test'], now['acc_test'] = test(sess, model, test_data, args.grained)
            print('acc_test：', now['acc_test']['three-way'])
            print('loss_test：', now['loss_test'])

            loss_train.append(now['loss_train'])
            loss_test.append(now['loss_test'])
            acc_train.append(now['acc_train']['three-way'])
            acc_test.append(now['acc_test']['three-way'])

            for key, value in now.items():
                details[key].append(value)
            output_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "  :  " + json.dumps(
                details) + "\r\n"
            # print(output_str)
            with open('result/%s.txt' % args.name, 'a') as f:
                f.writelines(output_str)
    loss = plt.figure('loss')
    acc = plt.figure('acc')

    loss_plt = loss.add_subplot(1, 1, 1)
    loss_plt.plot(x, loss_train, color="blue", linestyle="-", label="loss_train")
    loss_plt.plot(x, loss_test, color="red", linestyle="-", label="loss_test")
    loss_plt.legend(loc='upper right')

    acc_plt = acc.add_subplot(1, 1, 1)
    acc_plt.plot(x, acc_train, color="blue", linestyle="-", label="acc_train")
    acc_plt.plot(x, acc_test, color="red", linestyle="-", label="acc_test")
    acc_plt.legend(loc='upper right')

    loss.savefig('result/loss.svg')
    acc.savefig('result/acc.svg')
    with open('result/%s.txt' % args.name, 'a') as f:
        f.writelines("###############end training###########\r\n\r\n")
        f.close()
    print(datetime.now())
    print('******************** END ********************')
