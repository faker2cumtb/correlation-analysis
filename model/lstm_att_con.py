import tensorflow as tf
import numpy as np
import argparse
import time
from numpy import random
import collections
from data_utils.WordLoader import WordLoader
from model.Optimizer import OptimizerList
from tensorflow.python.ops import gradients_impl


# import pdb

## 构建模型

class AttentionLstm(object):
    """
    基于注意力机制的lstm
    """

    def __init__(self, wordlist, argv, aspect_num=0):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='lstm')
        parser.add_argument('--rseed', type=int, default=int(1000 * time.time()) % 19491001)
        parser.add_argument('--dim_word', type=int, default=100)
        parser.add_argument('--dim_hidden', type=int, default=100)
        parser.add_argument('--dim_aspect', type=int, default=100)
        parser.add_argument('--grained', type=int, default=3, choices=[3])
        parser.add_argument('--regular', type=float, default=0.0001)  # 0.001
        parser.add_argument('--optimizer', type=str, default='ADAGRAD')
        parser.add_argument('--word_vector', type=str, default='data/ch/ch_vec.txt')
        parser.add_argument('--lr_word_vector', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=0.01)
        args, _ = parser.parse_known_args(argv)
        self.args = args
        self.wordlist = wordlist
        self.name = args.name
        self.dim_word, self.dim_hidden = args.dim_word, args.dim_hidden
        self.dim_aspect = args.dim_aspect
        self.grained = args.grained
        self.regular = args.regular
        self.num = len(wordlist) + 1
        self.aspect_num = aspect_num

        self.init_param()
        self.init_function()

    def init_param(self):
        """
        参数初始化
        :return:
        """

        def shared_matrix(dim, name, u=0, b=0):
            with tf.variable_scope("one", reuse=tf.AUTO_REUSE):
                return tf.get_variable(name=name, initializer=tf.constant_initializer(
                    np.random.uniform(low=-u, high=u, size=dim) + b), shape=dim)

        u = lambda x: 1 / np.sqrt(x)

        dimc, dimh, dima = self.dim_word, self.dim_hidden, self.dim_aspect  # 300,300,300
        dim_lstm_para = dimh + dimc + dima  # 300

        self.Vw = self.load_word_vector(self.args.word_vector, self.wordlist, self.num, dimc)
        self.Wi = shared_matrix((dimh, dim_lstm_para), 'Wi', u(dimh))
        self.Wo = shared_matrix((dimh, dim_lstm_para), 'Wo', u(dimh))
        self.Wf = shared_matrix((dimh, dim_lstm_para), 'Wf', u(dimh))  # 300,900
        self.Wc = shared_matrix((dimh, dim_lstm_para), 'Wc', u(dimh))
        self.bi = shared_matrix((dimh, 1), 'bi', 0.)
        self.bo = shared_matrix((dimh, 1), 'bo', 0.)
        self.bf = shared_matrix((dimh, 1), 'bf', 0.)
        self.bc = shared_matrix((dimh, 1), 'bc', 0.)
        self.Ws = shared_matrix((dimh, self.grained), 'Ws', u(dimh))
        self.bs = shared_matrix((1, self.grained), 'bs', 0.)
        self.h0, self.c0 = np.zeros((1, dimh), dtype=np.float32), np.zeros((1, dimc), dtype=np.float32)
        self.params = [self.Vw, self.Wi, self.Wo, self.Wf, self.Wc, self.bi, self.bo, self.bf, self.bc, self.Ws,
                       self.bs]

        self.Wh = shared_matrix((dimh, dimh), 'Wh', u(dimh))
        self.Wv = shared_matrix((dima, dima), 'Wv', u(dimh))
        self.w = shared_matrix((dimh + dima,), 'w', 0.)
        self.Wp = shared_matrix((dimh, dimh), 'Wp', u(dimh))
        self.Wx = shared_matrix((dimh, dimh), 'Wx', u(dimh))
        self.params.extend([self.Wh, self.Wv, self.w, self.Wp, self.Wx])

        self.Va = shared_matrix((self.aspect_num, dima), 'Va', 0.01)
        self.params.extend([self.Va])
        self.optimizer = OptimizerList[self.args.optimizer](self.params, self.args.lr, self.args.lr_word_vector)

    def init_function(self):
        """
        构建tensorflow图模型,即相关操作和相关张量
        :return:
        """
        self.seq_idx = tf.placeholder(tf.int32, [1, None], name="seq_idx")  # X - tfhe Data
        self.solution = tf.placeholder(tf.int32, [1, None], name="solution")  # Y - tfhe Lables
        self.tar_scalar = tf.placeholder(tf.int32, [1], name='tar_scalar')
        self.len_train_data = tf.placeholder(tf.float32, [1], name="len_train_data")
        # self.seq_matrix = tf.gather(self.Vw, self.seq_idx, axis=0)
        self.tar_vector = tf.gather(self.Va, self.tar_scalar, axis=0)
        num_hidden = self.dim_hidden
        self.h, self.c = tf.zeros((1, self.dim_hidden), dtype=np.float32), tf.zeros((1, self.dim_hidden),
                                                                                    dtype=np.float32)
        self.previous_h_c_tuple = tf.stack([self.h, self.c], axis=0)
        # def encode(x_t, h_fore, c_fore, tar_vec):
        #     v = tf.concatenate([h_fore, x_t, tar_vec]) ##按行拼接
        #     f_t = tf.nnet.sigmoid(tf.dot(self.Wf, v) + self.bf)
        #     i_t = tf.nnet.sigmoid(tf.dot(self.Wi, v) + self.bi)
        #     o_t = tf.nnet.sigmoid(tf.dot(self.Wo, v) + self.bo)
        #     c_next = f_t * c_fore  i_t * tf.tanh(tf.dot(self.Wc, v) + self.bc)
        #     h_next = o_t * tf.tanh(c_next)
        #     return h_next, c_next
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.seq_matrix = tf.transpose(tf.gather(self.Vw, self.seq_idx), perm=[1, 0, 2])

            def encode(previous_h_c_tuple, current_x):
                prev_h, prev_c = tf.unstack(previous_h_c_tuple)
                v = tf.transpose(tf.concat([prev_h, current_x, self.tar_vector], axis=1), perm=[1, 0])
                f_t = tf.transpose(tf.nn.sigmoid(tf.matmul(self.Wf, v) + self.bf), perm=[1, 0])
                i_t = tf.transpose(tf.nn.sigmoid(tf.matmul(self.Wi, v) + self.bi), perm=[1, 0])
                o_t = tf.transpose(tf.nn.sigmoid(tf.matmul(self.Wo, v) + self.bo), perm=[1, 0])
                c_next = f_t * prev_c + i_t * tf.transpose(tf.tanh(tf.matmul(self.Wc, v) + self.bc), perm=[1, 0])
                h_next = o_t * tf.tanh(c_next)
                return tf.stack([h_next, c_next])

            self.scan_result = tf.scan(fn=encode, elems=self.seq_matrix, initializer=self.previous_h_c_tuple,
                                       name='hstates')
            # self.W = tf.get_variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")

            # self.seq_matrix = tf.gather(self.Vw, self.seq_idx, axis=0)
            ## 等于self.seq_matrix = tf.gather(self.Vw, self.seq_idx, axis=0)
            ##self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # self.embedded_chars_expanded = \

            #     tf.expand_dims(self.embedded_chars, -1)
        # self.lstm_cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        # ##self.h_drop_exp = tf.expand_dims(self.h_drop,-1)
        # self.lstm_out, self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell, self.embedded_chars,
        #                                                    dtype=tf.float32)  ##out = c, state = h
        # # embed()5tt
        # scan_result, _ = tf.scan(fn=encode, sequences=[self.seq_matrix], outputs_info=[h, c],
        #                              non_sequences=[self.tar_vector])
        embedding = self.scan_result[:, 0, 0,
                    :]  # embedding in there is a matrix, include[h_1, ..., h_n],[h_1,h_2....h_n]是按列组成的矩阵 每一步的隐层结果
        # attention
        self.embedding = embedding
        matrix_aspect = tf.zeros_like(embedding,
                                      dtype=tf.float32) + self.tar_vector  ##tf.config.floatX)[:, :self.dim_aspect]
        # pdb.set_trace()
        self.matrix_aspect = matrix_aspect
        hhhh = tf.concat([tf.matmul(embedding, self.Wh), tf.matmul(matrix_aspect, self.Wv)], axis=1)  ##y轴拼接,N*600
        self.hhhh = hhhh
        M_tmp = tf.tanh(hhhh)
        self.M_tmp = M_tmp
        # alpha_tmp = tf.nn.softmax(tf.reduce_sum(tf.multiply(M_tmp, self.w), reduction_indices=1))  ## N*600 X 600*N
        # alpha_tmp = tf.expand_dims(alpha_tmp, 0)
        alpha_tmp = tf.nn.softmax(
            tf.transpose(tf.matmul(M_tmp, tf.expand_dims(self.w, 1)), perm=[1, 0]))  ## N*600 X 600*1
        self.alpha_tmp = alpha_tmp
        r = tf.matmul(alpha_tmp, embedding)  # 1,N
        self.r = r
        h_star = tf.tanh(tf.matmul(r, self.Wp) + tf.matmul(tf.expand_dims(embedding[-1], 0), self.Wx))
        self.h_star = h_star
        # h_star = tf.expand_dims(h_star, 0)
        embedding = h_star  # embedding in there is a vector, represent h_n_star\   N
        self.embedding_2 = embedding
        random.seed(self.args.rseed)
        srng = tf.convert_to_tensor(np.asarray(np.random.binomial(size=embedding.shape, p=0.5, n=1), dtype=np.float32))
        # dropout
        embedding_for_train = embedding * srng
        embedding_for_test = embedding * 0.5

        self.pred_for_train = tf.nn.softmax(tf.matmul(embedding_for_train, self.Ws) + self.bs)
        self.pred_for_test = tf.nn.softmax(tf.matmul(embedding_for_test, self.Ws) + self.bs)
        # self.params = tf.trainable_variables()

        self.l2 = sum([tf.reduce_sum(param ** 2) for param in self.params]) - tf.reduce_sum(self.Vw ** 2)
        # print('l2:', self.l2.get_value())
        # print('sum:', sum([tf.sum(param ** 2) for param in self.params]).get_value())
        self.loss_sen = -tf.tensordot(tf.cast(self.solution, dtype=tf.float32), tf.log(self.pred_for_train), axes=2)
        self.loss_l2 = 0.5 * self.l2 * self.regular
        self.loss = self.loss_sen + self.loss_l2

        grads = tf.gradients(self.loss, self.params)
        # grads[0] = gradients_impl._IndexedSlicesToTensor(grads[0])
        # grads[-1] = gradients_impl._IndexedSlicesToTensor(grads[-1])
        self.grads = grads
        self.updates = collections.OrderedDict()
        self.grad = {}
        self.update = {}
        for param, grad in self.grad.items():
            self.update[param] = tf.assign(grad, np.asarray(np.zeros(grad.get_shape().as_list()),
                                                            dtype=np.float32))
        for param, grad in zip(self.params, grads):
            with tf.variable_scope("one", reuse=tf.AUTO_REUSE):
                g = tf.get_variable(
                    initializer=tf.constant_initializer(np.asarray(np.zeros(param.get_shape().as_list()))),
                    shape=param.get_shape().as_list(), name=param.name[:-2])
                self.grad[param] = g
                # self.updates[g] = g.assign(g + grad)
                self.updates[g] = g.assign(g + grad)
        for param, grad in self.grad.items():
            self.grad[param] = tf.assign(grad, (grad / self.len_train_data))
        self.train_op = self.optimizer.iterate(self.grad)

    def func_train(self, sess, sequences, tar_scalar, solution):
        """
        模型训练
        :param sess: tensorflow会话
        :param sequences: 训练语料序列
        :param tar_scalar: 目标人物index
        :param solution: 训练集label
        :return:
        """
        feed_dict = {
            self.seq_idx: np.expand_dims(sequences, axis=0),
            self.solution: solution,
            self.tar_scalar: np.expand_dims(tar_scalar, axis=0),
            self.h: self.h0,
            self.c: self.c0
        }
        loss, loss_sen, loss_l2, _ = sess.run(
            [self.loss, self.loss_sen, self.loss_l2, self.updates], feed_dict=feed_dict
        )
        return loss, loss_sen, loss_l2

    def func_test(self, sess, sequences, tar_scalar):
        """
        模型测试
        :param sess: tensorflow会话
        :param sequences: 测试语料序列
        :param tar_scalar: 目标人物index
        :return:
        """
        feed_dict = {
            self.seq_idx: np.expand_dims(sequences, axis=0),
            self.tar_scalar: np.expand_dims(tar_scalar, axis=0)
        }
        pred_for_test = sess.run(
            [self.pred_for_test], feed_dict=feed_dict)
        return pred_for_test

    def load_word_vector(self, fname, wordlist, num, dimc):
        """
        加载词向量
        :param fname: 词向量所在文件路径
        :param wordlist: 词列表
        :param num: 词数量
        :param dimc: 词向量维度
        :return:
        """
        loader = WordLoader()
        dic = loader.load_word_vector(fname, wordlist, self.dim_word)
        not_found = 0
        Vw = np.random.uniform(low=-0.01, high=0.01, size=(num, dimc))
        for word, index in wordlist.items():
            try:
                Vw[index] = dic[word]
            except:
                not_found += 1
        Vw = Vw.astype(np.float32)
        # Vw = tf.convert_to_tensor(Vw)
        with tf.variable_scope("one", reuse=tf.AUTO_REUSE):
            return tf.get_variable(initializer=tf.constant_initializer(Vw), name='Vw', shape=Vw.shape)
