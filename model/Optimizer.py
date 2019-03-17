import numpy as np
import logging
import tensorflow as tf


## 模型训练算法

class ADAGRAD(object):
    """
    adagrad模型训练算法
    """

    def __init__(self, params, lr, lr_word_vector=0.1, epsilon=1e-10):
        logging.info('Optimizer ADAGRAD lr %f' % (lr,))
        self.lr = lr
        self.lr_word_vector = lr_word_vector
        self.epsilon = epsilon
        self.acc_grad = {}
        for param in params:
            self.acc_grad[param] = tf.zeros_like(param, dtype=tf.float32)

    def iterate(self, grads):
        """
        迭代
        :param grads:
        :return: 梯度更新值
        """
        lr = self.lr
        epsilon = self.epsilon
        update = {}
        for param, grad in grads.items():
            if param.name == 'one/Vw:0':
                update[param] = param.assign(param - grad * self.lr_word_vector)
            else:
                self.acc_grad[param] = self.acc_grad[param] + grad ** 2
                param_update = lr * grad / (tf.sqrt(self.acc_grad[param]) + epsilon)
                update[param] = param.assign(param - param_update)
        return update


OptimizerList = {'ADAGRAD': ADAGRAD}
