import numpy as np
import tensorflow as tf


class Sentence(object):
    """docstring for sentence"""

    def __init__(self, content, target, rating, grained):
        self.content, self.target = content.lower(), target  ##小写化
        self.solution = np.zeros(grained, dtype=np.float64)  ##theano.config.floatX
        self.senlength = len(self.content.split(' '))
        try:
            self.solution[int(rating) + 1] = 1  ## 将极性化成one-hot
        except:
            exit()

    def stat(self, target_dict, wordlist, grained=3):
        """

        :param target_dict: 目标词典
        :param wordlist: 词列表
        :param grained: 几分类任务
        :return:
        """
        data, data_target, i = [], [], 0
        solution = np.zeros((self.senlength, grained), dtype=np.float64)  ##theano.config.floatX
        for word in self.content.split(' '):
            data.append(wordlist[word])  ##加入word在wordlist中的下标
            try:
                pol = Lexicons_dict[word]
                solution[i][pol + 1] = 1
            except:
                pass
            i = i + 1
        for word in self.target.split(' '):
            data_target.append(wordlist[word])
        return {'seqs': data, 'target': data_target, 'solution': np.array([self.solution]),
                'target_index': self.get_target(target_dict)}

    def get_target(self, dict_target):
        """
        得到目标人物的index
        :param dict_target:
        :return:
        """
        return dict_target[self.target]


class DataManager(object):
    """
    数据集管理,包括生成词典以及预先选择词向量
    """
    def __init__(self, dataset, grained=3):
        self.fileList = ['train', 'test', 'dev']
        self.origin = {}
        for fname in self.fileList:
            data = []
            with open('%s/ch/%s.txt' % (dataset, fname), encoding="utf8") as f:
                sentences = f.readlines()
                for i in range(int(len(sentences) / 3)):  ##/3得到组数
                    try:
                        content, target, rating = sentences[i * 3].strip(), sentences[i * 3 + 1].strip(), sentences[
                            ##内容 aspect 极性从标注训练集得到
                            i * 3 + 2].strip()
                        sentence = Sentence(content, target, rating, grained)
                        data.append(sentence)
                    except:
                        print('input data format wrong')
            self.origin[fname] = data
        self.gen_target()

    def gen_word(self):
        """
        对词进行编码，不区分维度
        :return: 词列表
        """
        wordcount = {}

        def sta(sentence):
            for word in sentence.content.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1  ##重复计数
                except:
                    wordcount[word] = 1
            # aspect
            for word in sentence.target.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except:
                    wordcount[word] = 1

        for fname in self.fileList:
            for sent in self.origin[fname]:
                sta(sent)
        words = wordcount.items()  ##返回元组数组(key,vaule)
        words = sorted(words, key=lambda x: x[1],
                       reverse=True)  ## sorted(words,key=lambda x: x[1], reverse=True) ##逆序排序words
        self.wordlist = {item[0]: index + 1 for index, item in enumerate(words)}  ##按照words顺序构成list
        return self.wordlist

    def gen_target(self, threshold=5):
        """
        生成词典,过滤低频词
        :param threshold: 过滤低频词的阈值
        :return: 词典
        """
        self.dict_target = {}
        # 统计每种label个数
        for fname in self.fileList:
            for sent in self.origin[fname]:
                if sent.target in self.dict_target:
                    self.dict_target[sent.target] = self.dict_target[sent.target] + 1
                else:
                    self.dict_target[sent.target] = 1
        i = 0
        for (key, val) in self.dict_target.items():
            if val < threshold:
                self.dict_target[key] = 0
            else:
                self.dict_target[key] = i
                i = i + 1
        return self.dict_target

    def gen_data(self, grained=3):
        """
        数据集划分
        :param grained: 分成几类
        :return: 训练集,验证集,测试集
        """
        self.data = {}
        for fname in self.fileList:
            self.data[fname] = []
            for sent in self.origin[fname]:
                self.data[fname].append(sent.stat(self.dict_target, self.wordlist))

        return self.data['train'], self.data['dev'], self.data['test']

    def word2vec_pre_select(self, mdict, word2vec_file_path, save_vec_file_path):
        """
        从word2vec文件中预先选择需要的词向量
        :param mdict: 词典
        :param word2vec_file_path: 原始词向量路径
        :param save_vec_file_path: 保存词向量路径
        :return:
        """
        list_seledted = ['']
        line = ''
        with open(word2vec_file_path) as f:
            for line in f:
                tmp = line.strip().split(' ', 1)
                if mdict.has_key(tmp[0]):
                    list_seledted.append(line.strip())
        list_seledted[0] = str(len(list_seledted) - 1) + ' ' + str(len(line.strip().split()) - 1)
        open(save_vec_file_path, 'w').write('\n'.join(list_seledted))
