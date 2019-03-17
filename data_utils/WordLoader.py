import numpy as np
from numpy import dtype, fromstring, float32 as REAL


## 加载词

class WordLoader(object):
    """
    加载词向量
    """

    def load_word_vector(self, fname, wordlist, dim, binary=None):
        """
        加载词向量
        :param fname:文件名
        :param wordlist: 词列表
        :param dim: 词向量维度
        :param binary: 是否是二进制文件
        :return: 向量化的语料
        """
        if binary == None:
            if fname.endswith('.txt'):
                binary = False
            elif fname.endswith('.bin'):
                binary = True
            else:
                raise NotImplementedError('Cannot infer binary from %s' % (fname))

        vocab = {}
        with open(fname, encoding="utf8") as fin:
            # header = fin.readline()
            # vocab_size, vec_size = map(int, header.split())
            if binary:
                binary_len = dtype(REAL).itemsize * vec_size
                for line_no in xrange(vocab_size):
                    try:
                        word = []
                        while True:
                            ch = fin.read(1)
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        vocab[str(word)] = fromstring(fin.read(binary_len), dtype=REAL)
                    except:
                        pass
            else:
                for line_no, line in enumerate(fin):
                    try:
                        parts = line.strip().split(' ')
                        if len(parts) != 100 + 1:
                            print("Wrong line: %s %s\n" % (line_no, line))
                        word, weights = parts[0], list(map(REAL, parts[1:]))
                        vocab[str(word)] = weights
                    except:
                        pass
        return vocab
