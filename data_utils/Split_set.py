from DataManager import Sentence
import numpy as np


## 数据集切分

def load_data():
    """
    从文件加载数据
    :return:数据列表
    """
    data = []
    with open('data/ch/data_new_j.txt') as f:

        sentences = f.readlines()
        for i in range(int(len(sentences) / 3)):  ##/3得到组数
            try:
                content, target, rating = sentences[i * 3].strip(), sentences[i * 3 + 1].strip(), sentences[
                    ##内容 aspect 极性从标注训练集得到
                    i * 3 + 2].strip()
                sentence = Sentence(content, target, rating, grained=3)
                data.append(sentence)
            except:
                print('input data format wrong')
    return data


def split_train(data, test_ratio):
    """
    划分训练集,测试集
    :param data: 数据
    :param test_ratio:测试集切分比率
    :return: 训练集,测试集
    """
    np.random.seed(43)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return np.array(data)[train_indices], np.array(data)[test_indices]


def write_file(data, name):
    """
    写入文件
    :param data:数据
    :param name: 文件名
    :return:
    """
    with open('data/ch/%s.txt' % name, 'w') as f:
        for i in range(len(data)):
            f.write(data[i].content + '\n')
            f.write(data[i].target + '\n')
            for j in range(0, 3):
                if data[i].solution[j] == 1:
                    f.write(str(j - 1) + '\n')
                    break


if __name__ == '__main__':
    data = load_data()
    train_data, mix_data = split_train(data=data, test_ratio=0.4)
    dev_data, test_data = split_train(data=mix_data, test_ratio=0.5)
    write_file(train_data, 'train')
    write_file(dev_data, 'dev')
    write_file(test_data, 'test')
