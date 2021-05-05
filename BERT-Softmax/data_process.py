import os
import re
import config
import logging
import numpy as np


def getlist(input_str):
    """
    将每个输入词转换为BMES标注
    """
    output_str = []
    if len(input_str) == 1:
        output_str.append('S')
    elif len(input_str) == 2:
        output_str = ['B', 'E']
    else:
        M_num = len(input_str) - 2
        M_list = ['M'] * M_num
        output_str.append('B')
        output_str.extend(M_list)
        output_str.append('E')
    return output_str


class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.files = config.files

    def process(self):
        for file_name in self.files:
            self.get_examples(file_name)

    def get_examples(self, mode):
        """
        将txt文件每一行中的文本分离出来，存储为words列表
        BMES标注法标记文本对应的标签，存储为labels
        若长度超过max_len，则直接按最大长度切分（可替换为按标点切分）
        """
        input_dir = self.data_dir + str(mode) + '.txt'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        with open(input_dir, 'r', encoding='utf-8') as f:
            word_list = []
            label_list = []
            num = 0
            sep_num = 0
            for line in f:
                words = []
                line = line.strip()  # remove spaces at the beginning and the end
                if not line:
                    continue  # line is None
                for i in range(len(line)):
                    if line[i] == " ":
                        continue  # skip space
                    words.append(line[i])
                text = line.split(" ")
                labels = []
                for item in text:
                    if item == "":
                        continue
                    labels.extend(getlist(item))
                if len(words) > config.max_len:
                    # 直接按最大长度切分
                    sub_word_list = get_sub_list(words, config.max_len - 5, config.sep_word)
                    sub_label_list = get_sub_list(labels, config.max_len - 5, config.sep_label)
                    word_list.extend(sub_word_list)
                    label_list.extend(sub_label_list)
                    sep_num += 1
                else:
                    word_list.append(words)
                    label_list.append(labels)
                num += 1
                assert len(labels) == len(words), "labels 数量与 words 不匹配"
                assert len(word_list) == len(label_list), "label 句子数量与 word 句子数量不匹配"
            print("We have", num, "lines in", mode, "file processed")
            print("We have", sep_num, "lines in", mode, "file get sep processed")
            # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("-------- {} data process DONE!--------".format(mode))


def get_process():
    if os.path.exists(config.train_dir):
        os.remove(config.train_dir)
    if os.path.exists(config.test_dir):
        os.remove(config.test_dir)
    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.process()


def read_file(mode='training'):
    """读取文件"""
    input_dir = config.data_dir + str(mode) + '.txt'
    word_list = []
    len_list = []
    with open(input_dir, 'r', encoding='utf-8') as f:
        for line in f:
            words = []
            line = line.strip()  # remove spaces at the beginning and the end
            if not line:
                continue  # line is None
            for i in range(len(line)):
                if line[i] == " ":
                    continue  # skip space
                words.append(line[i])
            if len(words) > config.max_len:
                sub_word_list = get_sep_list(words, config.sep_word)
                # sub_word_list = get_sub_list(words, config.max_len - 5, config.sep_word)
                for wl in sub_word_list:
                    if len(wl) > config.max_len or len(wl) == 0:
                        continue
                    word_list.append(wl)
                    len_list.append(len(wl))
            else:
                word_list.append(words)
                len_list.append(len(words))
    return len_list, word_list


def get_len(mode='training'):
    """统计处理后的句子长度"""
    len_list, word_list = read_file(mode)
    lens = {'<100': 0, '100-200': 0, '200-500': 0, '500-1000': 0, '>1000': 0}
    print(len(len_list), "sentences in the", mode, "file.")
    for i in len_list:
        if i <= 100:
            lens['<100'] += 1
        elif 100 < i <= 200:
            lens['100-200'] += 1
        elif 200 < i <= 500:
            lens['200-500'] += 1  # 94 sentences' len > 256 in test.txt
        elif 500 < i <= 1000:
            lens['500-1000'] += 1
        elif i > 1000:
            lens['>1000'] += 1
    return lens


def print_len():
    """打印统计的句子长度"""
    for file_name in config.files:
        lens = get_len(file_name)
        print(lens)


def get_sub_list(init_list, sublist_len, sep_word):
    """直接按最大长度切分"""
    list_groups = zip(*(iter(init_list),) * sublist_len)
    end_list = [list(i) + list(sep_word) for i in list_groups]
    count = len(init_list) % sublist_len
    if count != 0:
        end_list.append(init_list[-count:])
    else:
        end_list[-1] = end_list[-1][:-1]  # remove the last sep word
    return end_list


def add_sep_word(s_, sep_word):
    """add sep word to string"""
    new = []
    for i, item in enumerate(s_):
        if item == "，" or item == "。" or item == "；":
            if i == len(s_)-2:
                if s_[-1] == '':
                    new.append(item)
                    continue
            item += sep_word
        new.append(item)
    s_ = new
    return s_


def get_sep_list(init_list, sep_word):
    """按标点切分"""
    w = "".join(init_list)
    s = re.split(r"([。])", w)
    s = add_sep_word(s, sep_word)
    s.append("")
    s = ["".join(i) for i in zip(s[0::2], s[1::2])]
    r = []
    for sub_list in s:
        r.append(list(sub_list))
    return r


if __name__ == "__main__":
    get_process()
    # print_len()
