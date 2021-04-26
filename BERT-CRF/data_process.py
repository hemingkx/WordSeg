import os
import re
import config
import logging
import numpy as np


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
        """
        input_dir = self.data_dir + str(mode) + '.txt'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        with open(input_dir, 'r', encoding='utf-8') as f:
            word_list = []
            label_list = []
            num = 0
            for line in f:
                words = []
                line = line.strip()  # remove spaces at the beginning and the end
                if not line:
                    continue  # line is None
                for i in range(len(line)):
                    if line[i] == " ":
                        continue  # skip space
                    words.append(line[i])
                w = "".join(words)
                s = re.split(r"([，。；])", w)
                s = add_sep_word(s, config.sep_word)
                s.append("")
                s = ["".join(i) for i in zip(s[0::2], s[1::2])]
                for w_ in s:
                    if len(w_) > 228 or len(w_) == 0:  # make sure l < 256
                        continue
                    word_list.append(list(w_))
                sl = re.split(r"([，。；])", line)
                sl = add_sep_word(sl, config.sep_word)
                sl.append("")
                sl = ["".join(i) for i in zip(sl[0::2], sl[1::2])]
                for l_ in sl:
                    labels = []
                    text = l_.split(" ")
                    for item in text:
                        if item == "":
                            continue
                        labels.extend(getlist(item))
                    if len(labels) > 228 or len(labels) == 0:
                        continue
                    label_list.append(labels)
                # print(word_list[num])
                # print(label_list[num])
                num += 1
                assert len(word_list) == len(label_list), "labels 数量与 words 不匹配"
            print("We have", num, "lines in", mode, "file processed")
            # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("-------- {} data process DONE!--------".format(mode))


def get_process():
    """处理数据集数据"""
    if os.path.exists(config.train_dir):
        os.remove(config.train_dir)
    if os.path.exists(config.test_dir):
        os.remove(config.test_dir)
    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.process()


def read_file(mode='training'):
    """读取文件并切分句子"""
    input_dir = config.data_dir + str(mode) + '.txt'
    word_list = []
    label_list = []
    len_list = []
    llen_list = []
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
            w = "".join(words)
            s = re.split(r"([，。；])", w)  # ，。、；》
            s = add_sep_word(s, config.sep_word)
            s.append("")
            s = ["".join(i) for i in zip(s[0::2], s[1::2])]
            for w_ in s:
                # if len(w_) > 200:
                # continue
                word_list.append(w_)
                len_list.append(len(w_))
            sl = re.split(r"([，。；])", line)
            sl = add_sep_word(sl, config.sep_word)
            sl.append("")
            sl = ["".join(i) for i in zip(sl[0::2], sl[1::2])]
            for l_ in sl:
                labels = []
                text = l_.split(" ")
                for item in text:
                    if item == "":
                        continue
                    labels.extend(getlist(item))
                # if len(labels) > 200:
                # continue
                label_list.append(labels)
                llen_list.append(len(labels))
    return len_list, llen_list


def get_len(len_type="word", mode='training'):
    """统计句子长度"""
    if len_type == "word":
        len_list, _ = read_file(mode)
    else:
        _, len_list = read_file(mode)
    lens = {'<10': 0, '10-50': 0, '50-100': 0, '100-200': 0, '200-500': 0, '500-1000': 0, '>1000': 0}
    print(len_type, ": ", len(len_list), "sentences in the", mode, "file.")
    for i in len_list:
        if i <= 10:
            lens['<10'] += 1
        if 10 < i <= 50:
            lens['10-50'] += 1
        if 50 < i <= 100:
            lens['50-100'] += 1
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
    """打印句子长度分布"""
    for file_name in config.files:
        lens = get_len(mode=file_name)
        print(lens)
        lens = get_len(len_type="label", mode=file_name)
        print(lens)


def get_sub_list(init_list, sublist_len, sep_word):
    """按长度切分句子"""
    list_groups = zip(*(iter(init_list),) * sublist_len)
    end_list = [list(i) + list(sep_word) for i in list_groups]
    count = len(init_list) % sublist_len
    if count != 0:
        end_list.append(init_list[-count:])
    else:
        end_list[-1] = end_list[-1][:-1]  # remove the last sep word
    return end_list


if __name__ == "__main__":
    get_process()
    # print_len()
