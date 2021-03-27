import os
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

    def data_process(self):
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
                num += 1
                words = []
                line = line.strip()  # remove spaces at the beginning and the end
                # print(line)
                if not line:
                    continue  # line is None
                for i in range(len(line)):
                    if line[i] == " ":
                        continue  # skip space
                    words.append(line[i])
                # print(words)
                word_list.append(words)
                text = line.split(" ")
                # print(text)
                labels = []
                for item in text:
                    if item == "":
                        continue
                    labels.extend(getlist(item))
                # print(labels)
                label_list.append(labels)
                assert len(labels) == len(words), "labels 数量与 words 不匹配"
            print("We have", num, "lines in", mode, "file processed")
            # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("-------- {} data process DONE!--------".format(mode))
