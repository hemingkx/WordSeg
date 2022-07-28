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
        for file_name in self.files:            # file_name = train && test
            self.get_examples(file_name)

    def get_examples(self, mode):
        """
        函数功能：为样本数据集中的词组添加BMES标签，并将字的集合和标签的集合保存为二进制文件，方便之后模型训练读取

        参数 ：
            mode 样本数据集的函数名

        返回值：
            将txt文件每一行中的文本分离出来，存储为words列表，将所有行合并为word_list
            BMES标注法标记文本对应的标签，存储为labels，将所有行合并为 label_list
        
        细节：
            本函数中最重要的函数是第74行的getlist，其给每个字添加上了标签，具体细节详见该函数处说明
        """
        input_dir = self.data_dir + str(mode) + '.txt'          # 输入文件 ,self.data_dir = os.getcwd() + '/data/'
        output_dir = self.data_dir + str(mode) + '.npz'         # 输出文件
        if os.path.exists(output_dir) is True:
            return  
        with open(input_dir, 'r', encoding='utf-8') as f:       #打开文件
            word_list = []
            label_list = []
            num = 0
            for line in f:              # 逐行读取文件    举例 "共同  创造  美好  的  新  世纪  ——  二○○一年  新年  贺词"
                num += 1
                words = []
                line = line.strip()  # remove spaces at the beginning and the end
                # print(line)
                if not line:
                    continue  # line is None
                for i in range(len(line)):
                    if line[i] == " ":
                        continue  # skip space
                    words.append(line[i])           # 按字切分句子    words="共同创造美好的新世纪-—二○○一年新年贺词"
                # print(words)
                word_list.append(words)             # word_list 字集
                text = line.split(" ")                              # text=["共同","创造","美好","的","新","世纪","——","二○○一年","新年","贺词"]
                # print(text)
                labels = []
                for item in text:
                    if item == "":
                        continue
                    labels.extend(getlist(item))        # 给训练集中的每行句子中的每个词语添加标签  举例 ： "二○○一年" 对应标签为 "BMMME"
                # print(labels)
                label_list.append(labels)                # label_list 标签集
                assert len(labels) == len(words), "labels 数量与 words 不匹配"
            print("We have", num, "lines in", mode, "file processed")
            # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)      # 将word_list,label_list保存到一个二进制文件中
            """
            np.savez_compressed 对应的读取二进制文件方法详见本网址 https://www.cnblogs.com/wushaogui/p/9142019.html
            """
            logging.info("-------- {} data process DONE!--------".format(mode))
