import os
import logging
import numpy as np


class Vocabulary:
    """
    构建词表
    """
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.files = config.files
        self.vocab_path = config.vocab_path
        self.max_vocab_size = config.max_vocab_size
        self.word2id = {}
        self.id2word = None
        self.label2id = config.label2id
        self.id2label = config.id2label

    def __len__(self):
        return len(self.word2id)

    def vocab_size(self):
        return len(self.word2id)

    def label_size(self):
        return len(self.label2id)

    # 获取词的id
    def word_id(self, word):
        return self.word2id[word]

    # 获取id对应的词
    def id_word(self, idx):
        return self.id2word[idx]

    # 获取label的id
    def label_id(self, word):
        return self.label2id[word]

    # 获取id对应的词
    def id_label(self, idx):
        return self.id2label[idx]

    def get_vocab(self):
        """
        函数功能：
            进一步处理，将word和label转化为id
            word2id: dict,每个字对应的序号
            idx2word: dict,每个序号对应的字
        细节：
            如果是第一次运行代码则可直接从65行开始看就行（第一次运行没有处理好的vocab可供直接读取）
            该函数统计样本集中每个字出现的次数，并制作成词表，然后按照出现次数从高到低排列，并给每个字赋予一个唯一的id（出现次数越多id越小）
            
            最终 self.word2id 形如 {"我":1 , "你":2 , "他":3 ...}

        输出：
            保存为二进制文件
        """
        # 如果有处理好的，就直接load
        if os.path.exists(self.vocab_path):
            data = np.load(self.vocab_path, allow_pickle=True)
            # '[()]'将array转化为字典
            self.word2id = data["word2id"][()]
            self.id2word = data["id2word"][()]
            logging.info("-------- Vocabulary Loaded! --------")
            return
        # 如果没有处理好的二进制文件，就处理原始的npz文件
        word_freq = {}
        for file in self.files:
            data = np.load(self.data_dir + str(file) + '.npz', allow_pickle=True)   # 打开之前压缩好的 词-标签 的.npz文件
            word_list = data["words"]        # 读取其中的词列表
            # 常见的单词id最小
            for line in word_list:           # 按行读取
                for ch in line:              # 按字读取  
                    if ch in word_freq:      # 统计每个字出现的频率，并统计在word_freq中
                        word_freq[ch] += 1
                    else:
                        word_freq[ch] = 1
        index = 0
        sorted_word = sorted(word_freq.items(), key=lambda e: e[1], reverse=True)   # 按照字的出现频率降序排列
        # 构建word2id字典
        for elem in sorted_word:
            self.word2id[elem[0]] = index  # 出现频率越高的字出现在越前面
            index += 1
            if index >= self.max_vocab_size:
                break
        # id2word保存
        self.id2word = {_idx: _word for _word, _idx in list(self.word2id.items())}
        # 保存为二进制文件
        np.savez_compressed(self.vocab_path, word2id=self.word2id, id2word=self.id2word)
        logging.info("-------- Vocabulary Build! --------")


if __name__ == "__main__":
    import config
    if os.path.exists(config.vocab_path):
        os.remove(config.vocab_path)
    # 建立词表
    vocab = Vocabulary(config)
    vocab.get_vocab()
    print(len(vocab.word2id))

