import config
import logging

import torch

import warnings
warnings.filterwarnings('ignore')  # 滤去gensim警告信息

import gensim
from gensim.models import KeyedVectors

import codecs
import numpy as np
from tqdm import tqdm


def load_embedding_manually(path):
    """第一行是embedding数量和维度（792679 300），其余行是 word + embedding"""
    vocab_size, size = 0, 0
    vocab = {}
    vocab["i2w"], vocab["w2i"] = [], {}
    count = 0
    with codecs.open(path, "r", "utf-8") as f:
        first_line = True
        for line in tqdm(f):
            if first_line:
                first_line = False
                vocab_size = int(line.strip().split()[0])
                size = int(line.rstrip().split()[1])
                matrix = np.zeros(shape=(vocab_size, size), dtype=np.float32)
                continue
            vec = line.strip().split()
            if not vocab["w2i"].__contains__(vec[0]):
                vocab["w2i"][vec[0]] = count
                matrix[count, :] = np.array([float(x) for x in vec[1:]])
                count += 1
            if count == 10:
                break
    for w, i in vocab["w2i"].items():
        vocab["i2w"].append(w)
    return matrix, vocab, size, len(vocab["i2w"])


def embedding(vocab):
    # 使用gensim载入word2vec词向量
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        config.embedding_dir, binary=False, encoding='utf-8')
    vocab_size = len(vocab) + 1
    embed_size = config.embedding_size
    weight = torch.zeros(vocab_size, embed_size)
    cnt = 0
    for i in range(len(word2vec_model.index_to_key)):
        try:
            index = vocab.word_id(word2vec_model.index_to_key[i])
        except:
            continue
        cnt += 1
        weight[index, :] = torch.from_numpy(word2vec_model.get_vector(
            vocab.id_word(vocab.word_id(word2vec_model.index_to_key[i]))))
    logging.info("--------Pretrained Embedding Loaded ! ({}/{})--------".format(cnt, len(vocab)))
    return weight


if __name__ == "__main__":
    from data_process import Processor
    from Vocabulary import Vocabulary
    processor = Processor(config)
    processor.data_process()
    # 建立词表
    vocab = Vocabulary(config)
    vocab.get_vocab()
    matrix, emb_vocab, size, l = load_embedding_manually(config.embedding_dir)
    print(emb_vocab['i2w'][4])  # 大
    print(vocab.word_id(emb_vocab['i2w'][4]))  # 15
    w = embedding(vocab)

