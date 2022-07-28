import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

import os
import utils
import config
import logging
import numpy as np
from model import BiLSTM_CRF
from data_process import Processor
from Vocabulary import Vocabulary
from data_loader import SegDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from train import train, test
from sklearn.model_selection import train_test_split


def dev_split(dataset_dir):
    """"
    函数功能： 将数据按照9:1 切分为训练集和测试集

    返回值
        x_train： 训练集——字集合  距离
        y_train:  训练集——标签集合
        x_dev:    测试集——字集合
        y_dev     测试集——标签集合
    """
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev  


def k_fold_run():
    """train with k-fold"""
    # set the logger
    utils.set_logger(config.log_dir)
    # 设置gpu为命令行参数指定的id
    if config.gpu != '':
        device = torch.device(f"cuda:{config.gpu}")
    else:
        device = torch.device("cpu")
    logging.info("device: {}".format(device))
    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.data_process()
    # 建立词表
    vocab = Vocabulary(config)
    vocab.get_vocab()
    # 分离出验证集
    data = np.load(config.train_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    kf = KFold(n_splits=config.n_split)
    kf_data = kf.split(words, labels)
    kf_index = 0
    total_test_loss = 0
    total_f1 = 0
    for train_index, dev_index in kf_data:
        kf_index += 1
        word_train = words[train_index]
        label_train = labels[train_index]
        word_dev = words[dev_index]
        label_dev = labels[dev_index]
        test_loss, f1 = run(word_train, label_train, word_dev, label_dev, vocab, device, kf_index)
        total_test_loss += test_loss
        total_f1 += f1
    average_test_loss = float(total_test_loss) / config.n_split
    average_f1 = float(total_f1) / config.n_split
    logging.info("Average test loss: {} , average f1 score: {}".format(average_test_loss, average_f1))


def simple_run():
    """
    函数功能：
        对数据进行预处理，建立样本集和验证集，并将所有的字符找到其对应的标签，建立一一对应关系

    细节：
        data_process(): 为样本数据集中的词组添加BMES标签
        get_vocab(): 构建词表 self.word2id & self.id2word，具体详见函数解释
        dev_split()：按照9:1比例切分样本集和验证集
        run():运行模型、训练、测试
    """
    # set the logger
    utils.set_logger(config.log_dir)
    # 设置gpu为命令行参数指定的id
    if config.gpu != '':
        device = torch.device(f"cuda:{config.gpu}")
    else:
        device = torch.device("cpu")             # 用cpu跑模型
    logging.info("device: {}".format(device))
    # 处理数据，分离文本和标签
    processor = Processor(config)               # 找到数据集
    processor.data_process()                    # 给样本集添加BMES标签
    # 建立词表
    vocab = Vocabulary(config)                  # key=id(0-500) value="word"
    vocab.get_vocab()                           # 构建词表 self.word2id & self.id2word，具体详见函数解释
    # 分离出验证集
    word_train, word_dev, label_train, label_dev = dev_split(config.train_dir)  # 参数为训练集
    # simple run without k-fold
    run(word_train, label_train, word_dev, label_dev, vocab, device)    # 运行训练集和测试集


def run(word_train, label_train, word_dev, label_dev, vocab, device, kf_index=0):
    """
    函数功能： 
        1. 建立样本集和测试集的迭代器  train_loader / dev_loader
        2. 建立模型并映射到cpu设备上  module
        3. 建立优化器 optimizer
        4. 建立调整学习率的方法 scheduler
        5. 归一化处理crf模型中的参数  model.crf.parameters
        6. 训练、验证模型，并调整参数 train
        7. 测试最终的模型   test
    """
    # 测试集   验证集  词表 设备
    # build dataset
    train_dataset = SegDataset(word_train, label_train, vocab, config.label2id) # 训练集，包含了字列表和标签列表，这两个列表一一对应
    dev_dataset = SegDataset(word_dev, label_dev, vocab, config.label2id)       # 验证集
    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,      # 创造一个迭代器，方便接下来的模型迭代的访问训练集
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,          # 创造一个迭代器，方便接下来的模型迭代的访问验证集
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    # model
    model = BiLSTM_CRF(embedding_size=config.embedding_size,    # 初始化模型
                       hidden_size=config.hidden_size,
                       vocab_size=vocab.vocab_size(),
                       target_size=vocab.label_size(),
                       num_layers=config.lstm_layers,
                       lstm_drop_out=config.lstm_drop_out,
                       nn_drop_out=config.nn_drop_out)
    model.to(device)        # 将模型加载到CPU中
    # 初始化模型，把模型传入cpu
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas) #optimizer是一个优化器，可以保存当前的参数，并根据计算得到的梯度来更新参数
        # 关于优化器可以参考 https://blog.csdn.net/KGzhang/article/details/77479737
    scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma) #用于等间隔调整学习率的方法，每三个epoch调整一次学习率,调整倍数gamma是0，5；学习率衰减，参数变化幅度变小，便于收敛，
        # 关于学习率调整函数可以参考 https://zhuanlan.zhihu.com/p/69411064
    # how to initialize these parameters elegantly
    for p in model.crf.parameters():            # 归一化crf模型中的参数，将参数的值放缩到[-1,1]之间
        _ = torch.nn.init.uniform_(p, -1, 1)
    # train and test
    train(train_loader, dev_loader, vocab, model, optimizer, scheduler, device, kf_index)   # 模型训练、验证
    with torch.no_grad():
        # test on the final test set
        test_loss, f1 = test(config.test_dir, vocab, device, kf_index)
    return test_loss, f1


if __name__ == '__main__':
    if os.path.exists(config.log_dir):
        os.remove(config.log_dir)
    simple_run()
    # k_fold_run()
