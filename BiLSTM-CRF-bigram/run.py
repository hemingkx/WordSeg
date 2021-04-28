import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

import os
import utils
import config
import logging
import numpy as np
from model import BiLSTM_CRF
from embedding import embedding
from data_process import Processor
from Vocabulary import Vocabulary
from data_loader import SegDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from train import train, test
from sklearn.model_selection import train_test_split


def dev_split(dataset_dir):
    """split one dev set without k-fold"""
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
    """train without k-fold"""

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
    word_train, word_dev, label_train, label_dev = dev_split(config.train_dir)
    # simple run without k-fold
    run(word_train, label_train, word_dev, label_dev, vocab, device)


def run(word_train, label_train, word_dev, label_dev, vocab, device, kf_index=0):
    # build dataset
    train_dataset = SegDataset(word_train, label_train, vocab, config.label2id)
    dev_dataset = SegDataset(word_dev, label_dev, vocab, config.label2id)
    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    # get GloVe embedding
    if config.pretrained_embedding:
        embedding_weight = embedding(vocab)
    else:
        embedding_weight = None
    # model
    model = BiLSTM_CRF(embedding_size=config.embedding_size,
                       hidden_size=config.hidden_size,
                       drop_out=config.drop_out,
                       vocab_size=vocab.vocab_size(),
                       target_size=vocab.label_size(),
                       pretrained_embedding=config.pretrained_embedding,
                       embedding_weight=embedding_weight)
    model.to(device)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)
    # how to initialize these parameters elegantly
    for p in model.crf.parameters():
        _ = torch.nn.init.uniform_(p, -1, 1)
    # train and test
    train(train_loader, dev_loader, vocab, model, optimizer, scheduler, device, kf_index)
    with torch.no_grad():
        # test on the final test set
        test_loss, f1 = test(config.test_dir, vocab, device, kf_index)
    return test_loss, f1


if __name__ == '__main__':
    if os.path.exists(config.log_dir):
        os.remove(config.log_dir)
    simple_run()
