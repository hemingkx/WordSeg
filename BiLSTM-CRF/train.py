import torch
from torch.utils.data import DataLoader

import config
import logging
from data_loader import SegDataset
from metric import f1_score, bad_case, output_write

from tqdm import tqdm
import numpy as np

# 打印完整的numpy array
np.set_printoptions(threshold=np.inf)


def epoch_train(train_loader, model, optimizer, scheduler, device, epoch, kf_index=0):
    """
    函数功能：
        1. 使用BiLSTM模型计算每个字对应四个标签的概率，比如 "我":{"B":0.2 , "E":0.3 , "M":0 , "s":0.5 } model.forward_with_crf
        2. 计算梯度   loss.backward()
        3. 根据梯度更新优化器梯度   optimizer.step();
    
    细节：
        1. 为什么要清空模型、优化器的梯度：因为上一次训练得到的梯度对本次训练没有用处，所以需要清空梯度。
    """
    model.train()       # 还没看懂
    train_loss = 0.0
    for idx, batch_samples in enumerate(tqdm(train_loader)):    # tqdm 将参数装饰为迭代器；enumerate 将一个可遍历的对象组会为一个索引序列
        x, y, mask, lens = batch_samples
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        model.zero_grad()    # 把上一个batch的梯度归零，上一个梯度不能影响一个计算
        tag_scores, loss = model.forward_with_crf(x, mask, y)  # LSTM前置网络计算（计算每个字对应四个标签的概率，并计算损失值）
        train_loss += loss.item()
        # 梯度反传
        loss.backward()     # 计算梯度
        # 优化更新
        optimizer.step()    # 根据梯度，更新优化器参数
        optimizer.zero_grad()   # 清空梯度
    # scheduler
    scheduler.step()
    train_loss = float(train_loss) / len(train_loader)      # 计算平均损失值
    if kf_index == 0:
        logging.info("epoch: {}, train loss: {}".format(epoch, train_loss))
    else:
        logging.info("Kf round: {}, epoch: {}, train loss: {}".format(kf_index, epoch, train_loss))


def train(train_loader, dev_loader, vocab, model, optimizer, scheduler, device, kf_index=0):
    """
    函数功能：
        反复训练模型->验证参数可靠性->调整参数->继续训练模型
    细节：
        epoch_train()：训练模型参数
        dev()：验证模型参数可靠性，计算相关指标
        improve_f1 ：当前模型参数得到的效果与历史最佳参数得到的效果之间的差值，如果improve_f1大于1e-5，代表当前参数比历史最佳参数更好，就应该将当前参数设置为历史最佳参数
    """
    best_val_f1 = 0.0   # 历史最佳模型参数所对应的效率值，越高说明模型越好
    patience_counter = 0    # 无法得到更优参数的连续训练模型次数
    # start training
    for epoch in range(1, config.epoch_num + 1):        # 不停的进行模型训练
        epoch_train(train_loader, model, optimizer, scheduler, device, epoch, kf_index)  #训练模型参数
        # 模型参数验证
        with torch.no_grad():
            # dev loss calculation
            metric = dev(dev_loader, vocab, model, device)  #模型的验证，验证机指标计算（与真实值的偏差）
            val_f1 = metric['f1']       # 当前模型参数的效率值
            dev_loss = metric['loss']   # 当前模型参数的损失值，后续没用到
            if kf_index == 0:
                logging.info("epoch: {}, f1 score: {}, "
                             "dev loss: {}".format(epoch, val_f1, dev_loss))
            else:
                logging.info("Kf round: {}, epoch: {}, f1 score: {}, "
                             "dev loss: {}".format(kf_index, epoch, val_f1, dev_loss))
            improve_f1 = val_f1 - best_val_f1   #best_val_f1：历史最佳参数所得到的效果值 val_f1 当前参数所得到的效果值
            if improve_f1 > 1e-5:       # 当前参数更好，则将历史最佳参数设置为当前参数，并保存模型的参数
                best_val_f1 = val_f1
                if kf_index == 0:
                    torch.save(model, config.model_dir)
                else:
                    torch.save(model, config.exp_dir + "model_{}.pth".format(kf_index))     #模型参数保存
                logging.info("--------Save best model!--------")
                if improve_f1 < config.patience:        # 设置一个阈值5，如果连续训练5次都得不到更好的参数，则认为当前模型参数已达最优质，可以结束训练了
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1
            # Early stopping and logging best f1
            if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num: # 当连续5次训练都没有得到更好的参数 或者 模型训练次数达到上届，停止训练，退出模型
                logging.info("Best val f1: {}".format(best_val_f1))   #满足条件，停止训练
                break
    logging.info("Training Finished!")


def dev(data_loader, vocab, model, device, mode='dev'):
    """test model performance on dev-set"""
    model.eval()
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0
    for idx, batch_samples in enumerate(tqdm(data_loader)):    # 读取验证集数据
        words, labels, masks, lens = batch_samples
        sent_data.extend([[vocab.id2word.get(idx.item()) for i, idx in enumerate(indices) if mask[i] > 0]
                          for (mask, indices) in zip(masks, words)])
        words = words.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        y_pred = model.forward(words, training=False)     #预测验证集
        labels_pred = model.crf.decode(y_pred, mask=masks)
        targets = [itag[:ilen] for itag, ilen in zip(labels.cpu().numpy(), lens)]
        true_tags.extend([[vocab.id2label.get(idx) for idx in indices] for indices in targets])      # 真实标签
        pred_tags.extend([[vocab.id2label.get(idx) for idx in indices] for indices in labels_pred])  # 预测标签
        # 计算梯度
        _, dev_loss = model.forward_with_crf(words, masks, labels)
        dev_losses += dev_loss
    assert len(pred_tags) == len(true_tags)
    assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    f1, p, r = f1_score(true_tags, pred_tags)
    metrics['f1'] = f1      # p和r的几何平均
    metrics['p'] = p        # 准确率
    metrics['r'] = r        # 召回率
    metrics['loss'] = float(dev_losses) / len(data_loader)
    if mode != 'dev':
        bad_case(sent_data, pred_tags, true_tags)
        output_write(sent_data, pred_tags)
    return metrics


def load_model(model_dir, device):
    # Prepare model
    model = torch.load(model_dir)
    model.to(device)
    logging.info("--------Load model from {}--------".format(model_dir))
    return model


def test(dataset_dir, vocab, device, kf_index=0):
    """test model performance on the final test set"""
    data = np.load(dataset_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    # build dataset
    test_dataset = SegDataset(word_test, label_test, vocab, config.label2id)
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    if kf_index == 0:
        model = load_model(config.model_dir, device)
    else:
        model = load_model(config.exp_dir + "model_{}.pth".format(kf_index), device)
    metric = dev(test_loader, vocab, model, device, mode='test')
    f1 = metric['f1']
    p = metric['p']
    r = metric['r']
    test_loss = metric['loss']
    if kf_index == 0:
        logging.info("final test loss: {}, f1 score: {}, precision:{}, recall: {}"
                     .format(test_loss, f1, p, r))
    else:
        logging.info("Kf round: {}, final test loss: {}, f1 score: {}, precision:{}, recall: {}"
                     .format(kf_index, test_loss, f1, p, r))
    return test_loss, f1
