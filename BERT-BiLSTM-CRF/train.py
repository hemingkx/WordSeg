import torch
import logging
import torch.nn as nn
from tqdm import tqdm

import config
from model import BertSeg
from metrics import f1_score, bad_case, output_write, output2res


def train_epoch(train_loader, model, optimizer, scheduler, epoch):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_token_starts, batch_labels, _ = batch_samples
        # shift tensors to GPU if available
        batch_data = batch_data.to(config.device)
        batch_token_starts = batch_token_starts.to(config.device)
        batch_labels = batch_labels.to(config.device)
        batch_masks = batch_data.gt(0)  # get padding mask
        # compute model output and loss
        loss = model((batch_data, batch_token_starts),
                     token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
        train_losses += loss.item()
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir, local_rank):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model = BertSeg.from_pretrained(model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch)
        val_metrics = evaluate(dev_loader, model)
        val_f1 = val_metrics['f1']
        logging.info("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_metrics['loss'], val_f1))
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            #  选择一个进程保存
            if local_rank == 0:
                model.module.save_pretrained(model_dir)
                logging.info("--------Save best model!--------")
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val f1: {}".format(best_val_f1))
            break
    logging.info("Training Finished!")


def evaluate(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()

    id2label = config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags, ori_data = batch_samples
            # shift tensors to GPU if available
            batch_data = batch_data.to(config.device)
            batch_token_starts = batch_token_starts.to(config.device)
            batch_tags = batch_tags.to(config.device)
            sent_data.extend(ori_data)
            batch_masks = batch_data.gt(0)  # get padding mask
            label_masks = batch_tags.gt(-1)
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            dev_losses += loss.item()
            # shape: (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            if mode == 'dev':
                batch_output = model.module.crf.decode(batch_output, mask=label_masks)
            else:
                # (batch_size, max_len - padding_label_len)
                batch_output = model.crf.decode(batch_output, mask=label_masks)
            batch_tags = batch_tags.to('cpu').numpy()

            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)
    assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    f1, p, r = f1_score(true_tags, pred_tags)
    metrics['f1'] = f1
    metrics['p'] = p
    metrics['r'] = r
    if mode != 'dev':
        bad_case(sent_data, pred_tags, true_tags)
        output_write(sent_data, pred_tags)
        output2res()
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics
