# Chinese Word Segmentation

本项目为中文分词任务baseline的代码实现，模型包括

- BiLSTM-CRF
- BERT-base + X (softmax/CRF/BiLSTM+CRF)
- Roberta + X (softmax/CRF/BiLSTM+CRF)

本项目是 [CLUENER2020](https://github.com/hemingkx/CLUENER2020) 的拓展项目。

## Dataset

数据集来源于[SIGHAN 2005](http://sighan.cs.uchicago.edu/bakeoff2005/)第二届中文分词任务中的Peking University数据集。

## Model

本项目实现了中文分词任务的baseline模型，对应路径分别为：

- BiLSTM-CRF
- BERT-Softmax
- BERT-CRF
- BERT-LSTM-CRF

其中，根据使用的预训练模型的不同，BERT-base-X 模型可转换为 Roberta-X 模型。

## Requirements

This repo was tested on Python 3.6+ and PyTorch 1.5.1. The main requirements are:

- tqdm
- scikit-learn
- pytorch >= 1.5.1
- 🤗transformers == 2.2.2

To get the environment settled, run:

```
pip install -r requirements.txt
```

## Pretrained Model Required

需要提前下载BERT的预训练模型，包括

- pytorch_model.bin
- vocab.txt

放置在./pretrained_bert_models对应的预训练模型文件夹下，其中

**bert-base-chinese模型：**[下载地址](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) 。

注意，以上下载地址仅提供tensorflow版本，需要根据[huggingface suggest](https://huggingface.co/transformers/converting_tensorflow_models.html)将其转换为pytorch版本。

**chinese_roberta_wwm_large模型：**[下载地址](https://github.com/ymcui/Chinese-BERT-wwm#%E4%BD%BF%E7%94%A8%E5%BB%BA%E8%AE%AE) 。

如果觉得麻烦，pytorch版本的上述模型可以通过下方**网盘链接**直接获取😊：

链接: https://pan.baidu.com/s/1rhleLywF_EuoxB2nmA212w  密码: isc5

## Results

各个模型在数据集上的结果（f1 score）如下表所示：（Roberta均指RoBERTa-wwm-ext-large模型）

| 模型 | BiLSTM+CRF | Roberta+Softmax | Roberta+CRF | Roberta+BiLSTM+CRF |
| ---- | ---------- | --------------- | ----------- | ------------------ |
|      |            |                 |             |                    |

