# Chinese Word Segmentation

本项目为中文分词任务baseline的代码实现，模型包括

- BiLSTM-CRF
- BERT-base + X (softmax/CRF)
- Roberta + X (softmax/CRF)

本项目是 [CLUENER2020](https://github.com/hemingkx/CLUENER2020) 的拓展项目。

## Dataset

数据集来源于[SIGHAN 2005](http://sighan.cs.uchicago.edu/bakeoff2005/)第二届中文分词任务中的Peking University数据集。

## Model

本项目实现了中文分词任务的baseline模型，对应路径分别为：

- BiLSTM-CRF
- BERT-Softmax
- BERT-CRF

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

|    模型    | BiLSTM+CRF | Roberta+Softmax | Roberta+CRF |
| :--------: | :--------: | :-------------: | :---------: |
|  F1 Score  |   0.916    |    **0.946**    |  **0.946**  |
|   Recall   |   0.918    |      0.948      |  **0.951**  |
| Precision  |   0.913    |    **0.943**    |    0.942    |
|  OOV Rate  |   0.075    |      0.076      |    0.077    |
| OOV Recall |   0.431    |    **0.639**    |    0.636    |
| IV Recall  |   0.957    |      0.974      |  **0.977**  |

## Parameter Setting

### 1.model parameters

在./experiments/seg/config.json中设置了Bert/Roberta模型的基本参数，而在./pretrained_bert_models下的两个预训练文件夹中，config.json除了设置Bert/Roberta的基本参数外，还设置了'X'模型（如LSTM）参数，可根据需要进行更改。

### 2.other parameters

环境路径以及其他超参数在./config.py中进行设置。

## Usage

打开指定模型对应的目录，命令行输入：

```
python run.py
```

模型运行结束后，最优模型和训练log保存在./experiments/路径下。在测试集中的bad case保存在./case/bad_case.txt中。

## Attention

目前，当前模型的train.log已保存在./experiments/路径下，如要重新运行模型，请先将train.log移出当前路径，以免覆盖。