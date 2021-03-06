# Chinese Word Segmentation

本项目为中文分词任务baseline的代码实现，模型包括

- BiLSTM-CRF + pretrained embedding (unigram/unigram+bigram)
- BERT-base + X (softmax/CRF/BiLSTM-CRF)
- Roberta + X (softmax/CRF/BiLSTM-CRF)

本项目是 [CLUENER2020](https://github.com/hemingkx/CLUENER2020) 的拓展项目。

项目说明参考知乎文章：[紧追SOTA！教你用Pytorch上手BERT中文分词！](https://zhuanlan.zhihu.com/p/371842740)

## Dataset

数据集来源于[SIGHAN 2005](http://sighan.cs.uchicago.edu/bakeoff2005/)第二届中文分词任务中的Peking University数据集。

如有需要，这里是PKU数据集及评测脚本的百度网盘地址👇：

链接: https://pan.baidu.com/s/1yZrDSrvAXi-0mCxRp3L-sw  密码: 31qd

将数据集及评测脚本解压后放置在所用模型的./data/路径下即可。

## Model

本项目实现了中文分词任务的baseline模型，对应路径分别为：

- BiLSTM-CRF ( + pretrained embedding)
- BERT-Softmax
- BERT-CRF
- BERT-BiLSTM-CRF

其中，根据使用的预训练模型的不同，BERT-base-X 模型可转换为 RoBERTa-X 模型。

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

## Pretrained Embedding Required

BiLSTM-CRF中使用的unigram-embedding和bigram-embedding均来自[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors) ：

Various Co-occurrence Information$\rightarrow$Character Feature$\rightarrow$Word$\rightarrow$ Character (1-2)$\rightarrow$Context Word Vectors  [百度网盘下载地址](https://pan.baidu.com/s/1eeCS7uD3e_qVN8rPwmXhAw?_at_=1620177721918) 。

关于选用该文件的具体说明参见 [中文字符向量 #18](https://github.com/Embedding/Chinese-Word-Vectors/issues/18) 。

## Results

各个模型在数据集上的结果（f1 score）如下表所示：

> BERT均指Bert-chinese-base模型，RoBERTa均指RoBERTa-wwm-ext-large模型。

|            模型             | F1 Score  |  Recall   | Precision | OOV Rate | OOV Recall | IV Recall |
| :-------------------------: | :-------: | :-------: | :-------: | :------: | :--------: | --------- |
|         BiLSTM+CRF          |   0.927   |   0.922   |   0.933   |  0.058   |   0.622    | 0.940     |
|    BiLSTM+CRF (unigram)     |   0.922   |   0.927   |   0.916   |  0.058   |   0.538    | 0.939     |
| BiLSTM+CRF (unigram+bigram) |   0.945   |   0.947   |   0.944   |  0.058   |   0.649    | 0.962     |
|        BERT+Softmax         | **0.965** |   0.961   |   0.968   |  0.058   |   0.850    | **0.968** |
|          BERT+CRF           | **0.965** | **0.962** | **0.969** |  0.058   |   0.858    | **0.968** |
|       BERT+BiLSTM+CRF       |   0.964   |   0.960   | **0.969** |  0.058   | **0.861**  | 0.966     |
|         RoBERTa+CRF         |   0.962   |   0.960   |   0.965   |  0.058   |   0.827    | **0.968** |

我们在BiLSTM-CRF代码中实现了K折验证，结果如下：

|            Model            | K-fold | Average F1 Score | Average Test Loss |
| :-------------------------: | :----: | :--------------: | :---------------: |
|         BiLSTM+CRF          |   10   |      0.923       |      887.08       |
|    BiLSTM+CRF (unigram)     |   10   |      0.918       |      640.43       |
| BiLSTM+CRF (unigram+bigram) |   10   |    **0.943**     |      506.95       |

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

## Evaluation

本项目可以使用SIGHAN 2005官方脚本计算F1 Score等评测指标。

BiLSTM-CRF需在./data/路径下运行

```
perl scripts/score training_vocab.txt test.txt output.txt > score.txt
```

BERT + X需在./data/路径下运行

```
perl scripts/score training_vocab.txt test.txt res.txt > score.txt
```

## MultiGPU

由于Roberta模型占用显存较多，因此该项目中的BERT部分支持Multi GPU训练。BERT-base可直接设置为单卡训练。

单卡训练指令：

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 run.py
```

多卡训练指令：

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run.py
```

后台训练：

```
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 run.py >/dev/null 2>&1
```

## Attention

#### 1.log保存

目前，当前模型的train.log已保存在./experiments/路径下，如要重新运行模型，请先将train.log移出当前路径，以免覆盖。

#### 2.关于注释

本项目是 [CLUENER2020](https://github.com/hemingkx/CLUENER2020) 的拓展项目，因此有些代码注释没有改过来，对项目运行无任何影响。