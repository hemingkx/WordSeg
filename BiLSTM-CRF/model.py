import torch.nn as nn
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):

    def __init__(self, embedding_size, hidden_size, vocab_size, target_size, num_layers, lstm_drop_out, nn_drop_out):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.nn_drop_out = nn_drop_out
        # nn.Embedding: parameter size (num_words, embedding_dim)
        # for every word id, output a embedding for this word
        # input size: N x W, N is batch size, W is max sentence len
        # output size: (N, W, embedding_dim), embedding all the words
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=lstm_drop_out if num_layers > 1 else 0,
            bidirectional=True
        )
        if nn_drop_out > 0:
            self.dropout = nn.Dropout(nn_drop_out)
        self.classifier = nn.Linear(hidden_size * 2, target_size)
        # https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html
        self.crf = CRF(target_size, batch_first=True)

    def forward(self, unigrams, training=True):
        uni_embeddings = self.embedding(unigrams)   # 将字编码，从而节约存储空间，如 "你"编码为[0.2,0.1]
        sequence_output, _ = self.bilstm(uni_embeddings)        # 使用LSTM模型得到每个字对应四种标签的概率
        if training and self.nn_drop_out > 0:
            sequence_output = self.dropout(sequence_output)
        tag_scores = self.classifier(sequence_output)   # 转换数据维度，因为BiLSTM模型可以是n-m模型，即输入参数维度为n，输出参数维度为m，故需要转换数据维度
        return tag_scores

    def forward_with_crf(self, unigrams, input_mask, input_tags):
        """
        函数功能：
            1. 使用BiLSTM模型计算每个字对应的4个标签的概率 self.forware
            2. 使用crf算法计算损失值 self.crf
        """
        tag_scores = self.forward(unigrams)     # BiLSMT模型，得到每个字对应的每个标签的概率
        loss = self.crf(tag_scores, input_tags, input_mask) * (-1) 
        return tag_scores, loss
