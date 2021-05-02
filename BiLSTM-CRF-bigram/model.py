import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):

    def __init__(self, embedding_size, hidden_size, vocab_size, target_size, num_layers, lstm_drop_out, nn_drop_out,
                 pretrained_embedding=False, embedding_weight=None):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.nn_drop_out = nn_drop_out
        # nn.Embedding: parameter size (num_words, embedding_dim)
        # for every word id, output a embedding for this word
        # input size: N x W, N is batch size, W is max sentence len
        # output size: (N, W, embedding_dim), embedding all the words
        if pretrained_embedding:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight)
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(
            input_size=embedding_size * 2,
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

    def forward(self, unigrams, bigrams, training=True):
        uni_embeddings = self.embedding(unigrams)
        bi_embeddings = self.embedding(bigrams)
        outputs = torch.cat([uni_embeddings, bi_embeddings], dim=-1)
        sequence_output, _ = self.bilstm(outputs)
        if training and self.nn_drop_out > 0:
            sequence_output = self.dropout(sequence_output)
        tag_scores = self.classifier(sequence_output)
        return tag_scores

    def forward_with_crf(self, unigrams, bigrams, input_mask, input_tags):
        tag_scores = self.forward(unigrams, bigrams)
        loss = self.crf(tag_scores, input_tags, input_mask) * (-1)
        return tag_scores, loss
