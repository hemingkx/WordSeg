import os

data_dir = os.getcwd() + '/data/'
train_dir = data_dir + 'training.npz'
test_dir = data_dir + 'test.npz'
files = ['training', 'test']
vocab_path = data_dir + 'vocab.npz'
exp_dir = os.getcwd() + '/experiments/'
model_dir = exp_dir + 'model.pth'
log_dir = exp_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'
output_dir = data_dir + 'output.txt'

max_vocab_size = 1000000

n_split = 5
dev_split_size = 0.1
batch_size = 32
embedding_size = 128
hidden_size = 384
drop_out = 0.5
lr = 0.001
betas = (0.9, 0.999)
lr_step = 5
lr_gamma = 0.8

epoch_num = 30
min_epoch_num = 5
patience = 0.0002
patience_num = 5

gpu = '1'

# B：分词头部 M：分词词中 E：分词词尾 S：独立成词
label2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

id2label = {_id: _label for _label, _id in list(label2id.items())}
