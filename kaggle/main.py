import argparse
import os
import torch
from src.model import BasicAttention
from src.model import TextSentiment

from src.model import init_weights
from src.model import train
from src.model import evaluate

from src.utils import MyDataset
import math
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
from torchtext import data
from torchtext.data import Iterator, BucketIterator
train_path = 'data/sts-kaggle-train.csv'
valid_path = "data/sts-kaggle-train.csv"
test_path = "data/sts-kaggle-test.csv"
ENC_HID_DIM = 128
train_batch_size = 256



tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False)

train_data = MyDataset(train_path, text_field_a=TEXT,text_field_b=TEXT, label_field=LABEL, test=False, aug=1)
valid_data = MyDataset(valid_path, text_field_a=TEXT,text_field_b=TEXT, label_field=LABEL, test=False, aug=1)
# 因为test没有label,需要指定label_field为None
test_data = MyDataset(test_path, text_field_a=TEXT,text_field_b=TEXT,  label_field=None, test=True, aug=1)





TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))


train_iter, val_iter = BucketIterator.splits(
        (train_data, valid_data), # 构建数据集所需的数据集
        batch_sizes=(16, 16),
        device=0, # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.sentence_a), # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
)
weight_matrix = TEXT.vocab.vectors
model = TextSentiment(weight_matrix,TEXT,300)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
test_iter = Iterator(test_data, batch_size=16, device=0, sort=False, sort_within_batch=False, repeat=False)
for idx, batch in enumerate(train_iter):

    text_a,text_b, label = batch.sentence_a,batch.sentence_b, batch.similarity


N_EPOCHS = 30
CLIP = 1

best_valid_loss = float('inf')

BATCH_SIZE = 128
criterion= nn.CosineEmbeddingLoss()

optimizer = optim.Adam(model.parameters())
for epoch in range(N_EPOCHS):



    train_loss = train(model, train_iter, optimizer, CLIP)





    if train_loss < best_valid_loss:
        best_valid_loss = train_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(epoch)
    print(train_loss)


