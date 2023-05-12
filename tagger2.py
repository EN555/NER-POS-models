import torch
from preprocess_ner_pos import *
import torch.nn as nn
from torch import optim
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import random
import numpy as np
import torchtext
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
import os
from typing import List
import torch.utils.data as data
import re

class simple_mlp(nn.Module):
    def __init__(self, batch_size, vocab_size, embedding_dim, hidden_dim, num_labels):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim*5, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        embeddings = self.word_embeddings(x)
        flattened = embeddings.view(embeddings.size(0), -1)
        tanh_out = self.dropout(torch.tanh(self.linear1(flattened)))
        return F.softmax(self.linear2(tanh_out), dim=1)

class InitDataset(data.Dataset):
    def __init__(self, data, vocab, label):
        self.data = data
        self.tokenizer = lambda x: x.split(" ")
        self.vocab = vocab
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        unk_token = self.vocab['<UNK>']
        start2_token = self.vocab["start2"]
        ####### check ##########
        # check = [(word, get_word(word)) for word in self.data[idx][0] if self.vocab[word]==unk_token]
        # if len(check)>0:
        #     print(check)
        ########################
        # NNP - not the first and uppercase NNPS not start and uppercase + s at the end
        first = True if self.vocab[self.data[idx][0][1]] == start2_token else False # check if it the first word
        text = torch.tensor([self.vocab[get_word(first, word)] if get_word(first, word) != '<UNK>' else self.vocab[word.lower()] for word in self.data[idx][0]])
        text = text.view(-1)  # set shape to (seq_len,)
        return text, self.label[self.data[idx][1]]

