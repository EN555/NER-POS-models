import torch
from preprocess_ner_pos import *
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
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
    def __init__(self, data, vocab, label, test:bool=False):
        self.data = data
        self.tokenizer = lambda x: x.split(" ")
        self.vocab = vocab
        self.label = label
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.test:
            unk_token = self.vocab['<UNK>']
            start2_token = self.vocab["start2"]
            first = True if self.vocab[self.data[idx][1]] == start2_token else False  # check if it the first word
            text = torch.tensor([self.vocab[get_word(first, word)] if get_word(first, word) != '<UNK>' else self.vocab[word.lower()] for word in self.data[idx]])
            text = text.view(-1)
            return text
        unk_token = self.vocab['<UNK>']
        start2_token = self.vocab["start2"]
        first = True if self.vocab[self.data[idx][0][1]] == start2_token else False # check if it the first word
        set_sent = []
        for word in self.data[idx][0]:
            res_reg = get_word(first, word)
            if res_reg != '<UNK>':
                set_sent.append(self.vocab[res_reg])
            else:
                set_sent.append(self.vocab[word.lower()])
        text = torch.tensor(set_sent)
        # text = torch.tensor([self.vocab[word] for word in self.data[idx][0]])
        text = text.view(-1)
        return text, self.label[self.data[idx][1]]



def preprocess(path_train: str, path_dev: str, path_test: str):
    TRAIN = extract_all_the_triplets(open(path_train, 'r').readlines(), False)
    VALIDATION = extract_all_the_triplets(open(path_dev, 'r').readlines(), False)
    TEST = extract_all_the_triplets(open(path_test, 'r').readlines(), True)
    return TRAIN, VALIDATION, TEST

def preprocess_ner(path_train: str, path_dev: str, path_test: str):
    TRAIN = extract_all_the_triplets_ner(open(path_train, 'r').readlines(), False)
    VALIDATION = extract_all_the_triplets_ner(open(path_dev, 'r').readlines(), False)
    TEST = extract_all_the_triplets_ner(open(path_test, 'r').readlines(), True)
    return TRAIN, VALIDATION, TEST


def yield_tokens(data_iter):
    for text, label in data_iter:
        yield [i.lower() for i in text]

def label_yield_tokens(data_iter):
    for text, label in data_iter:
        yield [label]

def get_vocab(train_datapipe, input= True):
    if input:
        vocab = build_vocab_from_iterator(yield_tokens(train_datapipe), specials=[ 'ADJ','CD','VBD','VBZ','MD','NNS','NNP','NNPS','VBG',"<UNK>","<DATE>", "<URL>", "PHONE"], max_tokens=20000, min_freq=10)
        vocab.set_default_index(vocab['<UNK>'])  # when there have no token like this
    else:
        vocab = build_vocab_from_iterator(label_yield_tokens(train_datapipe))
    return vocab

def create_graphs(path:str, ls:List, epochs: int, type:str):
    if type == "loss":
        epochs = list(np.arange(0,epochs+1,1))
        new_ = copy.copy(ls)
    else:
        epochs = list(np.arange(0,epochs+2,1))
        new_ = copy.copy(ls)
        new_.insert(0,0)
    plt.plot(epochs, new_)
    plt.xlabel('Number of Iteration')
    if type == "loss":
        plt.ylabel('Loss')
    else:
        plt.ylabel("Accuracy")
    plt.title("NER Task")
    plt.savefig(path)
    plt.clf()

def train_pos(case, path_train, path_dev, path_test):
    epochs = 50
    embedding_dim = 50
    hidden_dim = 50
    device_ids = [0]
    batch_size = 400
    device = torch.device("cuda:{}".format(device_ids[0]))
    train, dev, test = preprocess(path_train, path_dev, path_test)
    vocab = get_vocab(train)
    label_vocab = get_vocab(train, input=False)
    train_dataset = InitDataset(train, vocab, label_vocab)
    dev_dataset = InitDataset(dev, vocab, label_vocab)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=True, drop_last=True)
    model = simple_mlp(batch_size, len(vocab), embedding_dim, hidden_dim, len(label_vocab))
    if case == "continue":
        model.load_state_dict(torch.load("models/ass1/pos/checkpoint-3"))
        model.eval()
    model.to(device)
    torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    dev_acc_all = []
    loss_acc_all = []
    for epoch in tqdm(range(epochs)):
        train_running_loss = 0
        dev_running_loss = 0
        train_acc, dev_acc = 0, 0
        print("###########start train##########")
        for batch in train_dataloader:
            instances, labels = batch
            optimizer.zero_grad()
            output = model(instances.to(device))
            predictions = torch.argmax(output, dim=1)
            train_acc += (predictions == labels.to(device)).sum().item()
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        scheduler.step(dev_acc/(batch_size*len(dev_dataloader)))
        # scheduler.step()
        print("########start dev##########")
        for i, batch in enumerate(dev_dataloader):
            instances, labels = batch
            optimizer.zero_grad()
            output = model(instances.to(device))
            predictions = torch.argmax(output, dim=1)
            dev_acc += (predictions == labels.to(device)).sum().item()
            loss = criterion(output, labels.to(device))
            dev_running_loss += loss.item()
        dev_acc_all.append(dev_acc/(batch_size*len(dev_dataloader)))
        loss_acc_all.append(dev_running_loss / len(dev_dataloader))
        if epoch%10 == 0:
            torch.save(model.state_dict(), f"models/ass1/pos/checkpoint-{epoch//10}")
            create_graphs(f"pos_plot-loss-{epoch//10}",loss_acc_all, epoch,"loss")
            create_graphs(f"pos_plot-acc-{epoch//10}",dev_acc_all, epoch,"acc")
        print(f"epoch loss number {epoch}\n")
        print(f"Train: {train_running_loss / len(train_dataloader)} and The acc is: {train_acc/(batch_size*len(train_dataloader))}")
        print(f"Dev : {dev_running_loss / len(dev_dataloader)} and The acc is: {dev_acc/(batch_size*len(dev_dataloader))}")
    torch.save(model.state_dict(), "models/ass1/ner/checkpoint-1")

def predict_test(path_weights, test_dataset, embedding_dim, hidden_dim, label_vocab, vocab):
    test_dataset_ = InitDataset(test_dataset,vocab, label_vocab, True)
    test_dataloader = DataLoader(test_dataset_, batch_size=1, shuffle=False, drop_last=True)
    model = simple_mlp(1, len(vocab), embedding_dim, hidden_dim, len(label_vocab))
    model.load_state_dict(torch.load(path_weights))
    model.eval()
    device_ids = [0]
    device = torch.device("cuda:{}".format(device_ids[0]))
    model.to(device)
    test = iter(open("pos/test").readlines())
    f = open('test1.pos', 'w')
    for i, sample in enumerate(test_dataloader):
        output = model(sample.to(device))
        predictions = torch.argmax(output, dim=1)
        nx = next(test).replace("\n", "")
        f.write(f"{nx} {label_vocab.lookup_token(predictions.item())}\n")
        if sample.numpy()[0][-1] == vocab["end2"]:
            next(test)
            f.write("\n")




if __name__ == "__main__":
    torch.manual_seed(1234)
    ## POS ##
    train_pos("new","pos/train", "pos/dev", "pos/test")
    # train, dev, test = preprocess("pos/train", "pos/dev", "pos/test")
    # vocab = get_vocab(train)
    # label_vocab = get_vocab(train, input=False)
    # predict_test("models/ass1/pos/checkpoint-3", test, 50, 50, label_vocab, vocab)
    # print(label_vocab.lookup_tokens(range(len(label_vocab))))
    ## NER ##
    # train, dev, test = preprocess_ner("ner/train", "ner/dev", "ner/test")
    # train_pos("ner","ner/train", "ner/dev", "ner/test" )
#     print(len(train),len(dev), len(test))
#     print(train)
