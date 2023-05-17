import torch
from tagger1 import *
import copy
from preprocess_ner_pos import *
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import random
from sklearn.utils.class_weight import compute_class_weight
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

class simple_ner(nn.Module):
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

class InitDatasetNER(data.Dataset):
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
            reg_word = self.vocab["<DIS>"]
            text = [self.vocab[word.lower()] for word in self.data[idx]]
            if not self.data[idx][2][0].isupper():
                text[2] = reg_word
            text = torch.tensor(text)
            text = text.view(-1)  # set shape to (seq_len,)
            return text
        reg_word = self.vocab["<DIS>"]
        text = [self.vocab[word.lower()] for word in self.data[idx][0]]
        if not self.data[idx][0][2][0].isupper():
            text[2] = reg_word
        text = torch.tensor(text)
        text = text.view(-1)  # set shape to (seq_len,)
        return text, self.label[self.data[idx][1]]

def get_vocab_ner(train_datapipe):
    vocab = build_vocab_from_iterator(yield_tokens(train_datapipe), specials=["<UNK>","<DIS>"],max_tokens=60000)
    vocab.set_default_index(vocab['<UNK>'])  # when there have no token like this
    return vocab


def data_weights(data, vocab):
    kp = []
    for text, label in data:
        kp.append(vocab.vocab[label])
    return kp

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

def train_ner(case,path_train,path_dev,path_test):
    epochs = 101
    embedding_dim = 50
    hidden_dim = 50
    device_ids = [0]
    batch_size = 100
    device = torch.device("cuda:{}".format(device_ids[0]))
    train, dev, test = preprocess_ner(path_train, path_dev, path_test)
    vocab = get_vocab_ner(train)
    label_vocab = get_vocab(train, input=False)
    train_dataset = InitDatasetNER(train, vocab, label_vocab)
    dev_dataset = InitDatasetNER(dev, vocab, label_vocab)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=True, drop_last=True)
    model = simple_mlp(batch_size, len(vocab), embedding_dim, hidden_dim, len(label_vocab))
    if case == "continue":
        model.load_state_dict(torch.load("models/ass1/ner/ver2-checkpoint-5"))
        model.eval()
    model.to(device)
    torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    all_labels = data_weights(train,label_vocab)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(all_labels),
        y=all_labels
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    dev_acc_all = []
    loss_acc_all = []
    for epoch in tqdm(range(epochs)):
        train_running_loss = 0
        dev_running_loss = 0
        train_acc, dev_acc = 0, 0
        train_examples, dev_examples = 0,0
        for batch in train_dataloader:
            instances, labels = batch
            optimizer.zero_grad()
            output = model(instances.to(device))
            predictions = torch.argmax(output, dim=1)
            for idx, pred in enumerate(predictions):
                if not (label_vocab.lookup_token(pred.item()) == 'O' and labels[idx] == pred.item()):
                    train_examples+=1
                    if labels[idx] == pred.item():
                        train_acc +=1
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        # incorrect_indices =[]
        for i, batch in enumerate(dev_dataloader):
            instances, labels = batch
            optimizer.zero_grad()
            output = model(instances.to(device))
            predictions = torch.argmax(output, dim=1)
            # dev_acc += (predictions == labels.to(device)).sum().item()
            for idx, pred in enumerate(predictions):
                # print(label_vocab.lookup_token(pred.item()))
                if not (label_vocab.lookup_token(pred.item()) == 'O' and labels[idx] == pred.item()):
                    dev_examples+=1
                    if labels[idx] == pred.item():
                        dev_acc +=1
            loss = criterion(output, labels.to(device))
            dev_running_loss += loss.item()
        dev_acc_all.append(dev_acc/dev_examples)
        loss_acc_all.append(dev_running_loss / len(dev_dataloader))
        if epoch%10 == 0:
            torch.save(model.state_dict(), f"models/ass1/ner/ver-final-checkpoint-{epoch//10}")
            create_graphs(f"ner_plot-loss-final-ver3-{epoch//10}",loss_acc_all, epoch,"loss")
            create_graphs(f"ner_plot-acc-final-{epoch//10}",dev_acc_all, epoch,"acc")
        scheduler.step(dev_acc/dev_examples)
        print(f"epoch loss number {epoch}\n")
        print(f"Train: {train_running_loss / len(train_dataloader)} and The acc is: {train_acc/train_examples}")
        print(f"Dev : {dev_running_loss / len(dev_dataloader)} and The acc is: {dev_acc/dev_examples}")

def predict_test_ner(path_weights, test_dataset, embedding_dim, hidden_dim, label_vocab, vocab):
    test_dataset_ = InitDatasetNER(test_dataset,vocab, label_vocab, True)
    test_dataloader = DataLoader(test_dataset_, batch_size=1, shuffle=False, drop_last=True)
    model = simple_ner(1, len(vocab), embedding_dim, hidden_dim, len(label_vocab))
    model.load_state_dict(torch.load(path_weights))
    model.eval()
    device_ids = [0]
    device = torch.device("cuda:{}".format(device_ids[0]))
    model.to(device)
    test = iter(open("ner/test").readlines())
    f = open('test1.ner', 'w')
    for i, sample in enumerate(test_dataloader):
        output = model(sample.to(device))
        predictions = torch.argmax(output, dim=1)
        nx = next(test).replace("\n", "")
        f.write(f"{nx} {label_vocab.lookup_token(predictions.item())}\n")
        if sample.numpy()[0][-1] == vocab["end2"]:
            next(test)
            f.write("\n")


if __name__ == "__main__":
    torch.manual_seed(124)
    ## NER ##
    train, dev, test = preprocess_ner("ner/train", "ner/dev", "ner/test")
    vocab = get_vocab_ner(train)
    label_vocab = get_vocab(train, input=False)
    # train_ner("dev", "ner/train", "ner/dev", "ner/test" )
    # print(label_vocab.lookup_tokens(range(len(label_vocab))))
#     print(len(train),len(dev), len(test))
#     print(train)
    predict_test_ner("models/ass1/ner/ver-final-checkpoint-6", test, 50, 50, label_vocab, vocab)