import torch
from preprocess_ner_pos import *
import torch.nn as nn
from torch import optim
from ner_train import *
import pandas as pd
from tagger1 import *
import torchtext.vocab as vocab
from tqdm import tqdm
from torchtext.vocab import Vocab
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

class simple_mlp_pretrain_words(nn.Module):
    def __init__(self, batch_size, vocab_size, embedding_dim, hidden_dim, num_labels, words_weights):
        super().__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(words_weights, freeze=False)
        # self.word_embeddings = nn.Embedding(vocab_size, 50)
        self.linear1 = nn.Linear(embedding_dim*5, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        embeddings = self.word_embeddings(x)
        flattened = embeddings.view(embeddings.size(0), -1)
        tanh_out = self.dropout(torch.tanh(self.linear1(flattened)))
        return F.softmax(self.linear2(tanh_out), dim=1)

def create_vocab_from_all(train_data):
    torch.manual_seed(123)
    with open("words_pretrain.txt", "r") as file:
        vocab_words = [word.replace("\n", "") for word in file.readlines()]
    vectors = np.loadtxt("pretrain_vectors.txt")
    vocab = build_vocab_from_iterator([vocab_words] + list(yield_tokens(train_data)),specials=['ADJ', 'CD', 'VBD', 'VBZ', 'MD', 'NNS', 'NNP', 'NNPS', 'VBG', "<UNK>","<DATE>", "<URL>", "PHONE"])
    vocab.set_default_index(vocab['<UNK>'])  # when there have no token like this
    weight_tensor = torch.normal(mean=0.5, std=0.5, size=(len(vocab), 50))
    for idx, word in enumerate(vocab_words):
        loc = vocab[word]
        weight_tensor[loc] = torch.from_numpy(vectors[idx])
    return vocab, weight_tensor

def train_pos_pretrained(case, path_train, path_dev, path_test):
    epochs = 50
    embedding_dim = 50
    hidden_dim = 50
    device_ids = [0]
    batch_size = 400
    device = torch.device("cuda:{}".format(device_ids[0]))
    train, dev, test = preprocess(path_train, path_dev, path_test)
    vocab, weight_tensor = create_vocab_from_all(train)
    label_vocab = get_vocab(train, input=False)
    train_dataset = InitDataset(train, vocab, label_vocab)
    dev_dataset = InitDataset(dev, vocab, label_vocab)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=True, drop_last=True)
    model = simple_mlp_pretrain_words(batch_size, len(vocab), embedding_dim, hidden_dim, len(label_vocab), weight_tensor)
    if case == "continue":
        model.load_state_dict(torch.load("models/ass1/pos/checkpoint-3"))
        model.eval()
    model.to(device)
    torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
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
            torch.save(model.state_dict(), f"models/ass1/pos/3-checkpoint-{epoch//10}")
            create_graphs(f"3-pos_plot-loss-{epoch//10}",loss_acc_all, epoch,"loss")
            create_graphs(f"3-pos_plot-acc-{epoch//10}",dev_acc_all, epoch,"acc")
        print(f"epoch loss number {epoch}\n")
        print(f"Train: {train_running_loss / len(train_dataloader)} and The acc is: {train_acc/(batch_size*len(train_dataloader))}")
        print(f"Dev : {dev_running_loss / len(dev_dataloader)} and The acc is: {dev_acc/(batch_size*len(dev_dataloader))}")
    torch.save(model.state_dict(), "models/ass1/pos/3-checkpoint-1")

def predict_test_pos_pretrain(path_weights, test_dataset, embedding_dim, hidden_dim, label_vocab, vocab):
    test_dataset_ = InitDataset(test_dataset,vocab, label_vocab, True)
    test_dataloader = DataLoader(test_dataset_, batch_size=1, shuffle=False, drop_last=True)
    model = simple_mlp_pretrain_words(1, len(vocab), embedding_dim, hidden_dim, len(label_vocab), vocab)
    model.load_state_dict(torch.load(path_weights))
    model.eval()
    device_ids = [0]
    device = torch.device("cuda:{}".format(device_ids[0]))
    model.to(device)
    test = iter(open("pos/test").readlines())
    f = open('test3.pos', 'w')
    for i, sample in enumerate(test_dataloader):
        output = model(sample.to(device))
        predictions = torch.argmax(output, dim=1)
        nx = next(test).replace("\n", "")
        f.write(f"{nx} {label_vocab.lookup_token(predictions.item())}\n")
        if sample.numpy()[0][-1] == vocab["end2"]:
            next(test)
            f.write("\n")
if __name__ == "__main__":
    # custom_embeddings = vocab.Vectors(name='word_vec_sep.txt')
    # train_pos_pretrained("new","pos/train", "pos/dev", "pos/test")

    train, dev, test = preprocess("pos/train", "pos/dev", "pos/test")
    vocab, weights = create_vocab_from_all(train)
    label_vocab = get_vocab(train, input=False)
    # train_ner("dev", "ner/train", "ner/dev", "ner/test" )
    # print(label_vocab.lookup_tokens(range(len(label_vocab))))
#     print(len(train),len(dev), len(test))
#     print(train)
    predict_test_pos_pretrain("models/ass1/pos/3-checkpoint-2", test, 50, 50, label_vocab, vocab)