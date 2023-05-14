import torch
from preprocess_ner_pos import *
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import torch.utils.data as data
from typing import List
import re

MAX_WORD_LEN = 20
WINDOW_SIZE = 3
EMBEDDING_DIM = 50

patterns = [
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*(ed|en)$', 'VBD'),  # simple past
    (r'.*es$', 'VBZ'),  # 3rd singular present
    (r'.*ould$', 'MD'),  # modals
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'^[A-Z][a-z]*$', 'NNP'),
    (r'^[A-Z][a-z]*s$', 'NNPS')
]


def extract_windows(data_lines: List, test: bool):
    cleaned_list = [string.replace("\n", "") for string in data_lines]
    sentence, sentences = [], []
    for idx, word in enumerate(cleaned_list):
        if word == "":
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(word)
    windows = []
    for idx, sentence in enumerate(sentences):
        sentence = np.insert(sentence, 0, [f'start{i + 1}' for i in range(WINDOW_SIZE // 2)])
        sentence = np.append(sentence, [f'end{i + 1}' for i in range(WINDOW_SIZE // 2)])
        for idx_word in range(len(sentence) - WINDOW_SIZE - 1):
            if test:
                windows.append([word for word in sentence[idx_word: idx_word + WINDOW_SIZE]])
            else:
                windows.append(
                    (
                        [word.split(" ")[0] for word in sentence[idx_word: idx_word + WINDOW_SIZE]],
                        sentence[idx_word + WINDOW_SIZE // 2].split(" ")[1]
                    )
                )
    return windows


def get_word(first, word: str):
    for pattern in patterns:
        if pattern[1] == 'NNP' or pattern[1] == 'NNPS':
            if not first and re.match(pattern[0], word):
                return pattern[1]
        elif re.match(pattern[0], word):
            return pattern[1]
    return "<UNK>"


class SimpleMLP(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_labels):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, EMBEDDING_DIM)
        conv_out = 100
        self.linear1 = nn.Linear(EMBEDDING_DIM * WINDOW_SIZE + conv_out, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(hidden_dim, num_labels)
        self.conv = nn.Conv1d(in_channels=(MAX_WORD_LEN-1)*EMBEDDING_DIM, out_channels=conv_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv_activation = nn.ReLU()

    def forward(self, x):
        word_embeddings = self.word_embeddings(x[:, :, 0])
        char_embeddings = self.word_embeddings(x[:, :, 1:])
        char_embeddings = char_embeddings.reshape(
            char_embeddings.shape[0],
            char_embeddings.shape[1],
            char_embeddings.shape[2] * char_embeddings.shape[3]
        )
        char_embeddings = char_embeddings.transpose(1, 2)
        char_embeddings = self.conv_activation(self.max_pool(self.conv(char_embeddings)))
        word_embeddings = word_embeddings.view(word_embeddings.size(0), -1)
        char_embeddings = torch.squeeze(char_embeddings)
        embeddings = torch.cat([word_embeddings, char_embeddings], dim=1)
        tanh_out = self.dropout(torch.tanh(self.linear1(embeddings)))
        return F.softmax(self.linear2(tanh_out), dim=1)


class InitDataset(data.Dataset):
    def __init__(self, data_windows, vocab, label, test: bool = False):
        self.data = data_windows
        self.tokenizer = lambda x: x.split(" ")
        self.vocab = vocab
        self.label = label
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start2_token = self.vocab["start2"]
        first = self.vocab[self.data[idx][1]] == start2_token
        window_indexes = []
        for word in self.data[idx][0]:
            if get_word(first, word) != '<UNK>':
                word_index = self.vocab[get_word(first, word)]
            else:
                word_index = self.vocab[word.lower()]
            curr_indcies = [word_index]
            for char in word:
                curr_indcies.append(self.vocab[char.lower()])
            if len(curr_indcies) > MAX_WORD_LEN:
                curr_indcies = curr_indcies[:MAX_WORD_LEN]
            else:
                n = len(curr_indcies)
                for _ in range(MAX_WORD_LEN - n):
                    curr_indcies.append(self.vocab["padding"])
            window_indexes.append(curr_indcies)
        text = torch.tensor(window_indexes)
        if self.test:
            return text
        return text, self.label[self.data[idx][1]]


def preprocess(path_train: str, path_dev: str, path_test: str):
    TRAIN = extract_windows(open(path_train, 'r').readlines(), False)
    VALIDATION = extract_windows(open(path_dev, 'r').readlines(), False)
    TEST = extract_windows(open(path_test, 'r').readlines(), True)
    return TRAIN, VALIDATION, TEST


# def preprocess_ner(path_train: str, path_dev: str, path_test: str):
#     TRAIN = extract_windows(open(path_train, 'r').readlines(), False)
#     VALIDATION = extract_windows(open(path_dev, 'r').readlines(), False)
#     TEST = extract_windows(open(path_test, 'r').readlines(), True)
#     return TRAIN, VALIDATION, TEST


def yield_tokens(data_iter):
    for text, label in data_iter:
        yield [i.lower() for i in text]


def label_yield_tokens(data_iter):
    for text, label in data_iter:
        yield [label]


def build_word_vocab(train_datapipe):
    chars = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    vocab = build_vocab_from_iterator(yield_tokens(train_datapipe),
                                      specials=['ADJ', 'CD', 'VBD', 'VBZ', 'MD', 'NNS', 'NNP', 'NNPS', 'VBG',
                                                "<UNK>", "<DATE>", "<URL>", "PHONE", "padding"] + chars,
                                      max_tokens=20000, min_freq=10)
    vocab.set_default_index(vocab['<UNK>'])  # when there have no token like this
    return vocab


def build_label_vocab(train_datapipe):
    vocab = build_vocab_from_iterator(label_yield_tokens(train_datapipe))
    return vocab


def create_graphs(path: str, ls: List, epochs: int, type: str):
    if type == "loss":
        epochs = list(np.arange(0, epochs + 1, 1))
        new_ = copy.copy(ls)
    else:
        epochs = list(np.arange(0, epochs + 2, 1))
        new_ = copy.copy(ls)
        new_.insert(0, 0)
    plt.plot(epochs, new_)
    plt.xlabel('Number of Iteration')
    if type == "loss":
        plt.ylabel('Loss')
    else:
        plt.ylabel("Accuracy")
    plt.savefig(path)
    plt.clf()


def train_pos(path):
    epochs = 50
    hidden_dim = 50
    batch_size = 400
    train, dev, test = preprocess(f"{path}/train", f"{path}/dev", f"{path}/test")
    vocab = build_word_vocab(train)
    label_vocab = build_label_vocab(train)
    train_dataset = InitDataset(train, vocab, label_vocab)
    dev_dataset = InitDataset(dev, vocab, label_vocab)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=True, drop_last=True)
    model = SimpleMLP(len(vocab), hidden_dim, len(label_vocab))
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
            output = model(instances)
            predictions = torch.argmax(output, dim=1)
            train_acc += (predictions == labels).sum().item()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        scheduler.step(dev_acc / (batch_size * len(dev_dataloader)))
        print("########start dev##########")
        for i, batch in enumerate(dev_dataloader):
            instances, labels = batch
            optimizer.zero_grad()
            output = model(instances)
            predictions = torch.argmax(output, dim=1)
            dev_acc += (predictions == labels).sum().item()
            loss = criterion(output, labels)
            dev_running_loss += loss.item()
        dev_acc_all.append(dev_acc / (batch_size * len(dev_dataloader)))
        loss_acc_all.append(dev_running_loss / len(dev_dataloader))
        if epoch % 5 == 0:
            torch.save(model, f"checkpoints/pos-{epoch // 5}")
            create_graphs(f"pos_plot-loss-{epoch // 5}", loss_acc_all, epoch, "loss")
            create_graphs(f"pos_plot-acc-{epoch // 5}", dev_acc_all, epoch, "acc")
        print(f"epoch loss number {epoch}\n")
        print(
            f"Train: {train_running_loss / len(train_dataloader)} and The acc is: {train_acc / (batch_size * len(train_dataloader))}")
        print(
            f"Dev : {dev_running_loss / len(dev_dataloader)} and The acc is: {dev_acc / (batch_size * len(dev_dataloader))}")
    torch.save(model, "checkpoints/pos-final")


def predict_test(path_weights, test_dataset, hidden_dim, label_vocab, vocab):
    test_dataset_ = InitDataset(test_dataset, vocab, label_vocab, True)
    test_dataloader = DataLoader(test_dataset_, batch_size=1, shuffle=False, drop_last=True)
    model = SimpleMLP(len(vocab), hidden_dim, len(label_vocab))
    model.load_state_dict(torch.load(path_weights))
    model.eval()
    test = iter(open("pos/test").readlines())
    f = open('test1.pos', 'w')
    for i, sample in enumerate(test_dataloader):
        output = model(sample)
        predictions = torch.argmax(output, dim=1)
        nx = next(test).replace("\n", "")
        f.write(f"{nx} {label_vocab.lookup_token(predictions.item())}\n")
        if sample.numpy()[0][-1] == vocab["end2"]:
            next(test)
            f.write("\n")


if __name__ == "__main__":
    torch.manual_seed(1234)
    train_pos("pos")
