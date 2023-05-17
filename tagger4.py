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
from sklearn.utils.class_weight import compute_class_weight

MAX_WORD_LEN = 11

patterns = [
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*(ed|en)$', 'VBD'),  # simple past
    (r'.*es$', 'VBZ'),  # 3rd singular present
    (r'.*ould$', 'MD'),  # modals
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'^[A-Z][a-z]*$', 'NNP'),
    (r'^[A-Z][a-z]*s$', 'NNPS')
]


def build_word_vocab_characters():
    chars = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    vocab = build_vocab_from_iterator(iter(chars), specials= ["<UNK>", "padding"])
    vocab.set_default_index(vocab['<UNK>'])  # when there have no token like this
    return vocab


vocab_character = build_word_vocab_characters()


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
        sentence = np.insert(sentence, 0, [f'start{i + 1}' for i in range(5 // 2)])
        sentence = np.append(sentence, [f'end{i + 1}' for i in range(5 // 2)])
        for idx_word in range(len(sentence) - 5 - 1):
            if test:
                windows.append([word for word in sentence[idx_word: idx_word + 5]])
            else:
                windows.append(
                    (
                        [word.split()[0] for word in sentence[idx_word: idx_word + 5]],
                        sentence[idx_word + 5 // 2].split()[1]
                    )
                )
    return windows


def create_vocab_from_all(train_data, mode: str):
    torch.manual_seed(123)
    if mode == "pos":
        specials = ['ADJ', 'CD', 'VBD', 'VBZ', 'MD', 'NNS', 'NNP', 'NNPS', 'VBG', "<UNK>", "<DATE>", "<URL>", "PHONE"]
    else:
        specials = ["<UNK>", "<DIS>"]
    with open("vocab.txt", "r") as file:
        vocab_words = [word.replace("\n", "") for word in file.readlines()]
    vectors = np.loadtxt("wordVectors.txt")
    vocab = build_vocab_from_iterator([vocab_words] + list(yield_tokens(train_data)), specials=specials)
    vocab.set_default_index(vocab['<UNK>'])  # when there have no token like this
    weight_tensor = torch.normal(mean=0.5, std=0.5, size=(len(vocab), 50))
    for idx, word in enumerate(vocab_words):
        loc = vocab[word]
        weight_tensor[loc] = torch.from_numpy(vectors[idx])
    return vocab, weight_tensor


class SimpleMLP(nn.Module):
    def __init__(self, weight_tensor, hidden_dim, num_labels):
        super().__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(weight_tensor, freeze=False)
        nn.init.uniform_(self.word_embeddings.weight, -np.sqrt(3 / 50), np.sqrt(3 / 50))
        self.character_embeddings = nn.Embedding(len(vocab_character), 30)
        self.character_embeddings.weight.data[vocab_character["padding"]] = torch.zeros(30, requires_grad=False)
        nn.init.uniform_(self.character_embeddings.weight, -np.sqrt(3 / 30), np.sqrt(3 / 30))
        self.linear1 = nn.Linear((30 + 50) * 5, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(hidden_dim, num_labels)
        self.conv = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool1d(kernel_size=10, stride=1)
        self.conv_activation = nn.ReLU()

    def forward(self, x):
        word_embeddings = self.word_embeddings(x[:, :, 0])
        char_embeddings = self.character_embeddings(x[:, :, 1:])
        word_embeddings = word_embeddings.view(-1, word_embeddings.shape[2])
        char_embeddings = char_embeddings.view(-1, char_embeddings.shape[2], char_embeddings.shape[3])
        char_embeddings = char_embeddings.transpose(1, 2)
        char_embeddings = self.dropout(char_embeddings)
        char_embeddings = self.conv_activation(self.max_pool(self.conv(char_embeddings)))
        char_embeddings = char_embeddings.squeeze(2)
        embeddings = torch.cat([word_embeddings, char_embeddings], dim=1)
        embeddings = embeddings.view(x.shape[0], -1)
        tanh_out = self.dropout(torch.tanh(self.linear1(embeddings)))
        return F.softmax(self.linear2(tanh_out), dim=1)


class InitDataset(data.Dataset):
    def __init__(self, data_windows, vocab, label, test: bool = False, task_name: str = "pos"):
        self.data = data_windows
        self.vocab = vocab
        self.label = label
        self.test = test
        self.task_name = task_name

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _get_word_pos(first, word: str):
        for pattern in patterns:
            if pattern[1] == 'NNP' or pattern[1] == 'NNPS':
                if not first and re.match(pattern[0], word):
                    return pattern[1]
            elif re.match(pattern[0], word):
                return pattern[1]
        return "<UNK>"

    def get_word_ner(self, idx, window_index, word):
        start2_token = self.vocab["start2"]
        first = self.vocab[self.data[idx][1]] == start2_token
        if not first and window_index == 2 and not self.data[idx][0][2][0].isupper():
            return self.vocab["<DIS>"]
        return self.vocab[word.lower()]

    def get_word_pos(self, idx, word):
        start2_token = self.vocab["start2"]
        first = self.vocab[self.data[idx][1]] == start2_token
        idx_get_word = self._get_word_pos(first, word)
        if idx_get_word != '<UNK>':
            word_index = self.vocab[idx_get_word]
        else:
            word_index = self.vocab[word.lower()]
        return word_index

    def get_word(self, idx, window_index, word):
        if self.task_name == "pos":
            return self.get_word_pos(idx, word)
        return self.get_word_ner(idx, window_index, word)

    def __getitem__(self, idx):
        window_indexes = []
        for window_index, word in enumerate(self.data[idx][0]):
            curr_indcies = [self.get_word(idx, window_index, word)]
            for char in word:
                curr_indcies.append(vocab_character[char.lower()])
            if len(curr_indcies) > MAX_WORD_LEN:
                curr_indcies = curr_indcies[:MAX_WORD_LEN]
            else:
                n = len(curr_indcies)
                for _ in range(MAX_WORD_LEN - n):
                    curr_indcies.append(vocab_character["padding"])
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


def create_graphs(path: str, ls: List, epochs: int, type_g: str, task_name):
    if type_g == "loss":
        epochs = list(np.arange(0, epochs + 1, 1))
        new_ = copy.copy(ls)
    else:
        epochs = list(np.arange(0, epochs + 2, 1))
        new_ = copy.copy(ls)
        new_.insert(0, 0)
    plt.plot(epochs, new_)
    plt.xlabel('Number of Iteration')
    if type_g == "loss":
        plt.ylabel('Loss')
    else:
        plt.ylabel("Accuracy")
    plt.title(f"{task_name} Task")
    plt.savefig(path)
    plt.clf()


def ner_acc(predictions, label_vocab, labels):
    dev_examples = 0
    dev_acc = 0
    for idx, pred in enumerate(predictions):
        if not (label_vocab.lookup_token(pred.item()) == 'O' and labels[idx] == pred.item()):
            dev_examples += 1
            if labels[idx] == pred.item():
                dev_acc += 1
    return dev_acc, dev_examples


def pos_acc(predictions, labels):
    return (predictions == labels).sum().item(), predictions.shape[0]


def data_weights(data, vocab):
    kp = []
    for text, label in data:
        kp.append(vocab.vocab[label])
    return kp


def train_model(task_name):
    epochs = 50
    hidden_dim = 50
    batch_size = 400
    train, dev, test = preprocess(f"{task_name}/train", f"{task_name}/dev", f"{task_name}/test")
    vocab, weight_tensor = create_vocab_from_all(train, task_name)
    label_vocab = build_label_vocab(train)
    train_dataset = InitDataset(train, vocab, label_vocab, task_name=task_name)
    dev_dataset = InitDataset(dev, vocab, label_vocab, task_name=task_name)
    test_dataset = InitDataset(test, vocab, label_vocab, task_name=task_name)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False)
    model = SimpleMLP(weight_tensor, hidden_dim, len(label_vocab))
    model.eval()
    all_labels = data_weights(train,label_vocab)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(all_labels),
        y=all_labels
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float))
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    dev_acc_all = []
    loss_acc_all = []
    for epoch in tqdm(range(epochs)):
        train_running_loss = 0
        dev_running_loss = 0
        train_acc, dev_acc = 0, 0
        train_examples, dev_examples = 0, 0
        print("###########start train##########")
        for batch in train_dataloader:
            instances, labels = batch
            optimizer.zero_grad()
            output = model(instances)
            predictions = torch.argmax(output, dim=1)
            if task_name == "pos":
                curr_train_acc, curr_train_examples = pos_acc(predictions, labels)
            else:
                curr_train_acc, curr_train_examples = ner_acc(predictions, label_vocab, labels)
            train_acc += curr_train_acc
            train_examples += curr_train_examples
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        print("########start dev##########")
        for i, batch in enumerate(dev_dataloader):
            instances, labels = batch
            optimizer.zero_grad()
            output = model(instances)
            predictions = torch.argmax(output, dim=1)
            if task_name == "pos":
                curr_dev_acc, curr_dev_examples = pos_acc(predictions, labels)
            else:
                curr_dev_acc, curr_dev_examples = ner_acc(predictions, label_vocab, labels)
            dev_acc += curr_dev_acc
            dev_examples += curr_dev_examples
            loss = criterion(output, labels)
            dev_running_loss += loss.item()
        dev_acc_all.append(dev_acc / dev_examples)
        loss_acc_all.append(dev_running_loss / len(dev_dataloader))
        scheduler.step(dev_acc / dev_examples)
        if epoch % 5 == 0:
            torch.save(model, f"checkpoints/{task_name}/{epoch // 5}")
            create_graphs(f"checkpoints/{task_name}/plot-loss-{epoch // 5}", loss_acc_all, epoch, "loss", task_name)
            create_graphs(f"checkpoints/{task_name}/plot-acc-{epoch // 5}", dev_acc_all, epoch, "acc", task_name)
        print(f"epoch loss number {epoch}\n")
        print(
            f"Train: {train_running_loss / len(train_dataloader)} and The acc is: {train_acc / train_examples}")
        print(
            f"Dev : {dev_running_loss / len(dev_dataloader)} and The acc is: {dev_acc / dev_examples}")
    torch.save(model, f"checkpoints/{task_name}/final")
    predict_test(model, test_dataloader, label_vocab, vocab, task_name)


def load_model(path_weights, hidden_dim, weight_tensor, label_vocab):
    model = SimpleMLP(weight_tensor, hidden_dim, len(label_vocab))
    model.load_state_dict(torch.load(path_weights))
    model.eval()


def predict_test(model, test_dataloader, label_vocab, vocab, task_name: str):
    test = iter(open(f"{task_name}/test").readlines())
    f = open(f'test1.{task_name}', 'w')
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
    train_model("ner")
