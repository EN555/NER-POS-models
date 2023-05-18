import json
from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from tagger4 import SimpleMLP, preprocess, create_vocab_from_all, build_label_vocab, InitDataset, vocab_character
import torch

hidden_dim = 50


def load_model(path_weights):
    model = torch.load(path_weights)
    model.eval()
    return model


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def main():
    train, dev, test = preprocess("pos/train", "pos/dev", "pos/test")
    model: SimpleMLP = load_model(f"checkpoints/pos/final")
    vocab, weight_tensor = create_vocab_from_all(train, "pos")
    label_vocab = build_label_vocab(train)
    train_dataset = InitDataset(dev, vocab, label_vocab, task_name="pos")
    model.conv.register_forward_hook(get_activation('conv'))
    train_dataloader = DataLoader(train_dataset, 1, shuffle=False, drop_last=True)
    argmax_by_label_and_triple = defaultdict(lambda: defaultdict(int))
    score_by_filter_and_triple = defaultdict(lambda: defaultdict(list))
    argmax_by_filter_label_triple = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for element, _ in tqdm(train_dataloader):
        output = model(element)
        label_output = label_vocab.lookup_token(torch.argmax(output, dim=1))
        for word_idx in range(5):
            curr_triple = ""
            for char_idx in range(10):
                char_token = vocab_character.lookup_token(element[0][word_idx][char_idx + 1])
                curr_triple += char_token
            original_word = " "+curr_triple.replace("padding", " ")+" "
            windows = [original_word[i:i + 3] for i, _ in enumerate(original_word[:-2])]
            if len(windows) != 10:
                print(f"error in windows len: {original_word} to {len(windows)}, {windows}")
                continue
            for filter_idx in range(30):
                max_triple_idx = torch.argmax(activation["conv"][word_idx, filter_idx]).item()
                argmax_by_label_and_triple[label_output][windows[max_triple_idx]] += 1
                argmax_by_filter_label_triple[filter_idx][label_output][windows[max_triple_idx]] += 1
                for triple_idx in range(10):
                    triple_score = activation["conv"][word_idx, filter_idx, triple_idx].item()
                    triple_value = windows[triple_idx]
                    score_by_filter_and_triple[filter_idx][triple_value].append(triple_score)
    with open("argmax_by_label_and_triple.json", "w") as file:
        json.dump(dict(argmax_by_label_and_triple), file)
    with open("score_by_filter_and_triple.json", "w") as file:
        json.dump(dict(score_by_filter_and_triple), file)
    with open("argmax_by_filter_label_triple.json", "w") as file:
        json.dump(dict(argmax_by_filter_label_triple), file)


def extract_stuf():
    with open("argmax_by_label_and_triple.json", "r") as file:
        argmax_by_label_and_triple = json.load(file)
    noise_triple = [" en", "end", "nd1", "nd2", " st", "sta", "tar", "art", "rt1", "rt2", "t2 ", "d2 ", "d1 ", '   ', "2  ", "1  "]
    for label, argmax_by_triple in argmax_by_label_and_triple.items():
        print(label)
        for key in noise_triple:
            argmax_by_triple.pop(key, None)
        tirples = sorted(argmax_by_triple, key=argmax_by_triple.get, reverse=True)[:100]
        print(tirples)


def extract_stuff2():
    with open("argmax_by_filter_label_triple.json", "r") as file:
        argmax_by_filter_label_triple = json.load(file)
    for label, argmax_by_triple in argmax_by_filter_label_triple.items():
        print(label)
        print(sorted(argmax_by_triple, key=argmax_by_triple.get, reverse=True)[:100])


if __name__ == '__main__':
    main()

