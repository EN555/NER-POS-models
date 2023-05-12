import numpy as np
from typing import List
import re



def extract_all_the_triplets(data:List, test: bool):
    cleaned_list = [string.replace("\n", "") for string in data]
    sentence, sentences = [], []
    for idx, word in enumerate(cleaned_list):
        if word == "":
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(word)
    triplets = []
    for idx, sentence in enumerate(sentences):
        sentence = np.insert(sentence, 0, ['start1', 'start2'])
        sentence = np.append(sentence, ['end1', 'end2'])
        for idx_word in range(len(sentence)-4):
            if test:
                triplets.append([word for word in sentence[idx_word: idx_word+5]])
            else:
                triplets.append(([word.split(" ")[0] for word in sentence[idx_word: idx_word+5]], sentence[idx_word+2].split(" ")[1]))
    return triplets

def extract_all_the_triplets_ner(data:List, test: bool):
    cleaned_list = [string.replace("\n", "") for string in data]
    sentence, sentences = [], []
    for idx, word in enumerate(cleaned_list):
        if word == "":
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(word)
    triplets = []
    for idx, sentence in enumerate(sentences):
        sentence = np.insert(sentence, 0, ['start1', 'start2'])
        sentence = np.append(sentence, ['end1', 'end2'])
        for idx_word in range(len(sentence)-4):
            if test:
                triplets.append([word for word in sentence[idx_word: idx_word+5]])
            else:
                triplets.append(([word.split("\t")[0] for word in sentence[idx_word: idx_word+5]], sentence[idx_word+2].split("\t")[1]))
    return triplets

datetime_regex = re.compile(r'\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{2,4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}[./-]\w{3,}|[012]?\d:[0-5]\d(:[0-5]\d)?(\.\d+)?\b)')
url_regex = re.compile(r'https?://(?:[-\w]+\.)?[-\w]+(?:\.\w+)+[-\w/_\?&=%#]*')
phone_regex = re.compile(r'\b(\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b')
surrounding_special_chars = '!@#$%^&*()_-+={}[]|\:;"<>,.?/~` '
patterns = [
    (r'.*ing$', 'VBG'),               # gerunds
    (r'.*(ed|en)$', 'VBD'),                # simple past
    (r'.*es$', 'VBZ'),                # 3rd singular present
    (r'.*ould$', 'MD'),               # modals
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'^[A-Z][a-z]*$', 'NNP'),
    (r'^[A-Z][a-z]*s$', 'NNPS')
]
def pos_tag(first, word):
    for pattern in patterns:
        if pattern[1] == 'NNP' or pattern[1] == 'NNPS':
            if not first and re.match(pattern[0], word):
                return pattern[1]
        elif re.match(pattern[0], word):
            return pattern[1]
    return "<UNK>"

def get_word(first, word: str):
    # clean_w = re.sub('[^a-z]+', '', word.lower())
    # word = word.lstrip(surrounding_special_chars).rstrip(surrounding_special_chars)
    # if word.replace(",", "").replace('.', '', 1).isdigit():
    #     return 'CD'
    # elif re.match(datetime_regex, word):
    #     return "<DATE>"
    # elif re.match(url_regex, word):
    #     return "<URL>"
    # elif re.match(phone_regex, word):
    #     return "<PHONE>"
    return pos_tag(first, word)




if __name__ == "__main__":
    print(get_word(False, "including"))