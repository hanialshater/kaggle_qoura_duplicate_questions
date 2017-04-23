import pickle

import itertools
import torch
from torch.autograd import Variable

EOS_token = 1
PAD_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# def read_data():
#     with open("./data/words_and_tags.pkl", "rb") as f:
#         words, tags = pickle.load(f)
#
#     pairs = []
#     labels = []
#     for l in lines:
#         tokens = l.lower().split(',')
#         if len(tokens) != 6 or (len(tokens[3].split()) < 2 and len(tokens[4].split()) < 2):
#             continue
#         else:
#             q1 = [i for i in tokens[3].lower().replace("?", "").replace("?", "").replace(".", "").replace("(", "").replace(")", "").split() if i.strip() != ""]
#             q2 = [i for i in tokens[4].lower().replace("?", "").replace(",", "").replace(".", "").replace("(", "").replace(")", "").split() if i.strip() != ""]
#             lang.addSentence(q1)
#             lang.addSentence(q2)
#             pairs += [(q1, q2)]
#             labels += [tokens[5]]
#     return lang, pairs, labels

def read_data():
    lang = Lang("test")
    lines = open('/home/halshater/work/tutorials/kaggle/train.csv').read().strip().split('\n')[1:]

    pairs = []
    labels = []
    for l in lines:
        tokens = l.lower().split(',')
        if len(tokens) != 6 or (len(tokens[3].split()) < 2 and len(tokens[4].split()) < 2):
            continue
        else:
            q1 = [i for i in tokens[3].lower().replace("?", "").replace("?", "").replace(".", "").replace("(", "").replace(")", "").split() if i.strip() != ""]
            q2 = [i for i in tokens[4].lower().replace("?", "").replace(",", "").replace(".", "").replace("(", "").replace(")", "").split() if i.strip() != ""]
            lang.addSentence(q1)
            lang.addSentence(q2)
            pairs += [(q1, q2)]
            labels += [tokens[5]]
    return lang, pairs, labels


def indexes_from_sentence(lang, sentences):
    max_len = max([len(s) for s in sentences])
    return [[lang.word2index[word] for word in sentence] + [EOS_token] + [PAD_token for _ in range(len(sentence), max_len)]
            for sentence in sentences]


def variable_from_sentence(lang, sentences):
    indexes = indexes_from_sentence(lang, sentences)
    return Variable(torch.LongTensor(indexes))


def variables_from_pair(lang, pairs):
    input_variable = variable_from_sentence(lang, [p[0] for p in pairs])
    target_variable = variable_from_sentence(lang, [p[1] for p in pairs])
    return input_variable, target_variable


if __name__ == "__main__":
    lang, pairs, labels = read_data()