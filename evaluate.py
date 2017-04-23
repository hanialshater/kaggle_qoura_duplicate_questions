import torch
from torch.autograd import Variable
from tqdm import tqdm

from data_utils_new import Lang, variables_from_pair

import time

import numpy
from torch import nn, torch, optim
from torch.autograd import Variable
from data_utils_new import variables_from_pair, read_data
from embeddings import load_word_embeddings
from utils import timeSince
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_weights, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=False)

        del self.embedding.weight
        self.embedding.weight = nn.Parameter(embedding_weights)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        #output = torch.transpose(embedded, 0, 1)
        #packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, 256, self.hidden_size))


class Matcher(nn.Module):
    def __init__(self,
                 enc_input_size,
                 enc_hidden_size,
                 embedding_weights,
                 enc_n_layers):
        super(Matcher, self).__init__()
        self.encoder = EncoderRNN(enc_input_size, enc_hidden_size, embedding_weights, enc_n_layers)
        self.combiner = nn.Linear(enc_hidden_size, 2)

    def forward(self, input_1, input_2):
        hidden1 = self.encoder.initHidden()
        hidden2 = self.encoder.initHidden()
        _, hidden1 = self.encoder(input_1, hidden1)
        _, hidden2 = self.encoder(input_2, hidden2)
        _, hidden1 = self.encoder(input_1, hidden2)
        _, hidden2 = self.encoder(input_2, hidden1)
        #match = self.combiner(torch.cat((hidden1.squeeze(1), hidden2.squeeze(1)), 1))
        match = F.softmax(self.combiner(torch.abs(hidden1.squeeze(0) - hidden2.squeeze(0))))
        return match


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

def ev(n_pairs, matcher, pairs,labels):
    correct = 0
    for epoch in tqdm(range(0, n_pairs - 256, 256)):
        training_pair = variables_from_pair(lang, pairs[epoch: epoch + 256])
        label = Variable(torch.LongTensor([[int(labels[i][1]) for i in range(epoch, epoch + 256)]]))
        q1 = training_pair[0]
        q2 = training_pair[1]
        correct += evaluate(matcher, q1, q2, label).data[0]
    return correct / n_pairs


def evaluate(matcher, q1, q2, label):
    y = matcher(q1, q2).topk(1)[1].data[0][0]
    correct = torch.sum(y == label)
    return correct

lang, pairs, labels = read_data()
matcher = torch.load("matcher")
#print(ev(len(pairs), matcher, pairs, labels))
print(ev(10000, matcher, pairs, labels))
