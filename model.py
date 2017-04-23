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


def train(input_1, intput_2, label, matcher, optimizer, criterion):
    optimizer.zero_grad()
    y = matcher(input_1, intput_2)
    loss = criterion(y, label.squeeze(0))

    loss.backward()
    optimizer.step()

    return loss


def trainEpochs(lang, pairs, labels, matcher, n_epochs, print_every=1000, plot_every=100, learning_rate=0.001):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    optimizer = optim.Adam(matcher.parameters(), lr=learning_rate)
    r = numpy.random.randint(0, len(pairs) - 1, n_epochs)
    training_pairs = [pairs[i] for i in r]
    training_labels = [labels[i] for i in r]
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, n_epochs - 256, 256):
        training_pair = variables_from_pair(lang, training_pairs[epoch: epoch + 256])
        label = [int(training_labels[i][1]) for i in range(epoch, epoch + 256)]
        q1 = training_pair[0]
        q2 = training_pair[1]
        loss = train(q1, q2, Variable(torch.LongTensor([label])), matcher, optimizer, criterion)
        print_loss_total += loss.data[0]

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / float(print_every) * 256
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch + 1 / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg))
            print_loss_total = 0
            print_loss_avg = 0
            if False:
                samples = numpy.random.randint(0, len(pairs) - 1, 1000)
                sample_pairs = [pairs[i] for i in samples]
                sample_labels = [labels[i] for i in samples]
                evaluate(lang, sample_pairs, sample_labels, matcher)
    torch.save(matcher, "matcher")


def evaluate(lang, pairs, labels, matcher):
    correct = 0
    for pair, label in zip(pairs, labels):
        v = variables_from_pair(lang, pair)
        y = matcher(v[0], v[1]).topk(1)[1].data[0][0]
        # print("label %s, predicted: %i" %(label, y))
        if y == int(label[1]):
            correct += 1
    print("accuracy: %.2f" % (float(correct) / len(pairs)))

if __name__ == "__main__":
    lang, pairs, labels = read_data()
    embedding_weights = load_word_embeddings(lang)
    if False:
        matcher = Matcher(lang.n_words, 300, embedding_weights, 1)
    else:
        matcher = torch.load("matcher")
    for i in range(25):
        trainEpochs(lang, pairs, labels, matcher, len(pairs), print_every=256 * 10, learning_rate=0.0001)




# TODO: add batching
# TODO: add gpu
# TODO: init glove
# TODO: bilstm
# TODO: stacked rnn
# TODO: attention encode + matching
