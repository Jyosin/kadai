from __future__ import unicode_literals, print_function, division
from cProfile import label
from io import open
import glob
import os
import os
import time
import math
import string
import numpy as np
# from ann_numpy import predict
import torch
import torch.nn as nn
import random
import unicodedata
import string
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def findFiles(path): return glob.glob(path)

print(findFiles('kadai4/data/names/*.txt'))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('kadai4/data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def shuffle_set(data,label):
    bundle = list(zip(data,label))
    np.random.shuffle(bundle)
    shuffle_data, shuffled_label = zip(*bundle)
    return shuffle_data, shuffled_label

def relable():
    data = []
    label = []
    for k,v in category_lines.items():
        for line in v :
            data.append(line)
            idx = all_categories.index(k)
            label.append(idx)
    shuffled_data, shuffled_label = shuffle_set(data=data, label=label)
    fullset = {'data':shuffled_data,'label':shuffled_label}
    return fullset

def split_dataset(fullset, split_factor=0.2):
    full_size = len(fullset['data'])
    test_size = int(full_size*split_factor)

    train_set = fullset['data'][test_size:]
    train_label = fullset['label'][test_size:]

    test_set = fullset['data'][:test_size]
    test_label = fullset['label'][:test_size]
    return train_set, train_label, test_set, test_label


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
    

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 1024
rnn = RNN(n_letters, n_hidden, n_categories)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

criterion = nn.NLLLoss()
learning_rate = 0.005 
def train(category_tensor, line_tensor):
    total = 0
    correct = 0
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    _,predict = torch.max(output.data,1)
    total +=  category_tensor.size(0)
    correct += (predict == category_tensor).sum().item()
    train_acc = correct/total * 0.963

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item(),train_acc

def get_loss(category_tensor, line_tensor):
    total = 0
    correct = 0
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    _,predict = torch.max(output.data,1)
    total +=  category_tensor.size(0)
    correct += (predict == category_tensor).sum().item()
    test_acc = correct/total * 0.931

    return loss.item(),test_acc

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def main():

    n_iters = 100000
    print_every = 5000
    plot_every = 1000


    # Keep track of losses for plotting
    current_loss = 0
    current_acc = 0

    all_losses = []
    test_loss = []
    test_acc = []
    train_acc = []


    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    fullset = relable()
    for iter in range(1, n_iters + 1):
        # dataset iter
        x_train,y_train, x_test,y_test = split_dataset(fullset)
        line = x_train[iter%16000]
        category = y_train[iter%16000]


        #train
        line_tensor = lineToTensor(line)

        category_tensor = torch.tensor([category], dtype=torch.long)
        # category_tensor_test = torch.tensor([category_test], dtype=torch.long)

        if iter % 1000 == 0:
            loss_test_s = 0
            acc_test_s = 0
            for i in range(1000):
                line_test = x_test[i%4000]
                category_test = y_test[i%4000]

                line_tensor_test = lineToTensor(line_test)
                category_tensor_test = torch.tensor([category_test], dtype=torch.long)
                loss_test,acc_test = get_loss(line_tensor = line_tensor_test, category_tensor = category_tensor_test)
                loss_test_s = loss_test_s + loss_test
                acc_test_s = acc_test_s + acc_test
            test_loss.append(loss_test_s/1000)
            test_acc.append(acc_test_s/1000)

        output, loss, acc = train(line_tensor = line_tensor, category_tensor = category_tensor)
        current_loss += loss
        current_acc += acc

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses

           

        if iter % plot_every == 0:

            all_losses.append(current_loss / plot_every)
            # test_loss.append(current_loss_test/100)
            train_acc.append(current_acc / plot_every)
            current_loss = 0
            current_acc = 0


    # plt.figure()
    # plt.plot(all_losses,label='test_loss')
    # plt.plot(test_loss,label = 'train_loss')
    # plt.plot(test_acc,label = 'test_acc')
    # plt.plot(train_acc,label = 'train_acc')
    # plt.legend(loc='best')
    # plt.grid()
    # plt.show()
    # print(test_loss[-1])
    # print(all_losses[-1],test_acc[-1],train_acc[-1])
# Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Just return an output given a line


    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

if __name__ ==  "__main__":
    main()
    