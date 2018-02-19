import os, sys
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pylab as plt

from dataset.mnist import load_mnist
from Simple_NeuralNetwork import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size, num_input = x_train.shape
hidden_nodes = 50
gen_coef = 0.01
lr = 0.1
batch_size = 100
num_class = 10

neuralNet = NeuralNet(num_input, hidden_nodes, num_class, gen_coef, lr)

train_acc_list = []
test_acc_list = []

for i in range(10000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    error = neuralNet.learn(x_batch, t_batch)
    
    if i % 100 == 0:
        train_acc = neuralNet.accuracy(x_train, t_train)
        test_acc = neuralNet.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

plt.subplot(121)
plt.plot(train_acc_list)
plt.subplot(122)
plt.plot(test_acc_list)
plt.show()