import sys

import numpy as np

from functions.data_preparations import Data
from functions.neural_net import NeuralNetwork


class Train(object):
    def __init__(self, epochs, learning_rate, hidden_nodes):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_nodes = hidden_nodes
        self.output_nodes = 1
        self.network = None
        self.losses = None
        self.train_features, self.train_targets, self.test_features, \
        self.test_targets, self.val_features, self.val_targets = Data()._return()

    def MSE(self, y, Y):
        return np.mean((y - Y) ** 2)

    def _run(self):
        N_i = self.train_features.shape[1]
        self.network = NeuralNetwork(N_i, self.hidden_nodes, self.output_nodes, self.learning_rate)

        self.losses = {'train': [], 'validation': []}
        for e in range(self.epochs):
            batch = np.random.choice(self.train_features.index, size=128)
            for record, target in zip(self.train_features.ix[batch].values,
                                      self.train_targets.ix[batch]['cnt']):
                self.network.train(record, target)
            train_loss = self.MSE(self.network.run(self.train_features), self.train_targets['cnt'].values)
            val_loss = self.MSE(self.network.run(self.val_features), self.val_targets['cnt'].values)
            sys.stdout.write("\rProgress: " + str(100 * e / float(self.epochs))[:4]
                             + "% ... Training loss: " + str(train_loss)[:5]
                             + " ... Validation loss: " + str(val_loss)[:5]
                             + " - By Satyaki Sanyal")

            self.losses['train'].append(train_loss)
            self.losses['validation'].append(val_loss)

    def _return(self):
        self._run()
        return self.network, self.losses, self.test_targets, self.test_features
