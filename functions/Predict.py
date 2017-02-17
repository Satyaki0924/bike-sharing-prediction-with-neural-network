import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style

from functions.Train import Train
from functions.data_preparations import Data

style.use('ggplot')


class Predict(object):
    def __init__(self, epochs, learning_rate, hidden_nodes):
        self.network, self.losses, \
        self.test_targets, self.test_features = Train(epochs, learning_rate, hidden_nodes)._return()
        self.scaled_features, self.test_data, self.rides = Data()._plot_return()
        self.path = os.path.dirname(os.path.abspath(__file__))

    def _plot_input_data(self):
        self.rides[:24 * 10].plot(x='dteday', y='cnt')
        plt.savefig(self.path + '/../graphs/data_plot.png')
        plt.cla()
        plt.close()
        self._losses_plot()

    def _losses_plot(self):
        plt.plot(self.losses['train'], label='Training loss')
        plt.plot(self.losses['validation'], label='Validation loss')
        plt.legend()
        plt.savefig(self.path + '/../graphs/losses_plot.png')
        plt.cla()
        plt.close()
        print('\n\nGraphs are plotted and saved at the graphs folder, successfully')

    def _predict(self):
        try:
            fig, ax = plt.subplots(figsize=(15, 8))
            mean, std = self.scaled_features['cnt']
            predictions = self.network.run(self.test_features) * std + mean
            ax.plot((self.test_targets['cnt'] * std + mean).values, label='Data')
            ax.plot(predictions[0], label='Prediction')
            ax.set_xlim(right=len(predictions))
            ax.legend()

            dates = pd.to_datetime(self.rides.ix[self.test_data.index]['dteday'])
            dates = dates.apply(lambda d: d.strftime('%b %d'))
            ax.set_xticks(np.arange(len(dates))[12::24])
            _ = ax.set_xticklabels(dates[12::24], rotation=45)
            plt.savefig(self.path + '/../graphs/prediction.png')
            plt.cla()
            plt.close()
            self._plot_input_data()
        except Exception as e:
            print(e)
