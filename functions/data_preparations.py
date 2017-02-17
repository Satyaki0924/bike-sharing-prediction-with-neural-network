import os

import pandas as pd


class Data(object):
    def __init__(self):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.rides = None
        self.data = None
        self.test_features = None
        self.test_targets = None
        self.train_features = None
        self.train_targets = None
        self.val_features = None
        self.val_targets = None
        self.scaled_features = None
        self.test_data = None
        self._prepare()

    def _prepare(self):
        data_path = str(self.path + '/../Bike-Sharing-Dataset/hour.csv')
        self.rides = pd.read_csv(data_path)
        self._create_dummy()

    def _create_dummy(self):
        dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']

        for each in dummy_fields:
            dummies = pd.get_dummies(self.rides[each], prefix=each, drop_first=False)

        self.rides = pd.concat([self.rides, dummies], axis=1)
        fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                          'weekday', 'atemp', 'mnth', 'workingday', 'hr']
        self.data = self.rides.drop(fields_to_drop, axis=1)
        self._scale_data()

    def _scale_data(self):
        quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
        self.scaled_features = {}
        for each in quant_features:
            mean, std = self.data[each].mean(), self.data[each].std()
            self.scaled_features[each] = [mean, std]
            self.data.loc[:, each] = (self.data[each] - mean) / std
        self._train_test_split()

    def _train_test_split(self):
        self.test_data = self.data[-21 * 24:]
        self.data = self.data[:-21 * 24]

        # Separate the data into features and targets
        target_fields = ['cnt', 'casual', 'registered']
        features, targets = self.data.drop(target_fields, axis=1), self.data[target_fields]
        self.test_features, self.test_targets = self.test_data.drop(target_fields, axis=1), \
                                                self.test_data[target_fields]
        self.train_features, self.train_targets = features[:-60 * 24], targets[:-60 * 24]
        self.val_features, self.val_targets = features[-60 * 24:], targets[-60 * 24:]
        self._return()
        self._plot_return()

    def _return(self):
        return self.train_features, \
               self.train_targets, \
               self.test_features, \
               self.test_targets, \
               self.val_features, \
               self.val_targets

    def _plot_return(self):
        return self.scaled_features, self.test_data, self.rides
