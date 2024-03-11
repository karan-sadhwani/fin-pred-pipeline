
import random
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.optimizers import Adam


class NeuralNetworkModel:
    def __init__(self, input_dim, hl=2, hu=100, dropout=False, rate=0.3, regularize=False, reg=l1(0.0005), optimizer=Adam(learning_rate=0.0001)):
        self.input_dim = input_dim
        self.hl = hl
        self.hu = hu
        self.dropout = dropout
        self.rate = rate
        self.regularize = regularize
        self.reg = reg if regularize else None
        self.optimizer = optimizer
        self.model = self.initialise_model()

    @staticmethod
    def set_seeds(seed=100):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    @staticmethod
    def cw(df):
        c0, c1 = np.bincount(df["dir"])
        w0 = (1 / c0) * (len(df)) / 2
        w1 = (1 / c1) * (len(df)) / 2
        return {0: w0, 1: w1}

    def initialise_model(self):
        model = Sequential()
        model.add(Dense(self.hu, input_dim=self.input_dim, activity_regularizer=self.reg, activation="relu"))
        if self.dropout:
            model.add(Dropout(self.rate, seed=100))
        for _ in range(self.hl - 1):
            model.add(Dense(self.hu, activation="relu", activity_regularizer=self.reg))
            if self.dropout:
                model.add(Dropout(self.rate, seed=100))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])
        return model

    def fit(self, X_train, y_train, **kwargs):
        return self.model.fit(X_train, y_train, **kwargs)

    def predict(self, X):
        return self.model.predict(X)