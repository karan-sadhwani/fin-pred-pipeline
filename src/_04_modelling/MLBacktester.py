import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

class MLBacktester():
    ''' Class for the vectorized backtesting of Machine Learning-based classification.
    '''
    def __init__(self, train_df, test_df, feature_columns, target):
        self.train_df = train_df
        self.test_df = test_df
        self.feature_columns = feature_columns
        self.target = target
        self.model = LogisticRegression(C = 1e6, max_iter = 100000, multi_class = "ovr")

    def __repr__(self):
        rep = "MLBacktester(feature_columns  = {}, model = {})"
        return rep.format(self.feature_columns, self.model)
        
    def fit_model(self):
        ''' Fitting the ML Model.
        '''
        # direction is actually the dependent variable (up or down 1 or)
        self.model.fit(self.train_df[self.feature_columns], self.train_df[self.target])

    def make_predictions(self, df=None):
        '''  Evaluates model on train or test set.
        '''
        if df is None:
            df = self.test_df
            
        self.predict = self.model.predict(df[self.feature_columns])
        
    def evaluate_results(self, df=None, tc=None):
        ''' Checks the performance of the trading strategy and compares to "buy and hold".
        '''
        if df is None:
            df = self.test_df

        df["pred"] = self.predict
      
        # Calculate accuracy
        accuracy = accuracy_score(df[self.target], self.predict)
        print("Accuracy Score:", accuracy)

        # Calculate precision, recall, and F1-score
       
        # precision = precision_score(df[self.target], self.predict)
        # recall = recall_score(df[self.target], self.predict)
        # f1 = f1_score(df[self.target], self.predict)
        # print("Precision Score:", precision)
        # print("Recall Score:", recall)
        # print("F1 Score:", f1)

        # # Calculate AUC
        # auc = roc_auc_score(self.df[self.target], self.predict)
        # print("AUC Score:", auc)

        # # Confusion matrix
        # print("Confusion Matrix:")
        # print(confusion_matrix(df[self.target], self.predict))

        ''' Checks the performance of the trading strategy and compares to "buy and hold".
        '''
        # # calculate Strategy Returns
        # self.test_df["strategy"] = self.test_df["pred"] * self.test_df["log_returns"]
        # # determine the number of trades in each bar
        # self.test_df["trades"] = self.test_df["pred"].diff().fillna(0).abs()
        
        # # subtract transaction/trading costs from pre-cost return
        # self.test_df.strategy = self.test_df.strategy - self.test_df.trades * self.tc
        
        # # calculate cumulative returns for strategy & buy and hold
        # self.test_df["creturns"] = self.test_df["log_returns"].cumsum().apply(np.exp)
        # self.test_df["cstrategy"] = self.test_df['strategy'].cumsum().apply(np.exp)
        # self.results = self.test_df
        
        # perf = self.results["cstrategy"].iloc[-1] # absolute performance of the strategy
        # outperf = perf - self.results["creturns"].iloc[-1] # out-/underperformance of strategy
        # print(perf, outperf)
        # return round(perf, 6), round(outperf, 6)
        
    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "Logistic Regression: | TC = {}".format(self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
