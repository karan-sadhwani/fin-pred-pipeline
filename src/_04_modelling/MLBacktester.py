import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

class MLBacktester():
    ''' Class for the vectorized backtesting of Machine Learning-based classification.
    '''
    def __init__(self, model, train_df, test_df, feature_columns, target):
        self.train_df = train_df
        self.test_df = test_df
        self.feature_columns = feature_columns
        self.target = target
        self.model = model

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
      
        TP = sum((df[self.target] == 1) & (df["pred"] == 1))
        # Note FP also known as Type I error
        FP = sum((df[self.target] != 1) & (df["pred"] == 1))
        TN = sum((df[self.target] == -1) & (df["pred"] == -1))
        # Note FP also known as Type II error
        FN = sum((df[self.target] != -1) & (df["pred"] == -1))

        # Calculate accuracy
        accuracy = (TP + TN) / len(df[self.target])
        print("Accuracy Score:", accuracy)
        # Precision, proportion of true positive predictions out of all positive predictions made by the model.
        # ability to make correct positive predictions and avoid false alarms
        precision = TP / (TP + FP)
        # Recall: Recall measures the proportion of true positive predictions out of all actual positive instances in the data.
        # ability to capture all positive instances
        recall= TP / (TP + FN)
        # F1-score: The F1-score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall.
        # considers both false positives and false negatives, making it useful when there's an uneven class distribution
        f1 = 2 * (precision * recall) / (precision + recall)
       
        print("Precision Score:", precision)
        print("Recall Score:", recall)
        print("F1 Score:", f1)

        # Calculate AUC
        y_true_binary = [(1 if label == 1 else 0) for label in df[self.target].tolist()]
        y_score = [(1 if pred == 1 else 0) for pred in df["pred"].tolist()]
        auc = roc_auc_score(y_true_binary, y_score)
        print("AUC Score:", auc)

        # # Confusion matrix
        conf_matrix = np.array([[TP, FP], [FN, TN]])
        labels = np.array([['TP', 'FP'],
                       ['FN', 'TN']])
    
        print("Confusion Matrix:")
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                print(f"{conf_matrix[i][j]:<5}", end="")
                print(f"({labels[i][j]})", end=" ")
            print()



