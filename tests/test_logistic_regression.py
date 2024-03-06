# tests feature engineering and logistic regression model to ensure same results as known answers per tutorial
import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# import data and classes to test
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src._03_features.FeatureEngineer import FeatureEngineer
from src._04_modelling.MLBacktester import MLBacktester

data_dir = os.path.join(parent_dir, 'data')
csv_file = os.path.join(data_dir, "five_minute_pairs.csv")

# set up data to match tutorial
symbol = "EURUSD"
ptc = 0.00007
start = "2019-01-01"
end ="2020-08-31"
raw = pd.read_csv(csv_file, parse_dates = ["time"], index_col = "time")
raw = raw[symbol].to_frame().dropna()
raw = raw.loc[start:end]
raw.rename(columns={symbol: "price"}, inplace=True)

fe = FeatureEngineer(raw, lags=5, train_ratio=0.7)
fe.add_log_diff_column("price", new_variable_name="log_returns")
fe.binary_classifier("log_returns", new_variable_name="return_direction")
fe.train_test_split()
fe.train_df = fe.lagged_features(fe.train_df, lagged_features="log_returns")
fe.test_df = fe.lagged_features(fe.test_df, lagged_features="log_returns")

print(fe.train_df.info())
print(fe.test_df.info())


ml = MLBacktester(fe.train_df, fe.test_df, fe.feature_columns, target="return_direction")
ml.fit_model()
ml.make_predictions(fe.train_df)
ml.evaluate_results(fe.train_df)
ml.make_predictions(fe.test_df)
ml.evaluate_results(fe.test_df)


# results to compare to
# test set accuracy: 0.5120672414259074  strategy performance(0.176822, -0.893132)