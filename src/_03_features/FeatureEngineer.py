
import pandas as pd
import numpy as np

class FeatureEngineer():
    ''' Class for feature engineering. It creates an empty feature column list to which features are added using feature generation methods 
    '''
    def __init__(self, full_df, train_df=None, test_df=None, lags=None, train_ratio=None):
        self.full_df = full_df
        self.train_df = train_df
        self.test_df = test_df
        self.lags = lags
        self.train_ratio = train_ratio
        self.feature_columns = []

    def __repr__(self):
        rep = "FeatureEngineer(feature_columns  = {})"
        return rep.format(self.feature_columns)
    
    def add_log_diff_column(self, variable, new_variable_name=None):
        '''calculates log of ratio between current and previous period values
        '''
        # Check if the variable exists in the dataframe
        if variable in self.full_df.columns: 
            if new_variable_name:
                self.full_df[new_variable_name] = np.log(self.full_df[variable]/self.full_df[variable].shift(1))
            else:
                self.full_df['log_diff_' + variable] = np.log(self.full_df[variable]/self.full_df[variable].shift(1))
        else:
            print(f"The variable '{variable}' does not exist in the dataframe.")
    
    def binary_classifier(self, variable, new_variable_name=None):
        '''takes a value and returns -1 if the value is below 0, and 1 if the value is above or equal to 0
        '''
        # Check if the variable exists in the dataframe
        if variable in self.full_df.columns: 
            if new_variable_name:
                self.full_df[new_variable_name] = np.sign(self.full_df[variable])
        else:
            print(f"The variable '{variable}' does not exist in the dataframe.")

    def train_test_split(self):
        ''' splits the data into training set and test set
        '''
        full_df = self.full_df.copy()
        
        split_index = int(len(full_df) * self.train_ratio)
        split_date = full_df.index[split_index-1]
        train_start = full_df.index[0]
        test_end = full_df.index[-1]
        
        self.train_df = self.full_df.loc[train_start:split_date].copy()
        self.test_df = self.full_df.loc[split_date:test_end].copy()
        

    def lagged_features(self, df, lagged_features=None):
        """
        Create lagged features and update the specified dataset.
        """

        if lagged_features is None:
            lagged_features = self.feature_columns
        
        for feature in [lagged_features]:
            for lag in range(1, self.lags + 1):
                col = "{}_lag{}".format(feature, lag)
                df[col] = df[feature].shift(lag)
                if col not in self.feature_columns: 
                    self.feature_columns.append(col)
            df.dropna(inplace=True)
        
        return df
            
    
    def technical_indicators(self, df, variable, window=50):

        technical_indicators_features = ["sma", "boll", "min", "max", "mom", "vol"]

        df["sma"] = df[variable].rolling(window).mean() - df[variable].rolling(150).mean()
        df["boll"] = (df[variable] - df[variable].rolling(window).mean()) / df[variable].rolling(window).std()
        df["min"] = df[variable].rolling(window).min() / df[variable] - 1
        df["max"] = df[variable].rolling(window).max() / df[variable] - 1
        df["mom"] = df["log_returns"].rolling(3).mean()
        df["vol"] = df["log_returns"].rolling(window).std()

        self.feature_columns.extend(technical_indicators_features)
    
        return df

        

        