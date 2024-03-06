import numpy as np
import pandas as pd


class PreProcessor():
    ''' Class for preparing the raw data for analysis 
    '''
    def __init__(self, df, train_ratio=None):
        self.df = df
        self.train_ratio = train_ratio
        
    def __repr__(self):
        rep = "PreProcessor(train_ratio = {})"
        
    
    
    
   