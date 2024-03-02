
import pandas as pd
import numpy as np
import yfinance as yf
import json

class DataIngestor():
    ''' Class for ingesting price data from yahoo finance
    '''
    def __init__(self, ticker, start, end, interval):
        self._ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval
        self.get_data()

    def __repr__(self):
        rep = "DataIngestor(ticker = {}, start = {}, end = {}, interval = {})"
        return rep.format(self._ticker, self.start, self.end, self.interval)

    def get_data(self):
        ''' retrieves and prepares the data
        '''
        asset = yf.Ticker(self._ticker)
        data = asset.history(start = self.start, end = self.end, interval = self.interval)
        data.index.name = "date"
        self.data = data
    
    def apply_config(self, config_path=None):
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            self.clean_columns(config.get('columns_to_drop'), config.get('columns_to_rename'))

    def clean_columns(self, columns_to_drop=None, columns_to_rename=None):
        if columns_to_drop:
            self.data.drop(columns=columns_to_drop, inplace=True)
        if columns_to_rename:
            self.data.rename(columns=columns_to_rename, inplace=True)
    
   