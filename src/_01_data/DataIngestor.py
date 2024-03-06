
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
    
    def adjust_columns(self, column_mapping):
        self.data.drop(columns=column_mapping["columns_to_drop"], inplace=True)
        self.data.rename(columns=column_mapping["columns_to_rename"], inplace=True)
