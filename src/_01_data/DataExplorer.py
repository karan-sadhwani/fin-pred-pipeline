import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class DataExplorer():
    ''' Class for exploring price data from yahoo finance
    log_returns:
        calculates log returns
    plot_prices:
        creates a price chart
    plot_returns:
        plots log returns either as time series ("ts") or histogram ("hist")
    mean_return:
        calculates mean return
    std_returns:
        calculates the standard deviation of returns (risk)
    annualized_perf:
        calculates annulized return and risk
    '''

    def __init__(self, data, ticker, target=None):
        self.data = data
        self.ticker = ticker
    
    def __repr__(self):
        rep = "DataExplorer(asset = {})"
        return rep.format(self.ticker)

    def add_log_diff_column(self, variable):
        '''calculates log of ratio between current and previous period values
        '''
        # Check if the variable exists in the dataframe
        if variable in self.data.columns:
            # Compute the logarithm of the specified column and create a new column
            self.data['log_diff_' + variable] = np.log(self.data[variable]/self.data[variable].shift(1))
            if ['log_diff_' + variable] == ['log_diff_price']:
                self.data.rename(columns={'log_diff_price' : 'log_returns'}, inplace=True)
        else:
            print(f"The variable '{variable}' does not exist in the dataframe.")
        
    def plot_time_series(self, variable):
        ''' creates a price chart
        '''
        self.data[variable].plot(figsize = (12, 8))
        plt.title("{} : {}".format(self.ticker, variable), fontsize = 15)
    
    def plot_histogram(self, variable):
        ''' plots histogram ("hist") alongside a normal distribution
        '''
        x = np.linspace(self.data[variable].min(), self.data[variable].max(), 10000)
        y = stats.norm.pdf(x, loc = self.data[variable].mean(), scale = self.data[variable].std()) # creating y values for a normal distribution
        plt.figure(figsize = (20, 8))
        plt.hist(self.data[variable], bins = 500, density = True, label = "Frequency Distribution")
        plt.plot(x, y, linewidth = 3, color = "red", label = "Normal Distribution")
        plt.title("Normal Distribution", fontsize = 20)
        plt.xlabel(variable, fontsize = 15)
        plt.ylabel("pdf", fontsize = 15)
        plt.legend(fontsize = 15)
        plt.show()

        # self.data[variable].hist(figsize = (12, 8), bins = int(np.sqrt(len(self.data))), density=True)
        # plt.title("Frequency: {}".format(self.ticker), fontsize = 15)

    def resample(self, freq, agg_method):
        ''' calculates mean return, with resampling to interval of your choosing:
        'D': Daily frequency
        'B': Business day frequency
        'W-Fri': Weekly frequency, with Friday as the default week end
        'ME': Month end frequency
        'MS': Month start frequency
        'Q': Quarter end frequency
        'A', 'Y': Year end frequency
        'AS', 'YS': Year start frequency
        'H': Hourly frequency
        '''
        self.data = self.data.resample(freq).agg(agg_method)
        self.data.dropna(inplace=True)

    def distribution_stats(self, variable):
        '''calculates fundamental distribution stats
        '''
        print("mean: {} | std: {}".format(self.data[variable].mean(), self.data[variable].std()))
        print("skew: {} | kurtosis: {}".format(stats.skew(self.data[variable].dropna()), 
                                        stats.kurtosis(self.data[variable].dropna(), fisher = True)))
    def normal_test(self, variable):
        z_stat, p_value = stats.normaltest(self_data[variable].dropna())  
        z_stat # high values -> reject H0
        p_value # low values (close to zero) -> reject H0

    def annualised_perf(self, variable, periods_per_annum):
        '''calculates annulized return and risk
        '''
        ann_mean = self.data[variable].mean() * periods_per_annum
        ann_std = self.data[variable].mean() * np.sqrt(periods_per_annum)
        print("annualised mean: {} | std: {}".format(ann_mean, ann_std))

    def multiple(self):
        '''calculates multiple over investment period, also expressed as a percentage
        '''
        multiple = self.data.price[-1] / self.data.price[0]
        price_change = (multiple - 1) * 100
        print("Multiple: {} | Price Change %: {}".format(mean_return, risk))

    def plot_scatterplot(self, x_variable, y_variable):
        '''tbc see lecture 128
        '''
        self.data.plot(kind = "scatter", x = x_variable, y = y_variable, figsize = (15,12), s = 50, fontsize = 15)
        for i in self.data.index:
            plt.annotate(i, xy=(self.data.loc[i, x_variable]+0.00005, self.data.loc[i, y_variable]+0.00005), size = 15)
            plt.xlabel(x_variable, fontsize = 15)
            plt.ylabel(y_variable, fontsize = 15)
            plt.title("Correlation map", fontsize = 20)
            plt.show()

    # high volumes in peak trading hours lead to lower spreads and more volatility (which gives you better chance of covering cost)