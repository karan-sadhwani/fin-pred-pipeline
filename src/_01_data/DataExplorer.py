import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class DataExplorer():
    ''' Class for exploring data
    '''

    def __init__(self, data, data_desc, interval, target=None):
        self.data = data
        self.ticker = data_desc
        self.interval = interval
    
    def __repr__(self):
        rep = "DataExplorer(Data Description = {} | interval = {})"
        return rep.format(self.data_desc, self.interval)

    def add_diff_column(self, variable, overnight_gap=True):
        '''calculates abs change between current and previous period value
        '''
        if variable in self.data.columns:
            self.data["diff_" + variable] = self.data[variable].diff().abs()
        if not overnight_gap:
        # This creates a boolean mask where 'True' indicates the first row of each day
            is_start_of_day = self.data.index.to_series().dt.time == pd.Timestamp("09:30:00").time()
            # Set the price change to NaN for the first row of each day to exclude overnight changes
            self.data.loc[is_start_of_day, "diff_" + variable] = pd.NA


    def add_log_diff_column(self, variable, new_variable_name=None):
        '''calculates log of ratio between current and previous period values
        '''
        # Check if the variable exists in the dataframe
        if variable in self.data.columns: 
            if new_variable_name:
                self.data[new_variable_name] = np.log(self.data[variable]/self.data[variable].shift(1))
            else:
                self.data['log_diff_' + variable] = np.log(self.data[variable]/self.data[variable].shift(1))
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

    def resample(self, interval,interval_mapping, agg_method):
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
        pd_interval = interval_mapping[interval]['pandas_resample']
        self.data = self.data.resample(pd_interval).agg(agg_method)
        self.data.dropna(inplace=True)
        self.interval = interval

    def distribution_stats(self, variable):
        '''calculates fundamental distribution stats
        '''
        print("mean: {} | std: {}".format(self.data[variable].mean(), self.data[variable].std()))
        print("skew: {} | kurtosis: {}".format(stats.skew(self.data[variable].dropna()), 
                                        stats.kurtosis(self.data[variable].dropna(), fisher = True)))
    
    def normal_test(self, variable):
        z_stat, p_value = stats.normaltest(self.data[variable].dropna())  
        z_stat # high values -> reject H0
        p_value # low values (close to zero) -> reject H0

    def annualised_perf(self, variable, interval_mapping):
        '''calculates annulized return and risk
        annualised mean: this is the average rate of growth of variable per year, 
        taking into account compounding of variable growth ovr multiple periods
        ov
        cagr: smoothed annual rate at which variable grows over certain period
        '''
        periods_per_annum = interval_mapping[self.interval]['annual_multiplier']
        ann_mean = self.data[variable].mean() * periods_per_annum
        trading_cagr = np.exp(ann_mean) - 1 # based on trading period rather than calendar period
        ann_std = self.data[variable].mean() * np.sqrt(periods_per_annum)
        print("CAGR: {} | Annualised mean: {} | std: {}".format(trading_cagr, ann_mean, ann_std))


    def simple_perf(self, variable):
        '''calculates multiple over investment period, also expressed as a percentage
        '''
        multiple = self.data[variable][-1] / self.data[variable][0]
        pct_change = (multiple - 1) * 100
        calendar_days = (self.data.index[-1] - self.data.index[0]).days 
        print(calendar_days)
        calendar_years = calendar_days / 365.25
        calendar_cagr = multiple**(1/calendar_years) - 1
        print("Multiple: {} | Percentage change (%): {} | CAGR: {}".format(multiple, pct_change, calendar_cagr))

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
    
    def plot_barchart(self, variable):
        '''plots bar chart
        '''
        plot_df = self.data.dropna().groupby("hour")[[variable]].mean()
        plot_df.plot(kind = "bar", figsize = (12, 8), fontsize = 13)
        plt.xlabel('hour', fontsize = 15)
        plt.ylabel(variable, fontsize = 15)
        plt.title(variable, fontsize = 15)
        plt.show()

    def convert_timezone(self, timezone = "America/New_York"):
        '''converts timeszone to one of choosing as a new column
        '''
        self.data[timezone] = self.data.index.tz_convert(timezone)
