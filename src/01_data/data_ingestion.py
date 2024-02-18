import os
import pandas as pd
import yahoo_fin.stock_info as si
import yfinance as yf
from datetime import datetime, timedelta


# def fetch_historical_data(ticker, start_date, end_date):
#     """Fetches historical stock data for the specified ticker and date range."""
#     return si.get_data(ticker, start_date=start_date, end_date=end_date)

def calculate_dividend_yield(ticker, date, price):
    """Calculates dividend yield for a given date and stock price."""
    dividends = si.get_dividends(ticker, start_date=date - timedelta(days=365), end_date=date)
    annual_dividends = dividends.sum().iloc[0]
    return (float(annual_dividends) / price) * 100 if price else None

def calculate_pe_ratio(ticker, price):
    """Calculates P/E ratio using trailing twelve months EPS."""
    print(ticker)
    stats = si.get_stats(ticker)
    print(type(stats))
    print(stats)
    eps_ttm = pd.to_numeric(stats[stats.Attribute.str.contains("Diluted EPS")].Value.values[0], errors='coerce')
    return price / eps_ttm if eps_ttm and price else None

def fetch_dynamic_ratio(stats_val, attribute, date):
    """Fetches a dynamic financial ratio based on the attribute and date."""
    if attribute in stats_val.index:
        # Assuming the ratio is updated quarterly, find the closest previous quarter
        quarterly_dates = stats_val.columns[1:].to_series().apply(pd.to_datetime)
        closest_date = quarterly_dates[quarterly_dates <= date].max()
        ratio_value = pd.to_numeric(stats_val.loc[attribute, closest_date.strftime('%Y-%m-%d')], errors='coerce')
        return ratio_value
    return None

def fetch_historic_data(stock, start_date, end_date, interval):
    df_hist = stock.history(start=start_date,end=end_date, interval=interval)
    df_hist = df_hist.reset_index()
    df_hist['Date'] = pd.to_datetime(df_hist['Date'].astype(str))
    return df_hist

def fetch_fundamental_data(stock):
    q_financials = stock.quarterly_financials.transpose()
    financials = stock.financials.transpose()
    # Combine financial data for a comprehensive dataset
    df_fundamental = pd.concat([q_financials, financials]).sort_index().reset_index().rename(columns={'index': 'Date'})
    print(df_fundamental)
    df_fundamental = df_fundamental[['Date', 'Diluted EPS']]
    return df_fundamental

def calculate_dividends(hist, ticker):
    stock = yf.Ticker(ticker)
    dividends = stock.dividends.resample('Y').sum()
    hist['Annual Dividends'] = hist.index.year.map(dividends.to_dict())
    hist['Dividend Yield'] = hist['Annual Dividends'] / hist['Close']
    return hist

def encode_industry(stock):
    industry = stock.info['industry']
    return industry


# Main script execution
if __name__ == "__main__":

    tickers = ['AAPL']
    start_date ="2019-12-31"
    end_date ="2024-02-09"
    interval = "1d"

    # start_date = datetime.now() - timedelta(days=20)  # 5 years back
    # end_date = datetime.now()

    combined_data = pd.DataFrame()
    

    for ticker in tickers:

        stock = yf.Ticker(ticker)
        df_hist = fetch_historic_data(stock, start_date, end_date, interval)
        df_fundamental = fetch_fundamental_data(stock)
        print(df_fundamental)
        df_fundamental.info()
        print(df_hist)

       
        df_hist= pd.merge(df_hist, df_fundamental, how='left', on="Date")
        df_hist.info()
        df_hist['Diluted EPS'] = df_hist['Diluted EPS'].fillna(method='ffill')
        df_hist.info()
        # df_hist['EPS'] = eps_data.reindex(df_hist.index, method='ffill')  # Forward fill to propagate the last known EPS
        # df_hist['P/E Ratio'] = df_hist['Close'] / df_hist['EPS']

        # hist = calculate_dividends(hist, ticker)
        # industry = encode_industry(stock)
        # hist['Industry'] = industry
        # hist['Ticker'] = ticker
        # combined_data = pd.concat([combined_data, hist])


        # historical_data = fetch_historical_data(ticker, start_date, end_date)
        # stats = si.get_stats(ticker)
        # stats_val = si.get_stats_valuation(ticker)
        # dividends = si.get_dividends(ticker)
        # income_st = yf.Ticker(ticker).financials
        # income_st = yf.Ticker(ticker).financials
        # #earnings = si.get_earnings_history(ticker)
        # print(income_st)
        # #print(earnings)

    #     for index, row in historical_data.iterrows():
    #         date, price = index.date(), row['close']
    #         #historical_data.at[index, 'Dividend Yield'] = calculate_dividend_yield(ticker, date, price)
    #         historical_data.at[index, 'Trailing P/E'] = fetch_dynamic_ratio(stats_val, 'Diluted EPS (ttm)', date)
    #         historical_data.at[index, 'P/B Ratio'] = fetch_dynamic_ratio(stats_val, 'Price/Book (mrq)', date)  # Making P/B Ratio dynamic
    #         historical_data.at[index, 'Current Ratio'] = fetch_dynamic_ratio(stats_val, 'Current Ratio (mrq)', date)  # Making Current Ratio dynamic

    #     historical_data['Ticker'] = ticker
    #     data_frames = [combined_data, historical_data]  # Add more DataFrames as needed
    #     combined_data = pd.concat(data_frames, ignore_index=True)
      
    # combined_data.reset_index(inplace=True)
    # combined_data.rename(columns={'index': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    # final_data = combined_data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'EPS Ratio', 'P/B Ratio', 'Current Ratio']]

    # print(final_data.head())

    script_dir = os.path.dirname(__file__)

    # # Define the full path for the new CSV file
    csv_file_path = os.path.join(script_dir, 'my_dataframe.csv')

    # # Save the DataFrame to CSV
    df_hist.to_csv(csv_file_path, index=False)