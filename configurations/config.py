COLUMN_MAPPING = {
    "columns_to_drop": ["Open", "High", "Low", "Dividends", "Stock Splits"],
    "columns_to_rename": {
        "Close": "price",
        "Volume": "volume"
    },
}

INTERVAL_MAPPING = {
    "day": {
        "yahoo_finance": "1d",
        "pandas_resample": "B",
        "annual_multiplier": 252  # Approximate number of trading days in a year
    },
    "week": {
        "yahoo_finance": "1wk",
        "pandas_resample": "W-Fri",
        "annual_multiplier": 52
    },
    "month": {
        "yahoo_finance": "1mo",
        "pandas_resample": "M",
        "annual_multiplier": 12
    },
    "hour": {
        "yahoo_finance": "1h",
        "pandas_resample": "H",
        "annual_multiplier": 252 * 6.5  # Approximate trading hours in a year, assuming 6.5 hours per trading day
    },
    "30min": {
        "yahoo_finance": "30m",
        "pandas_resample": "30T",  # 'T' or 'min' can be used for minute frequency
        "annual_multiplier": 252 * 6.5 * 2  # Twice the hourly multiplier for half-hour intervals
    },
    "15min": {
        "yahoo_finance": "15m",
        "pandas_resample": "15T",
        "annual_multiplier": 252 * 6.5 * 4  # Four times the hourly multiplier for 15-minute intervals
    },
    "5min": {
        "yahoo_finance": "5m",
        "pandas_resample": "5T",
        "annual_multiplier": 252 * 6.5 * 12  # Twelve times the hourly multiplier for 5-minute intervals
    },
}