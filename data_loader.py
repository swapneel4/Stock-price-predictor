import yfinance as yf
import pandas as pd


def load_stock_data(stock_symbol):

    data = yf.download(stock_symbol, start="2018-01-01")

    # keep only closing price
    data = data[['Close']]

    # reset index so Date does not create problems
    data = data.reset_index()

    data.to_csv("stock_data.csv", index=False)

    return data