import psycopg2 as pg
import pandas as pd
import numpy as np

from config import DATABASE_URI

idx = pd.IndexSlice


class Returns:
    """
    Class for holding returns data.
    dataframe: expects multiindex pandas series containing closing prices
    log_returns: boolean. True means return log returns, false means arithmetic returns
    """

    def __init__(self, dataframe, log_returns):
        self.returns = self.calc_returns(dataframe["close"], log_returns)

    def calc_returns(self, dataframe, log_returns):
        """
        Calculate returns given closing prices
        dataframe: expects multiindex pandas series containing closing prices
        log_returns: boolean. True means return log returns, false means arithmetic returns
        """
        if log_returns:
            returns = np.log(dataframe) - np.log(dataframe.groupby(level=1).shift(1))
            return returns.dropna().unstack(level=1)
        else:
            returns = (dataframe / dataframe.groupby(level=1).shift(1)) - 1
            return returns.dropna().unstack(level=1)

    def __getattr__(self, name):
        """
        Defines attribute access. 'all' returns unstack dataframe for all tickers, else returns single ticker
        """
        if name == "all":
            return self.returns
        else:
            return self.returns.loc[idx[:, name]].dropna()


class Prices:
    """
    Class for holding prices data. Fetches data from postgres database
    tickers: list of strings. Tickers to fetch data for.
    """

    def __init__(self, tickers):
        self.tickers = tickers
        self.prices = self.get_prices()

    def get_prices(self):
        if len(self.tickers) == 1:
            sql = f"SELECT ticker, date, open, high, low, close, volume, closeadj FROM prices WHERE ticker='{self.tickers[0]}' AND frequency = 'DAILY' ORDER BY date DESC"
        else:
            sql = f"SELECT ticker, date, open, high, low, close, volume, closeadj FROM prices WHERE ticker IN {tuple(self.tickers)} AND frequency = 'DAILY' ORDER BY date DESC"
        with pg.connect(DATABASE_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                result = cur.fetchall()

        dataframe = pd.DataFrame.from_records(
            result,
            columns=[
                "ticker",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adjclose",
            ],
            coerce_float=True,
        ).sort_index()
        # Remove caret in front of index tickers
        dataframe["ticker"] = dataframe["ticker"].replace(r"[^\w]", "", regex=True)
        dataframe = dataframe.set_index(["date", "ticker"])
        return dataframe

    def __getattr__(self, name):
        """
        Defines attribute access. 'all' returns multiindex dataframe for all tickers, else returns single ticker
        """
        if name == "all":
            return self.prices
        else:
            return self.prices.loc[idx[:, name], :].droplevel(1)


class StockData:
    """
    Abstraction for accessing returns and price data.
    tickers: list of strings. Tickers to fetch data for.
    log_returns: boolean. True means return log returns, false means arithmetic returns. Default True.

    Correct usage:
        To get price data for all tickers: StockData.prices.all
        To get price data for one ticker: StockData.prices.AAPL
        And likewise for returns
    """

    def __init__(self, tickers, log_returns=True):
        self.tickers = tickers
        self.prices = Prices(tickers)
        self.returns = Returns(self.prices.all, log_returns)


class SPX:
    """
    Class for getting prices, returns, and 1-minute realized volatility for the S&P 500 index.
    """

    def __init__(self):
        with pg.connect(DATABASE_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT date, close FROM prices WHERE ticker='^GSPC' AND frequency = 'DAILY'"
                )
                result = cur.fetchall()
                self.prices = pd.Series(
                    [n[1] for n in result], [n[0] for n in result], dtype=np.float64
                ).sort_index()
        self.returns = self.calc_returns()
        self.vol = self.calc_vol()

    def calc_returns(self):
        """
        Calculates log daily close-to-close return.
        """
        returns = np.log(self.prices) - np.log(self.prices.shift(1))
        return returns.dropna()

    def calc_vol(self):
        """
        Calculates volatility using 1 min realized volatility method.
        Returns variance (square of standard deviation)
        """
        with pg.connect(DATABASE_URI) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT date, close FROM prices WHERE ticker='^GSPC' AND frequency = 'MINUTE'"
                )
                result = cur.fetchall()
                series = pd.Series(
                    [n[1] for n in result], [n[0] for n in result], dtype=np.float64
                ).sort_index()

        results = {}
        for idx, data in series.groupby(series.index.date):
            returns = np.log(data) - np.log(data.shift(1))
            results[idx] = np.sum(returns ** 2)

        results = pd.Series(results)
        results = results.reindex(pd.to_datetime(results.index))
        return results
