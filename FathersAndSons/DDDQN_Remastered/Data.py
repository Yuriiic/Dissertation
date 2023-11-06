import pandas as pd
import numpy as np


class Data:
    def __init__(self, timeframe, ticker, train=True):
        self.timeframe = timeframe  # timeframe of the data you wish to view - 1min, 15min, 30min, etc
        self.ticker = ticker  # which data you want to use
        self.data = self.load_data()
        self.preprocess_data(self.timeframe, train)
        self.step = 0
        self.offset = 0

    def load_data(self):
        print(f"Loading {self.ticker} Data")
        if self.ticker == "Oil":
            df = pd.read_csv(
                rf"C:\Research\Dissertation\Datasets\{self.ticker}_Summary.csv.gz"
            )
        elif self.ticker == "Wheat":
            df = pd.read_csv(
                rf"C:\Research\Dissertation\Datasets\{self.ticker}_Summary.csv.gz"
            )
        else:
            raise ValueError("Unrecognised ticker")
        print("---------------------")
        print(f"Finished Loading {self.ticker} Data")
        return df

    def preprocess_data(self, timeframe, train):
        """
        The preprocess function right now just converts the data into the timeframe of interest.
        In the future, you can include technical analysis components such as MACD, RSI, etc.
        Right now though we only have OHLCV and the 1 day returns
        :param timeframe:
        :return:
        """

        self.data.rename(columns={"Date-Time": "Period", "Last": "Close"}, inplace=True)
        # self.data = self.data[['Period', 'Open', 'High', 'Low', 'Last', 'Volume', 'No. Trades', 'Close Bid', 'No. Bids', 'Close Ask', 'No. Asks', 'Close Bid Size', 'Close Ask Size']]
        self.data = self.data[["Period", "Open", "High", "Low", "Close", "Volume"]]
        self.data["Period"] = pd.to_datetime(self.data["Period"])
        self.data.set_index("Period", inplace=True)
        if self.timeframe == "1M":
            pass
        elif self.timeframe == "D":
            self.data = self.data.resample(timeframe)
            agg_funcs = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            self.data = self.data.agg(agg_funcs)
        else:
            raise ValueError("Currently Unsupported Timeframe")

        self.data.reset_index(inplace=True)
        self.data["Period"] = self.data["Period"].dt.tz_localize(None)
        self.data["Weekend"] = (self.data["Period"].dt.dayofweek >= 5).astype(bool)
        self.data = self.data[self.data["Weekend"] == False]
        self.data = self.data.loc[:, self.data.columns != "Weekend"]
        self.data = self.data.dropna()
        self.data.drop(['Period'], axis=1, inplace=True)
        if train:
            self.data = self.data.head(int(len(self.data) * (0.8)))
        else:
            self.data_train = self.data.head(int(len(self.data) * (0.8)))
            self.data_test = self.data[~self.data.isin(self.data_train)].dropna()
            self.data = self.data_test

    def reset(self):
        # NOTE: check the index that days aren't missing.
        self.data.reset_index(drop=True, inplace=True)

        # I am not sure what this is about
        # I think it's to randomly pluck samples out of the data. This specifically is to start at a random place
        high = len(self.data.index)
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0

    def take_step(self):
        """Returns data for current trading day and done signal"""
        obs = self.data.iloc[self.offset + self.step]
        self.step += 1

        done = self.data.index[-1] == (self.offset + self.step)
        return obs, done
