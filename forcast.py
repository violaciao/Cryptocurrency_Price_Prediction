import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly


class CryptoForcaster:

    today = datetime.today().strftime("%Y-%m-%d")

    def __init__(
        self,
        ticker: str = "ETH-USD",
        start_date: str = "2016-01-01",
        data_interval: str = "1d",  # '1h'
        prediction_freq: str = "D",  # hour: 'H'; day: 'D'; only business days: 'B'
        prediction_periods: int = 90,
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.data_interval = data_interval
        self.prediction_freq = prediction_freq
        self.prediction_periods = prediction_periods
        try:
            self.company_name = yf.Ticker(ticker).get_info()["name"]
            self.is_crypto = True
        except:
            self.company_name = yf.Ticker(ticker).get_info()["longName"]
            self.is_crypto = False
        self.crypto_hist_df = self.get_crypto_hist_df()
        self.model = self.fit_model()
        self.crypto_forcast = self.get_crypto_forcast()

    def run(self):
        # self.plot_open_prices()
        self.plot_forcast()
        # self.plot_components()

    def get_crypto_hist_df(self):
        df = yf.download(
            tickers=self.ticker,
            start=self.start_date,
            end=self.today,
            # period = "ytd",
            interval=self.data_interval,
            group_by="ticker",
            auto_adjust=True,
            prepost=True,
            threads=True,
            proxy=None,
        )

        try:
            df["Date"] = df.index.tz_localize("US/Eastern").tz_localize(None)
        except:
            df["Date"] = df.index.tz_convert("US/Eastern").tz_localize(None)

        new_names = {
            "Date": "ds",
            "Open": "y",
        }

        return df[["Date", "Open"]].rename(columns=new_names)

    def plot_open_prices(self):
        # plot the open price

        x = self.crypto_hist_df["ds"]
        y = self.crypto_hist_df["y"]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=y))

        # Set title
        fig.update_layout(
            title_text=f"Time series plot of {self.company_name} Open Price",
        )

        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(
                                count=1, label="1m", step="month", stepmode="backward"
                            ),
                            # dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(visible=True),
                type="date",
            )
        )

        fig.show()

    def fit_model(self):

        if self.is_crypto:
            m = Prophet(
                seasonality_mode="multiplicative",
                mcmc_samples=50
            )
        else:
            m = Prophet()

        m.add_country_holidays(country_name="US")

        m.fit(self.crypto_hist_df)

        return m

    def get_crypto_forcast(self):

        if self.is_crypto:
            future = self.model.make_future_dataframe(
                periods=self.prediction_periods,
                freq=self.prediction_freq,  # hour: 'H'; day: 'D'; only business days: 'B'
            )
        else:
            future = self.model.make_future_dataframe(
                periods=self.prediction_periods, freq=self.prediction_freq
            )

        return self.model.predict(future)

    def plot_forcast(self):
        fig = plot_plotly(
            self.model,
            self.crypto_forcast,
            xlabel="Prediction Window",
            ylabel=f"{self.company_name} Price ($)",
        )
        fig.show()

    def plot_components(self):
        fig = plot_components_plotly(self.model, self.crypto_forcast)
        fig.show()


if __name__ == "__main__":
    crypto = CryptoForcaster(
        ticker="ETH-USD",
        start_date="2020-01-01",
        data_interval="1h",
        prediction_freq="D",
        prediction_periods=60,
    )
    crypto.run()
