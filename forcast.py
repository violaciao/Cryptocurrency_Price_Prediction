import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly


class CryptoForcaster:

    today = datetime.today().strftime('%Y-%m-%d')

    def __init__(
        self, 
        ticker: str = 'ETH-USD', 
        start_date: str = '2016-01-01'
        ):
        self.ticker = ticker
        self.start_date = start_date
        try:
            self.company_name = yf.Ticker(ticker).get_info()['name']
        except:
            self.company_name = yf.Ticker(ticker).get_info()['longName']
        self.crypto_hist_df = self.get_crypto_hist_df()
        self.crypto_multiplicative = self.get_crypto_multiplicative()
        self.crypto_forcast = self.get_crypto_forcast()


    def get_crypto_hist_df(self):
        df = yf.download(self.ticker, self.start_date, self.today).reset_index()

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
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
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

    def get_crypto_multiplicative(self):

        m = Prophet(
                seasonality_mode="multiplicative" 
            )
        m.fit(self.crypto_hist_df)

        return m


    def get_crypto_forcast(self, prediction_periods: int = 365):

        future = self.crypto_multiplicative.make_future_dataframe(periods = prediction_periods)

        return self.crypto_multiplicative.predict(future)
        

    def plot_forcast(self):
        fig = plot_plotly(self.crypto_multiplicative, self.crypto_forcast)
        fig.show()


    def plot_components(self):
        fig = plot_components_plotly(self.crypto_multiplicative, self.crypto_forcast)
        fig.show()