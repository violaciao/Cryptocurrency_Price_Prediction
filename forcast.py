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
            self.is_crypto = True
        except:
            self.company_name = yf.Ticker(ticker).get_info()['longName']
            self.is_crypto = False
        self.crypto_hist_df = self.get_crypto_hist_df()
        self.model = self.fit_model()
        self.crypto_forcast = self.get_crypto_forcast()


    def run(self):
        # self.plot_open_prices()
        self.plot_forcast()
        # self.plot_components()


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

    def fit_model(self):

        if self.is_crypto:
            m = Prophet(
                    seasonality_mode="multiplicative" 
                )
        else:
            m = Prophet()

        m.fit(self.crypto_hist_df)

        return m


    def get_crypto_forcast(self, prediction_periods: int = 365):

        if self.is_crypto:
            future = self.model.make_future_dataframe(
                periods = prediction_periods
                )
        else:
            future = self.model.make_future_dataframe(
                periods = prediction_periods, 
                freq='B'  # forecasting only business days
                )
    
        return self.model.predict(future)
        

    def plot_forcast(self):
        fig = plot_plotly(
            self.model, 
            self.crypto_forcast, 
            xlabel="Prediction Window",
            ylabel=f"{self.company_name} Price ($)"
            )
        fig.show()


    def plot_components(self):
        fig = plot_components_plotly(self.model, self.crypto_forcast)
        fig.show()