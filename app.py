# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics
from plotly import graph_objs as go


START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'FB', 'AMZN', 'TSLA', 'GLD', 'SLV', 'BTC-USD', 'ETH-USD')
selected_stock = st.selectbox('Select stock for prediction', stocks, index=7)

try:
    company_name = yf.Ticker(selected_stock).get_info()['name']
    is_crypto = True
except:
    company_name = yf.Ticker(selected_stock).get_info()['shortName']
    is_crypto = False


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY).reset_index()
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader(f'{company_name}: Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text=f'{company_name}: Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


# Predict forecast with Prophet.
st.subheader('Forecast data')

n_years = st.slider('Years of prediction:', 1, 3)
period = n_years * 365

df_train = data[['Date','Open']]
df_train = df_train.rename(columns={"Date": "ds", "Open": "y"})

if is_crypto:
    m = Prophet(seasonality_mode="multiplicative")
else:
    m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(
    m, 
    forecast,
    xlabel="Prediction Window",
    ylabel=f"{company_name} Price ($)"
    )
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


# Cross Validation Diagnose
run_cv = st.checkbox("Run cross validation")
cv_state = st.text("")
if run_cv:
    cv_state.text('Cross validation in process...')

    df_cv = cross_validation(m, initial='30 days', period='18 days', horizon='65 days')
    df_p = performance_metrics(df_cv)
    fig3 = plot_cross_validation_metric(df_cv, metric='rmse')
    st.write(fig3)

    cv_state.text('Cross validation... done!')
else:
    cv_state.text("")