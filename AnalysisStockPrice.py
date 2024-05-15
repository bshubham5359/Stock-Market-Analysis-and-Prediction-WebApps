# load library
import streamlit as st
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import yfinance as yf
import datetime

# print title of web app
st.title("Stock Market Analysis and Prediction")
st.markdown("> Stock Market Analysis and Prediction is the project on technical analysis, visualization and prediction using data provided by Yahoo Finance.")
st.markdown("> It is web app which predicts the future value of company stock or other ﬁnancial instrument traded on an exchange.")

# select any stock for analyse
stock_option = st.selectbox(
    "Which Stock  you want to analyse ?",
    ('TCS.NS', 'CTSH','INFY.NS','WIPRO.NS'))

# metadata for seleted stock
st.subheader('MetaData of Selected Stock:')
msft=yf.Ticker(stock_option)
stock_info = msft.info
if st.checkbox('show detailed financials:'):
# show financials:
    st.subheader('Stock Details')
    st.write("Name:", stock_info["longName"])
    st.write("Symbol:", stock_info["symbol"])
    st.write("Sector:", stock_info["sector"])
    st.write("Industry:", stock_info["industry"])
    st.write("Address:", stock_info["address1"],stock_info["address2"],stock_info["city"],stock_info["zip"],stock_info["country"])
    st.write("Website:", stock_info["website"])
    if st.checkbox('show wiki:'):
        st.write(stock_info["longBusinessSummary"])
# - income statement
    st.subheader('Income Statement')
    st.write(msft.income_stmt)
# - balance sheet
    st.subheader('Balance Sheet')
    st.write(msft.balance_sheet)
# - cash flow statement
    st.subheader('Cash Flow Statement')
    st.write(msft.cashflow)
    # get historical market data
    st.subheader('Historical Data-1month')
    hist = msft.history(period="1mo")
    st.write(hist)

# Load data from yahoo finance.
start=dt.date(2010,1,1)
end=dt.date.today()
#data=pdr.get_data_yahoo("GOOG", start, end)
data=yf.download(stock_option, start, end)

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

#fill nan vale with next value within columns
data.fillna(method="ffill",inplace=True)

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

# create checkbox
st.subheader("Want to show raw data:")
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
 
# show the description of data
st.subheader('Detail description about Datasets:')
descrb=data.describe()
st.write(descrb)

#create new columns like year, month, day
data["Year"]=data.index.year
data["Month"]=data.index.month
data["Weekday"]=data.index.day_name()

# dislay graph of open and close column
st.subheader('Graph of Close & Open:')
st.line_chart(data[["Open","Close"]])

# display plot of Adj Close column in datasets
st.subheader('Graph of Adjacent Close:')
st.line_chart(data['Adj Close'])

# display plot of volume column in datasets
st.subheader('Graph of Volume:')
st.line_chart(data['Volume'])

# create new cloumn for data analysis.
data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
data['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0
data = data[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]

# display the new dataset after modificaton
st.subheader('Newly format DataSet:')
st.dataframe(data.tail(500))

pd.set_option("mode.copy_on_write", True)
pd.options.mode.copy_on_write = True

forecast_col = 'Adj Close'
forecast_out = int(math.ceil(0.01 * len(data)))
data['label'] = data[forecast_col].shift(-forecast_out)

X = np.array(data.drop('label', axis=1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
data.dropna(inplace=True)
y = np.array(data['label'])

# split dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

# display the accuracy of forecast value.
st.subheader('Model Accuracy:')
st.write(confidence)

forecast_set = clf.predict(X_lately)
data['Forecast'] = np.nan

last_date = data.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)]+[i]
    last_date = data.iloc[-1].name
    dti = pd.date_range(last_date, periods=forecast_out+1, freq='D')
    index = 1
for i in forecast_set:
    data.loc[dti[index]] = [np.nan for _ in range(len(data.columns)-1)] + [i]
    index +=1

# display the forecast value.
st.subheader('Forecast value :')
st.dataframe(data.tail(50))

# display the graph of adj close and forecast columns
st.subheader('Graph of Adj Close and Forecast :-')
st.line_chart(data[["Adj Close","Forecast"]])

st.success('Done!')


st.text('© 2022 Stock-Market-Analysis-and-Prediction-WebApps')