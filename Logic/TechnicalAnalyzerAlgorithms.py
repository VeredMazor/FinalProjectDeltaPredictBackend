import pandas as pd
import yfinance as yf
import datetime
from yahooquery import Ticker
from datetime import datetime as d
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from bokeh.plotting import figure, show, output_notebook
import matplotlib.pyplot as plt
import seaborn as sns
from Logic.SentimentAnlysis import get_sentiment_of_stock
import numpy as np
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import SARIMAX, AutoReg
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
from pandas_datareader import data as dr


# If the test statistic of a time series is less than the critical value of a time series,
# the time series is stationary. If the p-value is less than or equal to 0.05, the time series is also considered stationary.


def weekly_armia_model(symbol):
    sns.set()
    # get data of stock
    start_training = datetime.date(2015, 1, 1)
    end_training = datetime.datetime.today()
    start_testing = datetime.date(2022, 6, 1)
    end_testing = datetime.datetime.today()
    ticker = symbol
    df = yf.download(ticker, start=start_training, end=end_training, progress=False)
    print(df.shape)
    df = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                               'Adj Close': 'last'})
    df.drop(columns=["Open", "High", "Low", "Close"], inplace=True)
    df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
    # print(auto_arima(df["adj_close"], m=7).summary())
    arima_model = auto_arima(df["adj_close"], start_p=0, d=1, start_q=0,
                             max_p=2, max_d=2, max_q=2, start_P=0,
                             D=1, start_Q=0, max_P=2, max_D=2,
                             max_Q=2, m=12, seasonal=True,
                             error_action='warn', trace=True,
                             supress_warnings=True, stepwise=True,
                             random_state=20, n_fits=10)

    # Summary of the model
    arima = ARIMA(df["adj_close"], order=arima_model.order, seasonal_order=arima_model.seasonal_order,
                  enforce_stationarity="True")
    print(arima_model.summary())
    print(pd.DataFrame(arima_model.predict(n_periods=20)))
    arima_results = arima.fit()
    # print(arima_results.summary())
    # Obtain predicted values
    # Make ARIMA forecast of next 10 values
    arima_value_forecast = arima_results.get_forecast(steps=10, information_set="filtered",
                                                      typ='levels').summary_frame()
    arima_value_forecast["dates"] = arima_value_forecast.index.date
    res = {}
    res["dates"] = arima_value_forecast["dates"].values.astype("str").tolist()
    res["mean"] = arima_value_forecast["mean"].values.astype("str").tolist()
    return (res)
    # plt.plot(arima_value_forecast, label="Predicted")


def daily_armia_model(symbol):
    # for daily basis
    data = []

    def parser(x):
        return d.strptime(x, '%Y-%m-%d-%H-%M-%S')

    sns.set()
    start_training = datetime.date(2015, 1, 3)
    end_training = datetime.datetime.today()
    start_testing = datetime.date(2022, 6, 1)
    end_testing = datetime.datetime.today()
    ticker = symbol
    df = yf.download(ticker, start=start_training, end=end_training, progress=False)
    df = df.reset_index()
    df['Price'] = df['Close']
    Quantity_date = df[['Price', 'Date']]
    Quantity_date.index = Quantity_date['Date'].map(lambda x: x)
    pd.set_option('mode.chained_assignment', None)
    Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
    Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
    # Quantity_date.index = pd.DatetimeIndex(Quantity_date["Date"])
    Quantity_date = Quantity_date.drop(['Date'], axis=1)
    quantity = Quantity_date.values
    # quantity.index=pd.DatetimeIndex(quantity["Date"])
    # df.set_index(df["Open"], drop=True, append=False, inplace=False, verify_integrity=False)
    # df.drop(columns=["Open", "High", "Low", "Close","Volume"], inplace=True)
    # df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)

    arima_model = auto_arima(quantity, start_p=0, d=1, start_q=0,
                             max_p=2, max_d=2, max_q=2, start_P=0,
                             D=1, start_Q=0, max_P=2, max_D=2,
                             max_Q=2, m=12, seasonal=True,
                             error_action='warn', trace=True,
                             supress_warnings=True, stepwise=True,
                             random_state=20, n_fits=20)

    # get best model orders
    print(arima_model.summary().tables[0][1][1])
    arima = ARIMA(quantity, order=arima_model.order, seasonal_order=arima_model.seasonal_order,
                  enforce_stationarity="True")
    # Fit ARIMA model
    arima_results = arima.fit()
    print(arima_results.forecast())
    # Obtain predicted values
    # Make ARIMA forecast of next x steps
    arima_value_forecast = arima_results.get_forecast(steps=10, information_set="filtered",
                                                      typ='levels').summary_frame()
    # Print forecast
    # print(arima_value_forecast)
    # plt.plot(arima_value_forecast["mean"], label="Predicted")
    arima_value_forecast.drop(columns=["mean_se", "mean_ci_lower", "mean_ci_upper"], inplace=True)
    # Convert the DataFrame to dict
    dictionaryObject = arima_value_forecast.to_dict();
    return (dictionaryObject)


def monte_carlo(Symbol):
    # ticker = 'NVDA'  # GOOGLE stock ticker
    data = pd.DataFrame(columns=[Symbol])
    data[Symbol] = dr.DataReader(Symbol, data_source='yahoo', start='2008-1-1', end='2022-03-23')['Adj Close']
    # print(data.head())

    returns = data.pct_change()
    returns.dropna(inplace=True)
    # print(returns.head())

    l = norm.ppf(0.10)
    u = norm.ppf(0.85)

    mean = returns.mean()
    stdev = returns.std()
    np.random.seed(42)
    n = np.random.normal(size=(30, 10))
    rows = n.shape[0]
    cols = n.shape[1]
    for i in range(0, rows):
        for j in range(0, cols):
            if n[i][j] > u:
                n[i][j] = u  # sets upper limit
            elif n[i][j] < l:
                n[i][j] = l  # sets lower limit
            else:
                n[i][j] = n[i][j]
            n[i][j] = (stdev * n[i][j]) + mean
    s = data.iloc[-1]
    pred = np.zeros_like(n) + 1
    pred[0] = s  # sets beginning point of simulations
    for i in range(1, 30):
        pred[i] = pred[(i - 1)] * (1 + n[(i - 1)])

    # print('\n')
    # print('Maximum Simulated Price : {}'.format(np.max(pred)))
    # print('Minimum Simulated Price : {}'.format(np.min(pred)))
    # for j in range(0, cols):
    #     print('Simulated Close Prices after 30 days : {}'.format(pred[-1][j]))
    result = {"Max": np.max(pred), "Min": np.min(pred)}
    return result


def arima_on_all():
    result = {}
    tickers = []
    sortedStocks={}
    #read list of 50 top stocks and make prediction only on 20
    with open('top50.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        if len(data) != 0:
            for i in data[:20]:
                tickers.append(i[0])
        # scrape stock news for top 50 S&P500 List
        for ticker in tickers:
            data = Ticker(ticker)
            modules = 'assetProfile earnings defaultKeyStatistics'
            info = data.get_modules(modules)
            financial = data.summary_detail

            predict = daily_armia_model(ticker)["mean"][0]
            close = round(data.price[ticker]["regularMarketPrice"], 3)
            result[ticker] = {"predcition": predict, "currentPrice": close, "delta": predict - close}
    stocks_to_invest=[]
    # # sort stocks from largest growth prediction to smallest
    # sortedStocks = sorted(result.items(), key=lambda x:  x[1]['delta'], reverse=True)[:]
    # print(sortedStocks)
    # for item in sortedStocks:
    #     #if stock has positive sentiment add to stocks to invest
    #     if float(get_sentiment_of_stock(item[0]).strip())>0:
    #         stocks_to_invest.append(str(item[0]))
    # print(stocks_to_invest)
    # return stocks_to_invest
    return result


def monte_carlo_on_all():
    result=[]
    counter = 0
    with open('top50.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        for symbol in data[:50]:
            if(symbol[0] != 'Symbol'):
                price = monte_carlo(symbol[0])
                symbolTicker = Ticker(symbol[0])
                #print(symbolTicker.price[symbol[0]]["regularMarketPrice"])
                realPrice = symbolTicker.price[symbol[0]]["regularMarketPrice"]
                result.append({"Symbol" : symbol[0],  "delta" : ((price["Max"] + price["Min"]) / 2) - realPrice})
                #print(result)
    f.close()
    recommendedStocks = []
    for i in range(0, len(result)):
        print(result[i]["Price"])
        if (result[i]["Price"]>0):
            recommendedStocks.append(result[i])

    print(recommendedStocks)
    return recommendedStocks

def monte_carlo_on20():
    result=[]
    counter = 0
    with open('top50.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        for symbol in data[:20]:
            if(symbol[0] != 'Symbol'):
                price = monte_carlo(symbol[0])
                symbolTicker = Ticker(symbol[0])
                #print(symbolTicker.price[symbol[0]]["regularMarketPrice"])
                realPrice = symbolTicker.price[symbol[0]]["regularMarketPrice"]
                result.append({"Symbol" : symbol[0],  "delta" : ((price["Max"] + price["Min"]) / 2) - realPrice})
                #print(result)
    f.close()
    return result

if __name__ == "__main__":
    # daily_armia_model("A")
    # print(daily_armia_model("F"))
    print(arima_on_all())
    # ************** PREPROCESSUNG ***********************