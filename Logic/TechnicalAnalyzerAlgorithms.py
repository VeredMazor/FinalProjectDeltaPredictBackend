import pandas as pd
import yfinance as yf
import datetime
from datetime import date
from yahooquery import Ticker
from datetime import datetime as d
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from bokeh.plotting import figure, show, output_notebook
import seaborn as sns
from Logic.SentimentAnlysis import get_sentiment_of_stock
import numpy as np
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import csv
import json
from pmdarima import auto_arima
from statsmodels.tsa.api import SARIMAX, AutoReg
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
from pandas_datareader import data as dr


# using the ARIMA MODEL we calculate weekly prediction with training data of stock prices from 2015 till today
def weekly_armia_model(symbol):
    sns.set()
    #the training data is from 2015 till today
    start_training = datetime.date(2015, 1, 1)
    end_training = datetime.datetime.today()
    ticker = symbol
    #dwnload historical prices
    df = yf.download(ticker, start=start_training, end=end_training, progress=False)
    df = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                               'Adj Close': 'last'})
    df.drop(columns=["Open", "High", "Low", "Close"], inplace=True)
    df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
    # find the optimal parameters using auto arima
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
    arima_results = arima.fit()
    # Make ARIMA forecast of next 10 values
    arima_value_forecast = arima_results.get_forecast(steps=10, information_set="filtered",
                                                      typ='levels').summary_frame()
    arima_value_forecast["dates"] = arima_value_forecast.index.date
    res = {}
    res["dates"] = arima_value_forecast["dates"].values.astype("str").tolist()
    res["mean"] = arima_value_forecast["mean"].values.astype("str").tolist()
    return (res)

# using the auto ARIMA MODELto get daily prediction with training data of stock prices from 2015 till today
def daily_armia_model(symbol):
    # for daily basis
    data = []

    def parser(x):
        return d.strptime(x, '%Y-%m-%d-%H-%M-%S')

    sns.set()
    # dateframes for historical prices
    start_training = datetime.date(2015, 1, 3)
    end_training = datetime.datetime.today()
    ticker = symbol
    # download historical prices
    df = yf.download(ticker, start=start_training, end=end_training, progress=False)
    df = df.reset_index()
    df['Price'] = df['Close']
    Quantity_date = df[['Price', 'Date']]
    # reformat the data to make it more suitable for the arima model
    Quantity_date.index = Quantity_date['Date'].map(lambda x: x)
    pd.set_option('mode.chained_assignment', None)
    Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
    # resample data using forward fill to fill in gaps for days when stock market is closed
    Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
    Quantity_date = Quantity_date.resample('1D').mean().ffill()
    quantity = Quantity_date.values

    # perform auto arima and find optimal parameters
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
    # Make ARIMA forecast of next x steps
    arima_value_forecast = arima_results.get_forecast(steps=8, information_set="filtered",
                                                      typ='levels').summary_frame()
    arima_value_forecast.drop(columns=["mean_se", "mean_ci_lower", "mean_ci_upper"], inplace=True)
    # Convert the DataFrame to dict
    dictionaryObject = arima_value_forecast.to_dict();
    return (dictionaryObject)


# this function performs monte carlo simulation using stock closing prices to predict future values
def monte_carlo(Symbol):
    x = datetime.datetime.now()
    start = dt.datetime(2011, 10, 1)
    end = dt.datetime(x.year, int(x.strftime("%m")), int(x.strftime("%d")))
    print(end)
    data = pd.DataFrame(columns=[Symbol])
    yahooData = yf.download(Symbol, start, end)
    count = len(yahooData)

    for i in range(count):
        data.loc[i] = [yahooData['Adj Close'][i]]

    returns = data.pct_change()
    returns.dropna(inplace=True)
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
    daily = 0
    for j in range(0, cols):
        daily = daily + pred[-29][j]
    result = {"Max": np.max(pred), "Min": np.min(pred), "DailyPrice": daily / 10}
    return result


# for the recommendation-perform arima on all of the relevant stocks
def arima_on_all():
    result = {}
    tickers = []
    sortedStocks = {}
    # read the top 50 stock list
    with open('top50.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        if len(data) != 0:
            for i in data[:20]:
                tickers.append(i[0])
        for ticker in tickers:
            data = Ticker(ticker)
            modules = 'assetProfile earnings defaultKeyStatistics'
            # get financial stock data
            info = data.get_modules(modules)
            financial = data.summary_detail
            predict = daily_armia_model(ticker)["mean"][0]
            close = round(data.price[ticker]["regularMarketPrice"], 3)
            # subtract latest close vaue from the predicted value to get the delta
            result[ticker] = {"predcition": predict, "currentPrice": close, "delta": predict - close}
    return result


# for the recommendation-perform monte carlo simulation on all the relevant stocks
def monte_carlo_on_all():
    result = []
    counter = 0
    with open('top50.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        for symbol in data[:50]:
            if (symbol[0] != 'Symbol'):
                price = monte_carlo(symbol[0])
                symbolTicker = Ticker(symbol[0])
                # print(symbolTicker.price[symbol[0]]["regularMarketPrice"])
                realPrice = symbolTicker.price[symbol[0]]["regularMarketPrice"]
                result.append({"Symbol": symbol[0], "delta": (price["DailyPrice"] - realPrice)})
                # print(result)
    f.close()
    recommendedStocks = []
    for i in range(0, len(result)):
        print(result[i]["Price"])
        if (result[i]["Price"] > 0):
            recommendedStocks.append(result[i])

    print(recommendedStocks)
    return recommendedStocks

# for the recommendation-perform monte carlo simulation on 20  relevant stocks
def monte_carlo_on20():
    result = []
    counter = 0
    # read list of top 50 files
    with open('top50.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        for symbol in data[:20]:
            if (symbol[0] != 'Symbol'):
                # perform monte carlo on a specific stock
                price = monte_carlo(symbol[0])
                symbolTicker = Ticker(symbol[0])
                # get stocks real price
                realPrice = symbolTicker.price[symbol[0]]["regularMarketPrice"]
                result.append({"Symbol": symbol[0], "delta": (price["DailyPrice"] - realPrice)})
    f.close()
    return result
