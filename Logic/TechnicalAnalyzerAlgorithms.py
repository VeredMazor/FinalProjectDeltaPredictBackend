import pandas as pd
import yfinance as yf
import datetime
from datetime import datetime as d
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from bokeh.plotting import figure, show, output_notebook
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import SARIMAX, AutoReg
from statsmodels.tsa.arima.model import ARIMA
from alpha_vantage.timeseries import TimeSeries



# If the test statistic of a time series is less than the critical value of a time series,
# the time series is stationary. If the p-value is less than or equal to 0.05, the time series is also considered stationary.

def adf_test(series, title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC')  # .dropna() handles differenced data

    labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    out = pd.Series(result[0:4], index=labels)

    for key, val in result[4].items():
        out[f'critical value ({key})'] = val

    print(out.to_string())  # .to_string() removes the line "dtype: float64"

    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


def weekly_armia_model(symbol):
    sns.set()
    #get data of stock
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
    #print(arima_results.summary())
    # Obtain predicted values
    # Make ARIMA forecast of next 10 values
    arima_value_forecast = arima_results.get_forecast(steps=10, information_set="filtered",typ='levels').summary_frame()
    arima_value_forecast["dates"] = arima_value_forecast.index.date
    res={}
    res["dates"]= arima_value_forecast["dates"].values.astype("str").tolist()
    res["mean"]=arima_value_forecast["mean"].values.astype("str").tolist()
    return (res)
    #plt.plot(arima_value_forecast, label="Predicted")

def daily_armia_model(symbol):
    # for daily basis
    data=[]
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
    #quantity.index=pd.DatetimeIndex(quantity["Date"])
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

    #get best model orders
    print(arima_model.summary().tables[0][1][1])
    arima = ARIMA(quantity, order= arima_model.order, seasonal_order=arima_model.seasonal_order , enforce_stationarity="True")
    # Fit ARIMA model
    arima_results = arima.fit()
    print(arima_results.forecast())
    # Obtain predicted values
    # Make ARIMA forecast of next x steps
    arima_value_forecast = arima_results.get_forecast(steps=10, information_set="filtered", typ='levels').summary_frame()
    # Print forecast
    #print(arima_value_forecast)
    #plt.plot(arima_value_forecast["mean"], label="Predicted")
    arima_value_forecast.drop(columns=["mean_se", "mean_ci_lower", "mean_ci_upper"], inplace=True)
    # Convert the DataFrame to dict
    dictionaryObject = arima_value_forecast.to_dict();
    return (dictionaryObject)



def monte_carlo():
    start = dt.datetime(2011, 1, 1)
    end = dt.datetime(2021, 1, 1)
    stock_data = yf.download('MSFT', start, end)
    returns = stock_data['Adj Close'].pct_change()
    daily_vol = returns.std()

    T = 252
    count = 0
    price_list = []
    last_price = stock_data['Adj Close'][-1]

    price = last_price * (1 + np.random.normal(0, daily_vol))
    price_list.append(price)

    for y in range(T):
        if count == 251:
            break
        price = price_list[count] * (1 + np.random.normal(0, daily_vol))
        price_list.append(price)
        count += 1

    plt.plot(price_list)
    plt.show()

    NUM_SIMULATIONS = 1000
    df = pd.DataFrame()
    last_price_list = []
    for x in range(NUM_SIMULATIONS):
        count = 0
        price_list = []
        price = last_price * (1 + np.random.normal(0, daily_vol))
        price_list.append(price)

        for y in range(T):
            if count == 251:
                break
            price = price_list[count] * (1 + np.random.normal(0, daily_vol))
            price_list.append(price)
            count += 1

        df[x] = price_list
        last_price_list.append(price_list[-1])

    fig = plt.figure()
    fig.suptitle("Monte Carlo Simulation: MSFT")
    plt.plot(df)
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.show()

    print("Expected price: ", round(np.mean(last_price_list), 2))
    print("Quantile (5%): ", np.percentile(last_price_list, 5))
    print("Quantile (95%): ", np.percentile(last_price_list, 95))


if __name__ == "__main__":
    # daily_armia_model("A")
    print("A")
    # ************** PREPROCESSUNG ***********************
    weekly_armia_model("F")

