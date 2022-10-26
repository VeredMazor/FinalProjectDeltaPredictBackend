import pandas as pd
import yfinance as yf
import datetime
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
import matplotlib.pyplot as plt


def armiaModel():
    global arima_fcast
    sns.set()
    start_training = datetime.date(2010, 1, 1)
    end_training = datetime.date(2022, 10, 25)
    start_testing = datetime.date(2022, 6, 1)
    end_testing = datetime.datetime.today()
    ticker = "AAPL"
    df = yf.download(ticker, start=start_training, end=end_training, progress=False)
    print(df.shape)
    df = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                                                 'Adj Close': 'last'})
    df.drop(columns=["Open", "High", "Low", "Close"], inplace=True)
    df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
    # print(df_training.tail())
    #df=df[["Adj Close"]]
    train = df.iloc[:-30]
    test = df.iloc[-30:]
    print(train.shape, test.shape)
    print(test.iloc[0], test.iloc[-1])
    from statsmodels.tsa.statespace.sarimax  import SARIMAX
    arima = SARIMAX(train["adj_close"], order=(2, 0, 2),enforce_stationarity=True)
    arima.initialize_stationary()
    # Fit ARIMA model
    arima_results = arima.fit()

    # Make ARIMA forecast of next 10 values
    arima_value_forecast = arima_results.get_forecast(steps=50,information_set="filtered").summary_frame()
    fig, ax = plt.subplots(figsize=(15, 5))
    arima_value_forecast['mean'].plot(ax=ax, style='k--')
    ax.fill_between(arima_value_forecast.index, arima_value_forecast['mean_ci_lower'], arima_value_forecast['mean_ci_upper'], color='k', alpha=0.1);
    # Print forecast
    print(arima_value_forecast)

    # df_training = yf.download(ticker, start=start_training, end=end_training, progress=False)
    # print(f"Downloaded {df_training.shape[0]} rows and {df_training.shape[1]} columns of {ticker} data")
    # df_training.tail()
    # ## Resampling to obtain weekly stock prices with the following rules
    # ## 'Open': first opening price of the month
    # ## 'High': max price of the month
    # ## 'Low': min price of the month of the month
    # ## 'Close' : closing price of the month
    # ## 'Adj Close' : adjusted closing price of the month
    #
    # df_training = df_training.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
    #                                              'Adj Close': 'last'})
    # df_training.drop(columns=["Open", "High", "Low", "Close"], inplace=True)
    # df_training.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
    # # print(df_training.tail())
    # start_training_str = (start_training + pd.Timedelta("5 days")).strftime("%B %Y")
    # end_training_str = (end_training - pd.Timedelta("5 days")).strftime("%B %Y")
    # sns.set(font_scale=1.2)
    # df_training['adj_close'].plot(figsize=(12, 8),
    #                               title=f"{ticker} weekly adjusted close prices ({start_training_str} - {end_training_str})")
    # ## Fitting the model(With more tuning of the parameters)
    # arima_fit = pm.auto_arima(df_training['adj_close'], error_action='raise', suppress_warnings=True, stepwise=True,
    #                           approximation=False, seasonal=True)
    #
    # ## Printing a summary of the model
    # #print(arima_fit.summary())
    # df_testing = yf.download(ticker, start=start_testing, end=end_testing, progress=False)
    # #print(f"Downloaded {df_testing.shape[0]} rows and {df_testing.shape[1]} columns of {ticker} data")
    # f_testing = df_testing.resample('W').agg(
    #     {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last'})
    # df_testing.drop(columns=["Open", "High", "Low", "Close"], inplace=True)
    # df_testing.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
    # # print(df_testing.head())
    #
    #
    # n_fcast1 = len(df_testing)
    # print(arima_fit.predict(n_periods=n_fcast1,alpha=0.05,return_conf_int=True))
    # show(plot_arima(  df_training['adj_close'],next_25))
    # arima_fcast = arima_fit.predict(n_periods=n_fcast1, return_conf_int=True, alpha=0.05)
    # print(arima_fcast)
    # arima_fcast = [pd.DataFrame(arima_fcast(), columns=['prediction'])]
    # arima_fcast = pd.concat(arima_fcast, axis=1).set_index(df_testing.index)
    # arima_fcast.head()
    # fig, ax = plt.subplots(1, figsize=(12, 8))
    #
    # ax = sns.lineplot(data=df_testing['adj_close'], color='black', label='Actual')
    #
    # ax.plot(arima_fcast.prediction, color='red', label='ARIMA(3, 1, 2)')
    #
    # ax.fill_between(arima_fcast.index, arima_fcast.lower_95,
    #                 arima_fcast.upper_95, alpha=0.2,
    #                 facecolor='red')
    #
    # ax.set(title=f"{ticker} stock price - actual vs. predicted", xlabel='Date',
    #        ylabel='Adjusted close price (US$)')
    # ax.legend(loc='upper left')
    #
    # plt.tight_layout()
    # plt.show()


def plot_arima(truth, forecasts, title="ARIMA", xaxis_label='Time',
               yaxis_label='Value', c1='#A6CEE3', c2='#B2DF8A',
               forecast_start=None, **kwargs):
    # make truth and forecasts into pandas series
    n_truth = truth.shape[0]
    n_forecasts = forecasts.shape[0]

    # always plot truth the same
    truth = pd.Series(truth, index=np.arange(truth.shape[0]))

    # if no defined forecast start, start at the end
    if forecast_start is None:
        idx = np.arange(n_truth, n_truth + n_forecasts)
    else:
        idx = np.arange(forecast_start, n_forecasts)
    forecasts = pd.Series(forecasts, index=idx)

    # set up the plot
    p = figure(title=title, plot_height=400, **kwargs)
    p.grid.grid_line_alpha = 0.3
    p.xaxis.axis_label = xaxis_label
    p.yaxis.axis_label = yaxis_label

    # add the lines
    p.line(truth.index, truth.values, color=c1, legend_label='Observed')
    p.line(forecasts.index, forecasts.values, color=c2, legend_label='Forecasted')

    return p

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
    armiaModel()



