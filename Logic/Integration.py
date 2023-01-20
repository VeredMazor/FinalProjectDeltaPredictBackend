from Logic.TechnicalAnalyzerAlgorithms import arima_on_all, monte_carlo_on20
from Logic.SentimentAnlysis import get_sentiment_of_stock


# this function combines the prediciton result of arima model and monte carlo simulation. It returns the 5 stocks with
# the highest growth and highest sentiment
def get_protfolio_recommendation():
    combined = {}
    sortedStocks = {}
    arima = arima_on_all()
    monte_carlo = monte_carlo_on20()
    #make an average between the two predictions
    for i in monte_carlo:
        combined[i["Symbol"]] = (i["delta"] + arima[i["Symbol"]]["delta"]) / 2
        combined[i["Sentiment"]] = get_sentiment_of_stock(item[0]).strip()
    stocks_to_invest = []
    # sort stocks from the largest growth prediction to smallest
    sortedStocks = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:]
    sortedStocks = sorted(combined.items(), key=lambda x: x[2], reverse=True)[:]
    print(sortedStocks)
    return stocks_to_invest[:5]


