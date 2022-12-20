from Logic.TechnicalAnalyzerAlgorithms import arima_on_all, monte_carlo_on20
from Logic.SentimentAnlysis import get_sentiment_of_stock


def get_protfolio_recommendation():
    combined=[]
    sortedStocks = {}
    arima=arima_on_all()
    print(arima)
    monte_carlo=monte_carlo_on20()
    for i in range(20):
        combined.append({"symbol":monte_carlo[i]["Symbol"], "average": (monte_carlo[i]["delta"] + arima[(monte_carlo[i]["Symbol"])]["delta"]) / 2,
                                 "sentiment": get_sentiment_of_stock(monte_carlo[i]["Symbol"])})
    stocks_to_invest = []
    # # sort stocks from largest growth prediction to smallest
    sortedStocks = sorted(combined, key=lambda x:  (x["average"], x["sentiment"]), reverse=True)[:]
    print(sortedStocks)
    for stock  in sortedStocks:
        stocks_to_invest.append(stock["symbol"])
    #print(sortedStocks)
    return stocks_to_invest[:5]

    #return combined
if __name__ == "__main__":
    # daily_armia_model("A")
    # print(daily_armia_model("F"))
    print(get_protfolio_recommendation())
    # ************** PREPROCESSUNG ***********************