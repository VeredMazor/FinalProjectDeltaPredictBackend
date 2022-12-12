from Logic.TechnicalAnalyzerAlgorithms import arima_on_all, monte_carlo_on20
from Logic.SentimentAnlysis import get_sentiment_of_stock


def get_protfolio_recommendation():
    combined={}
    sortedStocks = {}
    arima=arima_on_all()
    monte_carlo=monte_carlo_on20()
    for i in monte_carlo:
        combined[i["Symbol"]]=(i["delta"]+arima[i["Symbol"]]["delta"])/2
    stocks_to_invest = []
    # # sort stocks from largest growth prediction to smallest
    sortedStocks = sorted(combined.items(), key=lambda x:  x[1], reverse=True)[:]
    #print(sortedStocks)
    for item in sortedStocks:
        #if stock has positive sentiment add to stocks to invest
        if float(get_sentiment_of_stock(item[0]).strip())>0:
            stocks_to_invest.append(str(item[0]))
    print(stocks_to_invest[:5])
    # !/usr/bin/python
    #
    # import smtplib
    #
    # sender = 'irbtebh@yahoo.com'
    # receivers = ['irisgrabois@gmail.com']
    #
    # message = stocks_to_invest[:5]
    #
    # try:
    #     smtpObj = smtplib.SMTP('localhost')
    #     smtpObj.sendmail(sender, receivers, message)
    #     print( "Successfully sent email")
    #
    # except SMTPException:
    #     print(  "Error: unable to send email")
    #     "Error: unable to send email"
    return stocks_to_invest[:5]

    #return combined
if __name__ == "__main__":
    # daily_armia_model("A")
    # print(daily_armia_model("F"))
    print(get_protfolio_recommendation())
    # ************** PREPROCESSUNG ***********************