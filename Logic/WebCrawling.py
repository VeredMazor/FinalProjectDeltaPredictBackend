# Import libraries
from bs4 import BeautifulSoup
from finvizfinance.screener.overview import Overview
import finviz
from finvizfinance.quote import finvizfinance
from flask import jsonify
import csv
from urllib.request import urlopen
from pymongo import MongoClient
import pandas as pd
from urllib.request import urlopen, Request
from datetime import date as date1
import traceback
import csv
from pymongo import MongoClient
from pymongo import MongoClient, aggregation
# importing datetime module
from datetime import *
import dateutil.relativedelta as relativedelta
from datetime import date
from Logic.SentimentAnlysis import sentiment_on_all_files

# create mongoDB refernce and start flask app
cluster = MongoClient(
    "mongodb+srv://DeltaPredict:y8RD27dwwmBnUEU@cluster0.7yz0lgf.mongodb.net/?retryWrites=true&w=majority")
# create DB cluster reference
db = cluster["DeltaPredictDB"]


# get monthly stock news headlines from finviz and perform sentiment analysis
def get_stock_news():
    db.newsHeadlines.drop()
    db.sentimentScores.drop()
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}
    tickers = []
    # create ticker list from top 50 s&p stocks
    with open('top50.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        if len(data) != 0:
            for i in data[:]:
                tickers.append(i[0])
    try:
        # scrape stock news for top 50 S&P500 List
        for ticker in tickers[:50]:
            url = finwiz_url + ticker
            # user Request library to make http request to fetch the news form the API
            req = Request(url=url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
            try:
                response = urlopen(req)
                # Read the contents of the response into 'html' using beautiful soup parser
                content = BeautifulSoup(response, features="lxml")
                # Find 'news-table' in the Soup and load it into 'news_table'
                news_table = content.find(id='news-table')
                # Add the table to the  dictionary
                news_tables[ticker] = news_table

            # if there is  a problem with the ticker
            except:
                news_tables[ticker] = ""

        text_list = []
        # read each stocks data and get only necessary parts
        for t in tickers[:50]:
            stock = news_tables[t]
            try:
                # Get all the table rows tagged in HTML with <tr> into ‘stock_tr’
                stock_tr = stock.findAll('tr')
                for i, table_row in enumerate(stock_tr):
                    dict = {}
                    # Read the text of the element ‘a’ into ‘link_text’
                    a_text = table_row.a.text
                    # Read the text of the element ‘td’ into ‘data_text’
                    td_text = table_row.td.text
                    # update stock symbol in dict
                    dict["ticker"] = t
                    # get previous months name
                    prevMonth = (datetime.now() + relativedelta.relativedelta(months=-1)).strftime("%b")
                    # get all data of this month and append to dict
                    if prevMonth not in td_text:
                        dict["date"] = td_text
                        text_list.append(a_text)
                    else:
                        break
                # append all the news headlines
                dict["text"] = text_list
                df = pd.DataFrame(dict)

                text_list = []
                # df.to_csv('../Logic/newsHeadlines/' + t + ".csv", columns=['text'], index=True)
                if not  df.empty:
                    db.newsHeadlines.insert_many(df.to_dict('records'))
            except  Exception:
                exc = traceback.print_exc()
                # traceback.print_exc()
    except  Exception:
        exc = traceback.print_exc()
    sentiment_on_all_files()


def get_sp_list():
    # Scrape the entire S&P500 list from Wikipedia into a Pandas DataFrame;
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df.to_csv('S&P500-Info.csv')
    df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])


if __name__ == "__main__":
    dict = {}
    total = 0
