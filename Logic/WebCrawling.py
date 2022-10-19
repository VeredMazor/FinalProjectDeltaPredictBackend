import csv
from urllib.request import urlopen

from bs4 import BeautifulSoup
from finvizfinance.screener.overview import Overview
import finviz
from finvizfinance.quote import finvizfinance
from flask import jsonify
# Import libraries
from pymongo import MongoClient
import pandas as pd
from urllib.request import urlopen, Request
from datetime import date as date1
import traceback

# importing datetime module
from datetime import *

from Logic.SentimentAnlysis import sentiment_on_all_files


def get_stock_news():
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}
    import csv
    tickers = []
    with open('S&P500-Symbols.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        if len(data) != 0:
            for i in data[1:]:
                tickers.append(i[1])
    try:
        # scrape stock news for S&P500 List
        for ticker in tickers[:150]:
            url = finwiz_url + ticker
            req = Request(url=url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
            try:
                response = urlopen(req)
                # Read the contents of the file into 'html'
                html = BeautifulSoup(response, features="lxml")
                # Find 'news-table' in the Soup and load it into 'news_table'
                news_table = html.find(id='news-table')
                # Add the table to our dictionary
                news_tables[ticker] = news_table

            except:
                news_tables[ticker] = ""

        text_list = []
        # Read one single day of headlines for ‘’
        for t in tickers[:150]:
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
                    dict["ticker"] = t
                    currentMonth = datetime.now().month
                    if "Oct" in td_text:
                        dict["date"] = td_text
                        text_list.append(a_text)

                dict["text"] = text_list
                df = pd.DataFrame(dict)
                text_list = []
                df.to_csv('../Logic/newsHeadlines/' + t + ".csv", columns=['text'], index=True)
            except  Exception:
                traceback.print_exc()
    except  Exception:
        traceback.print_exc()
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
    get_stock_news()

