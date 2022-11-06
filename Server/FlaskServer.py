import data as data
import pymongo
from flask import Flask, render_template, jsonify, url_for, redirect, Response
from flask import request, make_response
from flask_cors import CORS, cross_origin
import datetime
import jsonpickle
import time
import csv
import json
import os
import flask
from yahooquery import Ticker
from pymongo import MongoClient, aggregation
import numpy as np
import jsonpickle
import pandas as pd
from yahooquery import Ticker
# Data Source
import yfinance as yf
from finvizfinance.quote import finvizfinance
from finviz.screener import Screener as stockScreener
from finvizfinance.screener.overview import Overview
from flask import Flask, jsonify, Response
from flask import request
from flask_cors import CORS, cross_origin
from pymongo import MongoClient
# importing  all the
# functions defined in test.py
# import simplejson as json
# get db
from waitress import serve
import sys
from yahooquery import Screener

from Logic.WebCrawling import get_stock_news, get_sp_list
from Logic.TechnicalAnalyzerAlgorithms import  daily_armia_model,weekly_armia_model

sys.path.insert(0, '\FinalProjectDeltaPredictBackend\Logic')


cluster = MongoClient(
    "mongodb+srv://DeltaPredict:y8RD27dwwmBnUEU@cluster0.7yz0lgf.mongodb.net/?retryWrites=true&w=majority")
app = Flask(__name__)
CORS(app)
db = cluster["DeltaPredictDB"]


@app.route("/")
def get_time():
    # Returning an api for showing in  reactjs
    return {
        'Name': "geek",
        "Age": "22",
        "Date": "x",
        "programming": "python"
    }


def get_most(signal):
    # get most active
    foverview = Overview()
    filters_dict = {'Index': 'S&P 500', 'Sector': 'Any'}
    foverview.set_filter(signal=signal, filters_dict=filters_dict)
    try:
        df = foverview.screener_view()
        df.to_csv(signal + '.csv', columns=['Ticker'], mode='w')
    except:
        return


def get_stock_data(symbol):
    res = []
    result_list = []
    try:
        with open(symbol, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            if len(data) != 0:
                for i in data[1:10]:
                    ticker = Ticker(i[1])
                    # company_name = ticker.info['longName']
                    # print(company_name)
                    x = {
                        "symbol": i[1],
                        "close": str(round(ticker.price[i[1]]["regularMarketPrice"], 3))
                    }
                    y = (jsonpickle.encode(x))
                    res.append(y)

    except:
        return res
    return res


def get_specific_stock_data(symbol):
    res = {}
    # ticker = yf.Ticker(symbol)
    # todays_data = ticker.history(period='1d')
    # company_name = ticker.info['longName']
    data = Ticker(symbol)
    modules = 'assetProfile earnings defaultKeyStatistics'
    info=data.get_modules(modules)
    financial=data.summary_detail
    x = {
        # "company": company_name,
        "symbol": symbol,
        "close": str(round(data.price[symbol]["regularMarketPrice"], 3)),
        "high": str(data.price[symbol]["regularMarketDayHigh"]),
        "volume": str(data.price[symbol]['regularMarketVolume']),
        "averageVolume": str(financial[symbol]['averageVolume']),
        "marketCap": str(data.price[symbol]['marketCap']),
        "name": str(data.price[symbol]['longName']),
        "previousClose": str(data.price[symbol]['regularMarketPreviousClose']),
        "dayLow": str(data.price[symbol]['regularMarketDayLow']),
        "info":str(info[symbol]["assetProfile"]["longBusinessSummary"]),
        "industry":str(info[symbol]["assetProfile"]["industry"]),
        "change":' {:+.2%}'.format(data.price[symbol]["regularMarketChangePercent"]),
        "regularMarketChange": ' {:+.2f}'.format(data.price[symbol]["regularMarketChange"], 3),
        "fiftyTwoWeekLow":str(financial[symbol]['fiftyTwoWeekLow']),
        "fiftyTwoWeekHigh":str(financial[symbol]['fiftyTwoWeekHigh']),
        "recommendation":str(data.financial_data[symbol]["recommendationKey"]),
         #"peRatio":str(data.index_trend[symbol]["PeRatio"])
    }
    y = json.dumps(x)
    # res.append(y)
    return y


def get_data(name):
    ticker = yf.Ticker(name)
    todays_data = ticker.history(period='1d')
    stock = finvizfinance(name)
    stock_description = stock.ticker_description()
    x = stock.ticker_fundament()
    x.update({'info': stock_description})
    x.update({'currentPrice': str(round(todays_data['Close'][0], 3))})
    x.update({'volume': str(todays_data['Volume'][0])})
    # x.update({ "logo": str(ticker.info['logo_url'])})
    return json.dumps(x)


def favorites_data(ticker_list):
    d = {}
    all_symbols = " ".join(ticker_list)
    myInfo = Ticker(all_symbols)
    myDict = myInfo.price
    x = []

    for ticker in ticker_list:
        ticker = str(ticker)
        d.update({'currentPrice': str(myDict[ticker]['regularMarketPrice'])})
        d.update({"dayLow": str(myDict[ticker]['regularMarketDayLow'])})
        d.update({"dayHigh": str(myDict[ticker]['regularMarketDayHigh'])})
        d.update({'volume': str(myDict[ticker]['regularMarketVolume'])})
        d.update({"symbol":ticker})
        x.append(json.dumps(d))
    return json.dumps(x)



# gets current data for stocks in favorites screen
@app.route('/favoritesData', methods=['POST'])
@cross_origin()
def getFavoriteStocks():
    req = request.get_json()
    email=req['email']["otherParam"]
    if request.method == 'POST':
        for itm in db.favoriteList.find({"Email": email}):
            if (itm.get('Email') == email):
                return favorites_data(itm.get('FavoriteStocks'))


@app.route('/fundamental', methods=['POST'])
@cross_origin()
def getData():
    req = request.get_json()
    if flask.request.method == 'POST':
        return get_specific_stock_data(req["Symbol"])


@app.route('/activeStockData', methods=['GET'])
@cross_origin()
def getMostActive():
    if flask.request.method == 'GET':
        return get_stock_data('Most Active.csv')


@app.route('/arimaResults', methods=['POST'])
@cross_origin()
def getArimaARes():
    result={}
    req = request.get_json()
    if flask.request.method == 'POST':
        result["weekly"]=weekly_armia_model(req["Symbol"])
        result["daily"] = daily_armia_model(req["Symbol"])
        return result


@app.route('/spesificStock', methods=['POST'])
@cross_origin()
def specific_data():
    req = request.get_json()
    if request.method == 'POST':
        return get_specific_stock_data(req["Symbol"])


@app.route('/losersStockData', methods=['GET'])
@cross_origin()
def getTopLosers():
    if flask.request.method == 'GET':
        return get_stock_data('Top Losers.csv')


@app.route('/gainersStockData', methods=['GET'])
@cross_origin()
def get_top_gainers():
    if flask.request.method == 'GET':
        return get_stock_data('Top Gainers.csv')


@app.route('/authenticate', methods=['GET', 'POST'])
@cross_origin()
def check():
    req = request.get_json()
    if request.method == 'POST':
        # check if login details are correct
        if db.users.count_documents({'Email': req["name"], 'Password': req["Password"]}, limit=1) != 0:
            return jsonify({'result': "true"})
        return jsonify({'result': "false"})
        # insert to DB
        # insert = {'userName': req["name"], 'Password': req["Password"]}
        # db.users.insert_one(insert)


    elif request.method == 'GET':
        json_string = "{'a': 1, 'b': 2}"
        return Response(json_string, mimetype='application/json')


@app.route('/signnup', methods=['GET', 'POST'])
@cross_origin()
def addUser():
    req = request.get_json()
    if request.method == 'POST':

        if db.users.count_documents({'Email': req["Email"], 'Password': req["Password"]}, limit=1) != 0:
            return jsonify({'result': "false"})
        else:
            insert = {'Email': req["Email"], 'Password': req["Password"]}
            db.users.insert_one(insert)
            return jsonify({'result': "true"})


@app.route('/getuser', methods=['GET', 'POST'])
@cross_origin()
def getUserData():
    req = request.get_json()
    if request.method == 'POST':
        print(req['otherParam'])
        for itm in db.favoriteList.find({"Email": req['otherParam']}):
            if (itm.get('Email') == req['otherParam']):
                print(itm.get('FavoriteStocks'))
                return jsonify({'result': itm.get('FavoriteStocks')})  # TO DO return the stocks list favorit to user

        print(req['otherParam'])
        insert = {'Email': req['otherParam'], 'FavoriteStocks': []}
        db.favoriteList.insert_one(insert)
        return ("itm.get('_id')")

def get_sector_stocks(sector):
    sec = "sec_" + sector["name"]
    filters = ['idx_sp500','exch_nasd', sec, 'geo_usa']  # Shows companies in NASDAQ which are in the S&P500
    stock_list = stockScreener(filters=filters, table='Overview', order='price')  # Get the performance ta
    return json.dumps(stock_list.data)



@app.route('/getSectorStocks', methods=['GET', 'POST'])
@cross_origin()
def get_Sector_Data():
    req = request.get_json()
    if request.method == 'POST':
        return get_sector_stocks(req['Sector'])



def spList():
    try:
        f = open("S&P500-Symbols.csv")
        # Do something with the file
    except IOError:
        print("File not accessible")
        get_sp_list()
    finally:
        f.close()


if __name__ == "__main__":
    #app.run(debug=True)
    spList()
    s = Screener()
    # get_stock_news()
    serve(app, host="0.0.0.0", port=5000,threads=20)
    get_most('Most Active')
    get_most('Top Gainers')
    get_most('Top Losers')

    # app.run(threaded=True)


def create_app():
    return app