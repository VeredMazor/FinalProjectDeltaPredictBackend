import csv
import data as data
import datetime
import flask
import json
import jsonpickle
import jsonpickle
import numpy as np
import os
import pandas as pd
import pymongo
import socket
import sys
import time
# Data Source
import yfinance as yf
from finviz.screener import Screener as stockScreener
from finvizfinance.quote import finvizfinance
from finvizfinance.screener.overview import Overview
from flask import Flask, jsonify, Response
from flask import Flask, render_template, jsonify, url_for, redirect, Response
from flask import request
from flask import request, make_response
from flask_cors import CORS, cross_origin
from flask_cors import CORS, cross_origin
from flask_mail import Mail, Message
from pymongo import MongoClient
from pymongo import MongoClient, aggregation
# importing  all the
# functions defined in test.py
# import simplejson as json
# get db
from waitress import serve
from yahooquery import Screener
from yahooquery import Ticker

from Logic.SentimentAnlysis import get_sentiment_of_stock
from Logic.TechnicalAnalyzerAlgorithms import daily_armia_model, weekly_armia_model, monte_carlo
from Logic.WebCrawling import get_stock_news, get_sp_list
from Logic.Integration import get_protfolio_recommendation
import smtplib, ssl

sys.path.insert(0, '\FinalProjectDeltaPredictBackend\Logic')

# create mongoDB refernce and start flask app
cluster = MongoClient(
    "mongodb+srv://DeltaPredict:y8RD27dwwmBnUEU@cluster0.7yz0lgf.mongodb.net/?retryWrites=true&w=majority")
app = Flask(__name__)
CORS(app)
# create DB cluster reference
db = cluster["DeltaPredictDB"]
# conigure email parameters
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'irisgrabois@gmail.com'
app.config['MAIL_PASSWORD'] = 'qiuzcvoctvrqemgf'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config["EMAIL_HOST_PASSWORD"] = "qiuzcvoctvrqemgf"


# create Mail instance from flask mail module
mail = Mail(app)


# get most active list from API
def get_most(signal):
    foverview = Overview()
    filters_dict = {'Index': 'S&P 500', 'Sector': 'Any'}
    foverview.set_filter(signal=signal, filters_dict=filters_dict)
    try:
        df = foverview.screener_view()
        df.to_csv(signal + '.csv', columns=['Ticker'], mode='w')
    except:
        return


# get financial stock data
def get_stock_data(symbol):
    res = []
    result_list = []
    try:
        # read the  stock symbols from file a list
        with open(symbol, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            if len(data) != 0:
                for i in data[1:10]:
                    ticker = Ticker(i[1])
                    x = {
                        "symbol": i[1],
                        "close": str(round(ticker.price[i[1]]["regularMarketPrice"], 3))
                    }
                    y = (jsonpickle.encode(x))
                    res.append(y)
    except:
        return res
    return res


# get finacial data of a specific stock from the API
def get_specific_stock_data(symbol):
    res = {}
    # get the API module
    data = Ticker(symbol)
    modules = 'assetProfile earnings defaultKeyStatistics'
    info = data.get_modules(modules)
    financial = data.summary_detail
    x = {
        "symbol": symbol,
        "close": str(round(data.price[symbol]["regularMarketPrice"], 3)),
        "high": str(data.price[symbol]["regularMarketDayHigh"]),
        "volume": str(data.price[symbol]['regularMarketVolume']),
        "averageVolume": str(financial[symbol]['averageVolume']),
        "marketCap": str(data.price[symbol]['marketCap']),
        "name": str(data.price[symbol]['longName']),
        "previousClose": str(data.price[symbol]['regularMarketPreviousClose']),
        "dayLow": str(data.price[symbol]['regularMarketDayLow']),
        "info": str(info[symbol]["assetProfile"]["longBusinessSummary"]),
        "industry": str(info[symbol]["assetProfile"]["industry"]),
        "change": ' {:+.2%}'.format(data.price[symbol]["regularMarketChangePercent"]),
        "regularMarketChange": ' {:+.2f}'.format(data.price[symbol]["regularMarketChange"], 3),
        "fiftyTwoWeekLow": str(financial[symbol]['fiftyTwoWeekLow']),
        "fiftyTwoWeekHigh": str(financial[symbol]['fiftyTwoWeekHigh']),
        "recommendation": str(data.financial_data[symbol]["recommendationKey"]),
    }
    y = json.dumps(x)
    return y


# get stock data according to favorite stocks list
def favorites_data(ticker_list):
    d = {}
    all_symbols = " ".join(ticker_list)
    myInfo = Ticker(all_symbols)
    myDict = myInfo.price
    x = []
    try:
        for ticker in ticker_list:
            ticker = str(ticker)
            d.update({"currentPrice": str(myDict[ticker]["regularMarketPrice"])})
            d.update({"dayLow": str(myDict[ticker]["regularMarketDayLow"])})
            d.update({"dayHigh": str(myDict[ticker]["regularMarketDayHigh"])})
            d.update({"volume": str(myDict[ticker]["regularMarketVolume"])})
            d.update({"symbol": ticker})
            x.append(json.dumps(d))
    except:
        pass
    return json.dumps(x)


# gets current data for stocks in favorites screen
@app.route('/favoritesData', methods=['POST'])
@cross_origin()
def getFavoriteStocks():
    req = request.get_json()
    email = req['email']["otherParam"]
    if request.method == 'POST':
        for itm in db.favoriteList.find({"Email": email}):
            if (itm.get('Email') == email):
                return favorites_data(itm.get('FavoriteStocks'))


# add stock selected by user to the favoirte list in DB
@app.route('/addStocktoFavoriteList', methods=['POST'])
@cross_origin()
def addStockToFavoriteStocks():
    req = request.get_json()
    email = req['Email']["otherParam"]
    symbol = req['Symbol']
    if request.method == 'POST':
        for itm in db.favoriteList.find({"Email": email}):
            if symbol in itm['FavoriteStocks']:
                print("true")
                return jsonify({'result': "true"})
            else:
                print("false")
                db.favoriteList.update_one({'Email': email}, {'$push': {'FavoriteStocks': symbol}})
                return jsonify({'result': "false"})


# delete stock selected by user from the favoirte list in DB
@app.route('/deletStocktoFavoriteList', methods=['POST'])
@cross_origin()
def deletStockToFavoriteStocks():
    req = request.get_json()
    email = req['Email']["userParam"]
    symbol = req['Symbol']
    if request.method == 'POST':
        print("true")
        db.favoriteList.update_one({'Email': email}, {'$pull': {'FavoriteStocks': symbol}})
        return jsonify({'result': "true"})


@app.route('/fundamental', methods=['POST'])
@cross_origin()
def getData():
    req = request.get_json()
    if flask.request.method == 'POST':
        return get_specific_stock_data(req["Symbol"])


# return most active stock financial data
@app.route('/activeStockData', methods=['GET'])
@cross_origin()
def getMostActive():
    if flask.request.method == 'GET':
        return get_stock_data('Most Active.csv')


# gets sentiment score of a specific stock from top 50 stocks
@app.route('/sentimentScore', methods=['POST'])
@cross_origin()
def get_sentiment_score():
    req = request.get_json()
    if request.method == 'POST':
        return jsonify(get_sentiment_of_stock(req['symbol']))


# get results of monte carlo prediction algoirthm and send to client
@app.route('/monteCarloResults', methods=['POST'])
@cross_origin()
def getMonteCarlo():
    result = {}
    req = request.get_json()
    print(req)
    if request.method == 'POST':
        result = monte_carlo(req["Symbol"])
        print(result)
        return result


# get results of ARIMA model prediction algorithm and send to client
@app.route('/arimaResults', methods=['POST'])
@cross_origin()
def getArimaARes():
    result = {}
    req = request.get_json()
    if flask.request.method == 'POST':
        # get weekly and daily arima prediction result
        print(req)
        result["weekly"] = weekly_armia_model(req["Symbol"])
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


# this function checks if user login details are correct in MONGO DB
@app.route('/authenticate', methods=['GET', 'POST'])
@cross_origin()
def check():
    req = request.get_json()
    if request.method == 'POST':
        # check if login details are correct
        if db.users.count_documents({'Email': req["name"], 'Password': req["Password"]}, limit=1) != 0:
            return jsonify({'result': "true"})
        return jsonify({'result': "false"})



    elif request.method == 'GET':
        json_string = "{'a': 1, 'b': 2}"
        return Response(json_string, mimetype='application/json')


# this function signs up new user and updates the DB
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


# get data for stocks in a specific sector (for ex. ENERGY)
def get_sector_stocks(sector):
    sec = "sec_" + sector["name"]
    filters = ['idx_sp500', 'exch_nasd', sec, 'geo_usa']  # Shows companies in NASDAQ which are in the S&P500
    stock_list = stockScreener(filters=filters, table='Overview', order='price')  # Get the performance ta
    return json.dumps(stock_list.data)


# get data for stocks in a specific sector (for ex. ENERGY)
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
    # return combined


#sending recommendation stocks result to a chosen email
@app.route("/mail", methods=['GET', 'POST'])
def index():
    req = request.get_json()
    email = req['Email']
    recipient = [email]
    if request.method == 'POST':
        recommendation_stocks = get_protfolio_recommendation()
        msg = Message("Recommended stock list from DeltaPredict", sender=("delta predict",'irbtebh@yahoo.com'),
                      recipients=recipient)
        msg.body = str(recommendation_stocks)
        mail.send(msg)
    return "Sent"


if __name__ == "__main__":
    #app.run(debug=True)
    spList()
    def run():
        #from webapp import app
        app.run(debug=True, use_reloader=False)
    # activate FLASK server
    run()
    with app.app_context():
        get_stock_news()
        get_most('Most Active')
        get_most('Top Gainers')
        get_most('Top Losers')
    #run()
    #serve(app, host="0.0.0.0", port=5000, threads=30)
    print("!")
    #get_stock_news()
    #scrape news headlines and perform sentiment analysis
    #get_stock_news()
    # create lists of active/gainers/losers stocks


def create_app():
    return app
