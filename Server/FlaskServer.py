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
from pymongo import MongoClient, aggregation
import numpy as np
import jsonpickle
import pandas as pd

# Data Source
import yfinance as yf
from finvizfinance.quote import finvizfinance
from finviz.screener import Screener
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

# caution: path[0] is reserved for script path (or '' in REPL)
from Logic.SentimentAnlysis import sentiment_on_all_files
from Logic.WebCrawling import get_stock_news, get_sp_list

sys.path.insert(0, '\FinalProjectDeltaPredictBackend\Logic')
from Logic import WebCrawling

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
                for i in data[1:6]:
                    ticker = yf.Ticker(i[1])
                    # company_name = ticker.info['longName']
                    # print(company_name)
                    todays_data = ticker.history(period='1d')
                    x = {
                        "symbol": i[1],
                        "close": str(round(todays_data['Close'][0], 3)),

                    }
                    y = (jsonpickle.encode(x))
                    res.append(y)

    except:
        return res
    return res


def get_specific_stock_data(symbol):
    res = {}
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    # company_name = ticker.info['longName']

    x = {
        # "company": company_name,
        "symbol": symbol,
        "close": str(round(todays_data['Close'][0], 3)),
        "high": str(ticker.info['dayHigh']),
        "volume": str(todays_data['Volume'][0]),
        "averageVolume": str(ticker.info['averageVolume']),
        "marketCap": str(ticker.info['marketCap']),
        "name": str(ticker.info['shortName']),
        "previousClose": str(ticker.info['previousClose']),
        "dayLow": str(ticker.info['dayLow']),
        "logo": str(ticker.info['logo_url'])

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


def favorites_data(name):
    d = {}
    ticker = yf.Ticker(name)
    todays_data = ticker.history(period='1d')
    stock = finvizfinance(name)
    x = stock.ticker_fundament()
    d.update({'currentPrice': str(round(todays_data['Close'][0], 3))})
    d.update({"dayLow": str(ticker.info['dayLow'])})
    d.update({"dayHigh": str(ticker.info['dayHigh'])})
    d.update({"change": str(x['Change'])})
    d.update({'volume': str(todays_data['Volume'][0])})
    return json.dumps(d)


def get_data_for_favorites(favorites):
    x = {}
    for f in favorites:
        x.update({f: favorites_data(f)})
    return x


# gets current data for stocks in favorites screen
@app.route('/favoritesData', methods=['POST'])
@cross_origin()
def get():
    req = request.get_json()
    if flask.request.method == 'POST':
        return get_data_for_favorites(req["Symbols"])


@app.route('/fundamental', methods=['POST'])
@cross_origin()
def getData():
    req = request.get_json()
    if flask.request.method == 'POST':
        return get_data(req["Symbol"])


@app.route('/activeStockData', methods=['GET'])
@cross_origin()
def getMostActive():
    if flask.request.method == 'GET':
        return get_stock_data('Most Active.csv')


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
    d={}
    sec = "sec_" + sector["name"]
    filters = ['idx_sp500','exch_nasd', sec, 'geo_usa']  # Shows companies in NASDAQ which are in the S&P500
    stock_list = Screener(filters=filters, table='Overview', order='price')  # Get the performance ta
    # list=[]
    # for d in stock_list.data:
    #     list.append(d["Ticker"])
    # res=get_data_for_favorites(list)
    # print (res)
    # Export the screener results to .csv
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
    # app.run(debug=True)
    spList()
    # get_stock_news()
    serve(app, host="0.0.0.0", port=5000, threads=6)
    # get_most('Most Active')
    # get_most('Top Gainers')
    # get_most('Top Losers')


    # app.run(threaded=True)


def create_app():
    return app
