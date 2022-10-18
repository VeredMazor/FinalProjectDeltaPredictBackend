import data as data
import pymongo
from flask import Flask, render_template, jsonify, url_for, redirect, Response
from flask import request, make_response
from flask_cors import CORS, cross_origin
import datetime
import jsonpickle
import time
import csv
import flask
from pymongo import MongoClient, aggregation
import numpy as np
# Data Source
import yfinance as yf
import pandas as pd
import bs4
from bs4 import BeautifulSoup
import requests
from finvizfinance.quote import finvizfinance
import json
from finvizfinance.screener.overview import Overview
from waitress import serve

# get db
from waitress import serve

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


def get_sp_list():
    # Scrape the entire S&P500 list from Wikipedia into a Pandas DataFrame;
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df.to_csv('S&P500-Info.csv')
    df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])


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
    }
    y = jsonpickle.encode(x)
   # res.append(y)
    return y


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
            if(itm.get('Email') == req['otherParam']):
                print(itm.get('FavoriteStocks'))
                return jsonify({'result': itm.get('FavoriteStocks')}) #TO DO return the stocks list favorit to user

        print(req['otherParam'])
        insert = {'Email': req['otherParam'], 'FavoriteStocks': []}
        db.favoriteList.insert_one(insert)
        return ("itm.get('_id')")







if __name__ == "__main__":
    app.run(debug=True)
    #serve(app, host="0.0.0.0", port=5000, threads=7)
    get_most('Most Active')
    get_most('Top Gainers')
    get_most('Top Losers')

    # app.run(threaded=True)



def create_app():
    return app
