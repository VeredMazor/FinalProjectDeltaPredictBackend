from flask import Flask, render_template, jsonify, url_for, redirect, Response
from flask import request, make_response
from flask_cors import CORS, cross_origin
import datetime
from pymongo import MongoClient

x = datetime.datetime.now()
cluster = MongoClient(
    "mongodb+srv://DeltaPredict:y8RD27dwwmBnUEU@cluster0.7yz0lgf.mongodb.net/?retryWrites=true&w=majority")
app = Flask(__name__)
app._static_folder = ''
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)
db = cluster["DeltaPredictDB"]


# cors = CORS(app, resources={r"/api/*": {"origins": "http://localhost:19006/"}})

@app.route("/")
def get_time():
    # Returning an api for showing in  reactjs
    return {
        'Name': "geek",
        "Age": "22",
        "Date": x,
        "programming": "python"
    }

@app.route('/home', methods=['GET', 'POST'])

@cross_origin()
def login():
    req = request.get_json()
    if request.method == 'POST':

        print(req["name"])
        return jsonify({'name': req["name"]})

    elif request.method == 'GET':
        json_string = "{'a': 1, 'b': 2}"
        return Response(json_string, mimetype='application/json')


@app.route('/authenticate', methods=['GET', 'POST'])
@cross_origin()
def check():
    req = request.get_json()
    if request.method == 'POST':
        # check if login details are correct
        if db.users.count_documents({'userName': req["name"], 'Password': req["Password"]}, limit=1) != 0:
            return jsonify({'result': "true"})
        return jsonify({'result': "false"})
        # insert to DB
        # insert = {'userName': req["name"], 'Password': req["Password"]}
        # db.users.insert_one(insert)


    elif request.method == 'GET':
        json_string = "{'a': 1, 'b': 2}"
        return Response(json_string, mimetype='application/json')



if __name__ == "__main__":
    app.run(debug=True)
    # serve(app, host="0.0.0.0", port=8080)


