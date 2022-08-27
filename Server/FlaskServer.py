# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from flask import Flask, render_template, url_for  # impor inctens of flask
from flask import request
from waitress import serve
from flask_cors import CORS, cross_origin
from flask import url_for
import datetime


x = datetime.datetime.now()

app = Flask (__name__) #becuase the instanc we can call flask and put the name val.
app._static_folder = ''

CORS(app)

@app.route("/") #Decorator in python, its basically saying that what url in your website i am going to navigate through and display you some html code.

def get_time():
    # Returning an api for showing in  reactjs
    return {
        'Name': "geek",
        "Age": "22",
        "Date": x,
        "programming": "python"
    }

@app.route('/login')
def login():
    return 'fgklfdmgkldm'

@app.route('/user/<username>')
def profile(username):
    return f'{username}\'s profile'


if __name__ == "__main__":
   app.run(debug=True)
    #serve(app, host="0.0.0.0", port=8080)




def create_app():
        return app
