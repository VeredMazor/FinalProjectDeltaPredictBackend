# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from flask import Flask, render_template
from flask import request
from waitress import serve
#from flask_cors import CORS

app = Flask (__name__)
app._static_folder = ''

#CORS(app)

@app.route("/")
@app.route('/data')
def get_time():
    # Returning an api for showing in  reactjs
    return {
        'Name': "geek",
        "Age": "22",
        "Date": x,
        "programming": "python"
    }

import datetime

x = datetime.datetime.now()


if __name__ == "__main__":
   # app.run(debug=True)
    serve(app, host="0.0.0.0", port=8080)




def create_app():
        return app
