# import required modules
import nltk
from nltk.corpus import opinion_lexicon
import pandas as pd
import csv
import os
from pymongo import MongoClient, aggregation
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import itertools

# assign directory
directory = '../Logic/newsHeadlines/'
# create pos and neg list from nltks opinion lexicon
pos_list = list(opinion_lexicon.positive())
neg_list = list(opinion_lexicon.negative())
positive = []
negative = []
symbols=["AAPL","MSFT","AMZN","TSLA","UNH","GOOGL","XOM","JNJ","GOOG","JPM","NVDA","CVX",'V',"PG","HD","LLY","MA","PFE","ABBV","BAC","MRK","PEP","KO","COST","META",
"MCD","WMT","TMO","CSCO","DIS","AVGO","WFC","COP","ABT","BMY","ACN","DHR","VZ","NEE","LIN","CRM","TXN","AMGN","RTX","HON","PM","ADBE","CMCSA"]
# create mongoDB refernce and start flask app
cluster = MongoClient(
    "mongodb+srv://DeltaPredict:y8RD27dwwmBnUEU@cluster0.7yz0lgf.mongodb.net/?retryWrites=true&w=majority")
# create DB cluster reference
db = cluster["DeltaPredictDB"]

# combine the lists we created with the LoughranMcDonald Master Dictionary
with open("../Logic/pos.csv", 'r', newline='') as in_file:
    reader = csv.reader(in_file)
    data = list(reader)
    positive = data + pos_list
with open("../Logic/neg.csv", 'r', newline='') as in_file:
    reader = csv.reader(in_file)
    words = list(reader)
    negative = words + neg_list


# calculate monthly sentiment score according to all news headlines in a file and append the result mean  itno the last row of the file
def add_score_to_file(symbol):
    #need to download these for the first time
    #nltk.download('stopwords')
    #nltk.download('wordnet')
    #nltk.download('omw-1.4')
    # nltk.download('punkt')
    total = 0
    # Creating Empty DataFrame and Storing it in variable df
    df = []
    # url = "../Logic/newsHeadlines/" + symbol
    # df = pd.read_csv(url)
    for itm in db.newsHeadlines.find({"ticker": symbol}):
        if itm.get('ticker') == symbol:
            df.append(itm)
    #df=pd.DataFrame(df)
    lemmatizer = WordNetLemmatizer()
    # get a list of stop words in english
    stop_words = stopwords.words('english')

    # perform text preprocessing
    def text_prep(s: str) -> list:
        corpus = str(s).lower()
        corpus = re.sub('[^a-zA-Z]+', ' ', corpus).strip()
        token_list = word_tokenize(corpus)
        # remove stop words and reduce forms of a word to a common base form
        word_list = [t for t in token_list if t not in stop_words]
        lemmatize = [lemmatizer.lemmatize(w) for w in word_list]
        return lemmatize
    res=[]
    #res=list(itertools.chain.from_iterable(res))

    # preproccess each word
    preprocess_tag = [text_prep(i["text"]) for i in df]
    preprocess_tag=list(itertools.chain.from_iterable(preprocess_tag))
    df={}
    df["preprocess_txt"] = preprocess_tag
    df['total_len'] = len(df['preprocess_txt'])

    # calculate number of positive and negative words according to the lists we created
    num_pos = len([i for i in  df["preprocess_txt"]  if i in positive])
    df['pos_count'] = num_pos
    num_neg = len([i for i in  df["preprocess_txt"]  if i in negative])
    df['neg_count'] = num_neg
    field_names = ['sentiment_score']
    # compute final score based on mean of (pos-negative)/total
    try:
        df['sentiment_score'] = (df['pos_count']/ (df['neg_count']+1))
        #sentiment_score = round(df['sentiment_score'].sum().mean(), 2)
        row_dict = {"symbol":symbol,'sentiment_score': round(df['sentiment_score'],2) }
        db.sentimentScores.insert_one(row_dict)
    except:
        row_dict = {'sentiment_score': 0}

    # with open('../Logic/newsHeadlines/' + symbol, 'a') as csv_file:
    #     dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
    #     dict_object.writerow(row_dict)

#perform sentiment calculation on all files from the top 50 s&p list
def sentiment_on_all_files():
    # iterate over files in the news directory and calculate sentiment
    for s in symbols:
        add_score_to_file(s)

#get stocks monthly sentiment from the end of the file
def get_sentiment_of_stock(ticker):
    # url = "../Logic/newsHeadlines/" + symbol + ".csv"
    # with open(url, 'r') as f:
    #     last_line = f.readlines()[-2]
    # return last_line
    for itm in db.sentimentScores.find({"symbol": ticker}):
        if itm.get('symbol') == ticker:
            return itm["sentiment_score"]



if __name__ == "__main__":
    dict = {}
    print(get_sentiment_of_stock("AAPL"))