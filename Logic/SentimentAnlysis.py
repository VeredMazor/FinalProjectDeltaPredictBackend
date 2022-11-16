import csv

import nltk
from nltk.corpus import opinion_lexicon
import pandas as pd

# import required module
import os

# assign directory
directory = '../Logic/newsHeadlines/'
pos_list = list(opinion_lexicon.positive())
neg_list = list(opinion_lexicon.negative())
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
positive=[]
negative=[]
with open("../Logic/pos.csv", 'r', newline='') as in_file:
    reader = csv.reader(in_file)
    data = list(reader)
    positive=data+pos_list
with open("../Logic/neg.csv", 'r', newline='') as in_file:
    reader = csv.reader(in_file)
    words = list(reader)
    negative = words + neg_list



def sentiment(sentence):
    senti = 0

    words = [word.lower() for word in nltk.word_tokenize(sentence)]
    for word in words:
        if word in positive:
            senti += 1
        elif word in negative:
            senti -= 1
    return senti


def add_score_to_file(symbol):
    #need to download these for the first time
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    total = 0
    url = "../Logic/newsHeadlines/" + symbol
    df = pd.read_csv(url)
    lemma = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    def text_prep(x: str) -> list:
        corp = str(x).lower()
        corp = re.sub('[^a-zA-Z]+', ' ', corp).strip()
        tokens = word_tokenize(corp)
        words = [t for t in tokens if t not in stop_words]
        lemmatize = [lemma.lemmatize(w) for w in words]
        return lemmatize

    preprocess_tag = [text_prep(i) for i in df['text']]
    df["preprocess_txt"] = preprocess_tag
    df['total_len'] = df['preprocess_txt'].map(lambda x: len(x))

    num_pos = df['preprocess_txt'].map(lambda x: len([i for i in x if i in positive]))
    df['pos_count'] = num_pos
    num_neg = df['preprocess_txt'].map(lambda x: len([i for i in x if i in negative]))
    df['neg_count'] = num_neg
    df['sentiment'] = round((df['pos_count'] - df['neg_count']) / df['total_len'], 2)
    sum = round(df['sentiment'].sum().mean(), 2)
    # get_sp_list()
    # dataset = pd.read_csv(url)
    # for i in dataset:
    #     scores = dataset["text"].apply(sentiment)
    #     # Convert the 'scores' list of dicts into a DataFrame
    # scores_df = pd.DataFrame(scores)
    # dataset = dataset.join(scores_df, rsuffix='_scores')
    # dataset.to_csv(url, index=False)
    # total = 0
    # with open(url, 'r', newline='') as in_file:
    #     reader = csv.reader(in_file)
    #     # skip header
    #     next(reader)
    #     for row in reader:
    #         total += (int(row[2]))
    # dataset = dataset.join(df, rsuffix='_total')
    # dataset.to_csv("./newsHeadlines/MMM.csv", index=[0])
    field_names = ['sentiment']
    row_dict = {'sentiment': sum}
    with open('../Logic/newsHeadlines/' + symbol, 'a') as csv_file:
        dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
        dict_object.writerow(row_dict)


def sentiment_on_all_files():
    # iterate over files in the news directory and calculate sentiment
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # add to file sentiment of all the lines totsl
            add_score_to_file(str(f)[23:])

def example ():
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    url = "../Logic/newsHeadlines/" + "AAPL.csv"
    df = pd.read_csv(url)
    lemma = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    def text_prep(x: str) -> list:
        corp = str(x).lower()
        corp = re.sub('[^a-zA-Z]+', ' ', corp).strip()
        tokens = word_tokenize(corp)
        words = [t for t in tokens if t not in stop_words]
        lemmatize = [lemma.lemmatize(w) for w in words]
        return lemmatize

    preprocess_tag = [text_prep(i) for i in df['text']]
    df["preprocess_txt"] = preprocess_tag
    df['total_len'] = df['preprocess_txt'].map(lambda x: len(x))

    num_pos = df['preprocess_txt'].map(lambda x: len([i for i in x if i in positive]))
    df['pos_count'] = num_pos
    num_neg = df['preprocess_txt'].map(lambda x: len([i for i in x if i in negative]))
    df['neg_count'] = num_neg
    df['sentiment'] = round((df['pos_count'] - df['neg_count']) / df['total_len'], 2)
    sum= round(df['sentiment'].sum().mean(),2)

def get_sentiment_of_stock(symbol):
    url = "../Logic/newsHeadlines/" + symbol+".csv"
    with open(url, 'r') as f:
        last_line = f.readlines()[-2]
    return last_line

if __name__ == "__main__":
    dict = {}
    #sentiment_on_all_files()
    #example()
    print(get_sentiment_of_stock("AAPL"))

