
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
    total = 0
    url = "../Logic/newsHeadlines/" + symbol
    # get_sp_list()
    dataset = pd.read_csv(url)
    for i in dataset:
        scores = dataset["text"].apply(sentiment)
        # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)
    dataset = dataset.join(scores_df, rsuffix='_scores')
    dataset.to_csv(url, index=False)
    total = 0
    with open(url, 'r', newline='') as in_file:
        reader = csv.reader(in_file)
        # skip header
        next(reader)
        for row in reader:
            total += (int(row[2]))
    # dataset = dataset.join(df, rsuffix='_total')
    # dataset.to_csv("./newsHeadlines/MMM.csv", index=[0])
    field_names = ['sentiment']
    row_dict = {'sentiment': total}
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


if __name__ == "__main__":
    dict = {}
    sentiment_on_all_files()