# import required modules
import nltk
from nltk.corpus import opinion_lexicon
import pandas as pd
import csv
import os
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# assign directory
directory = '../Logic/newsHeadlines/'
# create pos and neg list from nltks opinion lexicon
pos_list = list(opinion_lexicon.positive())
neg_list = list(opinion_lexicon.negative())
positive = []
negative = []

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
    url = "../Logic/newsHeadlines/" + symbol
    df = pd.read_csv(url)
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

    # preproccess each word
    preprocess_tag = [text_prep(i) for i in df['text']]
    df["preprocess_txt"] = preprocess_tag
    df['total_len'] = df['preprocess_txt'].map(lambda x: len(x))

    #calculate number of positive and negative words according to the lists we created
    num_pos = df['preprocess_txt'].map(lambda x: len([i for i in x if i in positive]))
    df['pos_count'] = num_pos
    num_neg = df['preprocess_txt'].map(lambda x: len([i for i in x if i in negative]))
    df['neg_count'] = num_neg
    field_names = ['sentiment_score']
    #compute final score based on mean of (pos-negative)/total
    try:
        df['sentiment_score'] = round((df['pos_count'] - df['neg_count']) / df['total_len'], 2)
        sentiment_score = round(df['sentiment_score'].sum().mean(), 2)
        row_dict = {'sentiment_score': sentiment_score}
    except:
        row_dict = {'sentiment_score': 0}
    #write the score into a file
    with open('../Logic/newsHeadlines/' + symbol, 'a') as csv_file:
        dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
        dict_object.writerow(row_dict)

#perform sentiment calculation on all files from the top 50 s&p list
def sentiment_on_all_files():
    # iterate over files in the news directory and calculate sentiment
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # add to file sentiment of all the lines totsl
            add_score_to_file(str(f)[23:])

#get stocks monthly sentiment from the end of the file
def get_sentiment_of_stock(symbol):
    url = "../Logic/newsHeadlines/" + symbol + ".csv"
    with open(url, 'r') as f:
        last_line = f.readlines()[-2]
    return last_line



if __name__ == "__main__":
    dict = {}