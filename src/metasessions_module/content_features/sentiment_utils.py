import os
import csv
import sys

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

from src.metasessions_module.utils import save_via_pickle, load_from_pickle


def load_words_sentiment_dictionary_csv(dictionary_path=None):
    if dictionary_path is None:
        dictionary_path = os.path.join('resources', 'sentiment', 'words_all_full_rating_utf_8.csv')

    words_sentiment = {}
    with open(dictionary_path) as dict_file:
        sentiment_reader = csv.DictReader(dict_file, delimiter=';', quotechar='"')
        for sentiment_row in sentiment_reader:
            word = sentiment_row['Words']
            words_sentiment[word] = float(sentiment_row['average rate'])

    return words_sentiment


def words_sentiment_dict_path():
    return os.path.join('resources', 'sentiment', 'words_sentiment.pkl')


def load_words_sentiment_dict(dict_path=None):
    if dict_path is None:
        dict_path = words_sentiment_dict_path()
    return load_from_pickle(dict_path)


def get_word_sentiment(sentiment_dict, word):
    return 0.0 if word not in sentiment_dict else sentiment_dict[word]


if __name__ == '__main__':
    words_sentiment_dictionary = load_words_sentiment_dictionary_csv()
    save_via_pickle(words_sentiment_dictionary, words_sentiment_dict_path())
