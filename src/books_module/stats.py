import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np

BOOKS_DB = 'bookmate'


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def smooth_points(Y, N=10):
    new_Y = []
    for i in range(0, len(Y)):
        smooth_N = N
        if i - N < 0:
            smooth_N = i
            new_Y.append(Y[i])
            continue
        elif i + N >= len(Y):
            smooth_N = len(Y) - i - 1
            new_Y.append(Y[i])
            continue

        sum = 0
        for j in range(-smooth_N, smooth_N):
            sum += Y[i + j]
        sum /= ((2 * smooth_N) + 1)
        new_Y.append(sum)

    return new_Y


def plot_abs_speed_plus_stats(book_id, stats):
    db = connect_to_mongo_database(BOOKS_DB)
    borders = db['%s_borders' % book_id].find({'avr_abs_speed': {'$exists': True}})

    x, y = list(range(0, borders.count())), list()
    for border in borders:
        y.append(border['avr_abs_speed'])

    # plot speed first
    y = smooth_points(y, N=5)
    plt.clf()
    plt.plot(x, y)

    # plot all needed params
    for param in stats:
        param_values = list()
        pages = db['%s_pages' % book_id].find()
        for page in pages:
            param_values.append(page[param])
        param_values = smooth_points(param_values, 5)
        plt.plot(x, param_values)

    plt.xlabel('Pages')
    plt.ylabel('Absolute speed')
    plt.legend()
    plt.tight_layout()

    plt.savefig('images/%s_speed_stats.png' % book_id)


def plot_stats(stats):
    db = connect_to_mongo_database(BOOKS_DB)
    pages = db['%s_pages' % book_id].find()

    x, y = list(range(0, pages.count())), list()

    for param in stats:
        param_values = list()
        pages = db['%s_pages' % book_id].find()
        for page in pages:
            param_values.append(page[param])
        param_values = smooth_points(param_values, 5)
        plt.plot(x, param_values)

    plt.xlabel('Pages')
    plt.ylabel('Absolute speed')
    plt.legend()
    plt.tight_layout()

    plt.savefig('images/%s_stats.png' % book_id)


def get_correlation(book_id, stats):
    file = open('logs/%s_corr.txt' % book_id, 'w')
    db = connect_to_mongo_database(BOOKS_DB)
    borders = db['%s_borders' % book_id].find({'avr_abs_speed': {'$exists': True}})

    speed = list()
    for border in borders:
        speed.append(border['avr_abs_speed'])

    for stat in stats:
        pages = db['%s_pages' % book_id].find()
        values = list()
        for page in pages:
            values.append(page[stat])
        correlation = np.corrcoef(speed, values)[0][1]
        file.write('[speed] and [%s]: %.3f \n' % (stat, correlation))

book_ids = ['2289']
for book_id in book_ids:
    # plot_abs_speed_plus_stats(book_id, ['person_verbs_part', 'labeled_word_part', 'dialogs_part',
    #                                     'sentiment_word_part'])
    # plot_stats(['avr_word_len'])
    get_correlation(book_id, ['avr_word_len',
                              'person_verbs_part',
                              'labeled_word_part',
                              'dialogs_part',
                              'sentiment_word_part',
                              'words_num',
                              'sentences_num',
                              'p_num',
                              'new_words_count'])