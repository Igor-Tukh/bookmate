import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
import json

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


def plot_stats(book_id, stats):
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


def merge_speed_to_pages(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    borders = db['%s_borders' % book_id].find({'avr_abs_speed': {'$exists': True}})
    pages = db['%s_pages' % book_id].find()

    for (page, border) in zip(pages, borders):
        db['%s_pages' % book_id].update({'_id': page['_id']},
                              {'$set': {
                                  'avr_abs_speed': border['avr_abs_speed'],
                                  # 'users_in_page': border['users_in_border'],
                                  # 'users_part_in_page': border['users_part_in_border'],
                                  # 'skip_part': border['skip_part']
                              }})
    return


def tsne_to_pages(book_id):
    db = connect_to_mongo_database(BOOKS_DB)
    pages = db['%s_pages' % book_id].find()
    with open('tsne/%s.txt' % book_id) as tsne_file:
        tsne = json.load(tsne_file)

    projections = tsne[0]['projections']
    for (projection, page) in zip(projections, pages):
        db['%s_pages' % book_id].update({'_id': page['_id']},
                                        {'$set': {'tsne_x': projection['tsne-0'],
                                                  'tsne_y': projection['tsne-1']}})

book_ids = ['2206']
for book_id in book_ids:
    merge_speed_to_pages(book_id)
    # tsne_to_pages(book_id)
    # plot_stats(book_id, ['avr_abs_speed', 'person_verbs_part', 'labeled_word_part', 'dialogs_part',
    #                                     'sentiment_words_portion'])
    # plot_stats(book_id, ['avr_word_len'])
    get_correlation(book_id, ['avr_word_len',
                              'person_verbs_part',
                              'labeled_word_part',
                              'dialogs_part',
                              'sentiment_words_portion',
                              'words_num',
                              'sentences_num',
                              'p_num',
                              'new_words_count'])