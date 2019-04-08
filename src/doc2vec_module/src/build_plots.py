from pymongo import MongoClient
from src.doc2vec_module.src.build_features import *

import matplotlib.pyplot as plt
import os
import csv


def connect_to_mongo_database(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db


def build_all_speed_distribution(book_id):
    db = connect_to_mongo_database('bookmate')
    speeds = db[book_id].find().distinct('speed')
    current_symbols_number = len(get_epub_book_text_with_ebook_convert(book_id))
    real_speeds = [(speed / 3568733) * current_symbols_number for speed in speeds]
    plt.clf()
    plt.hist(real_speeds, range=(0, 10000), bins=2000)
    plt.title("Speed distribution")
    plt.xlabel('Speed sym/min')
    plt.ylabel('Number of sessions')
    plt.savefig('../plots/article/%s_speed_distr.png' % book_id)


def build_user_speed_plots(book_id, user_id):
    db = connect_to_mongo_database('bookmate')
    sessions = db[book_id].find({"$or": [{'user_id': user_id}, {'user_id': int(user_id)}]})
    sessions = list(sessions)
    sessions.sort(key=lambda session: session['read_at'])
    current_symbols_number = len(get_epub_book_text_with_ebook_convert(book_id))

    plt.clf()
    plt.xlabel('Session sequence number')
    plt.ylabel('Position in the book')
    plt.plot(range(len(sessions)), [session['book_from'] for session in sessions])
    plt.savefig(os.path.join('..', 'plots', 'article', str(user_id) + '_percent.png'))

    plt.clf()
    plt.xlabel('Session serial (chronological) number')
    plt.ylabel('Speed sym/min')
    plt.plot(range(len(sessions)), [(session['speed'] / 3568733) * current_symbols_number for session in sessions])
    plt.savefig(os.path.join('..', 'plots', 'article', str(user_id) + '_speed.png'))


def build_several_thresholds(results_file, user_id, output_dir_path, user_ind):
    max_speeds = set()
    min_speeds = set()
    with open(results_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = {}

        for row in reader:
            if str(row['user_id']) != user_id:
                continue

            current = {'max_speed': float(row['max_speed']),
                       'min_speed': float(row['min_speed']),
                       'user_id': str(row['user_id']),
                       'mse': float(row['mean squared error'])}
            data[(current['min_speed'], current['max_speed'])] = current
            max_speeds.add(current['max_speed'])
            min_speeds.add(current['min_speed'])

    plt.clf()
    plt.xlabel('Min speed threshold')
    plt.ylabel('MSE')
    plt.title('User {user_id} predictions MSE'.format(user_id=user_ind))

    max_speeds = list(max_speeds)
    min_speeds = list(min_speeds)
    max_speeds.sort()
    min_speeds.sort()

    for max_speed in max_speeds:
        mses = [data[(min_speed, max_speed)]['mse'] for min_speed in min_speeds]
        plt.plot(min_speeds, mses, label='Max speed = {ms}'.format(ms=max_speed))

    plt.legend()
    plt.savefig(os.path.join(output_dir_path, str(user_id) + '.png'))


def build_plot_several_thresholds_for_users_min(results_file, user_ids, output_dir_path):
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.xlabel('Min speed threshold')
    plt.ylabel('MAE')
    plt.yscale('linear')

    max_speed = 0
    for user_ind, user_id in enumerate(user_ids):
        min_speeds = set()
        with open(results_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = {}

            for row in reader:
                if str(row['user_id']) != user_id:
                    continue

                current = {'max_speed': float(row['max_speed']),
                           'min_speed': float(row['min_speed']),
                           'user_id': str(row['user_id']),
                           'mae': float(row['mean absolute error'])}
                data[(current['min_speed'], current['max_speed'])] = current
                max_speed = current['max_speed']
                min_speeds.add(current['min_speed'])

        min_speeds = list(min_speeds)
        min_speeds.sort()
        mses = [data[(min_speed, max_speed)]['mae'] for min_speed in min_speeds]
        ax.plot(min_speeds, mses, label='User {user_id}'.format(user_id=user_ind + 1))
        ax.text(min_speeds[-1] + 25 * (user_ind / len(user_ids)),
                mses[-1],
                str(user_ind + 1),
                fontsize=8,
                horizontalalignment='center',
                verticalalignment='center')
        ax.text(min_speeds[0] - 25 * (user_ind / len(user_ids)),
                mses[0],
                str(user_ind + 1),
                fontsize=8,
                horizontalalignment='center',
                verticalalignment='center')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Users predictions MAE, max speed = {ms}'.format(ms=max_speed))
    plt.savefig(os.path.join(output_dir_path, 'results.png'), bbox_inches='tight')


def build_plot_several_thresholds_for_users_max(results_file, user_ids, output_dir_path):
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.xlabel('Max speed threshold')
    plt.ylabel('MAE')
    plt.yscale('linear')

    min_speed = 0
    for user_ind, user_id in enumerate(user_ids):
        max_speeds = set()
        with open(results_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = {}

            for row in reader:
                if str(row['user_id']) != user_id:
                    continue

                current = {'max_speed': float(row['max_speed']),
                           'min_speed': float(row['min_speed']),
                           'user_id': str(row['user_id']),
                           'mae': float(row['mean absolute error'])}
                data[(current['min_speed'], current['max_speed'])] = current
                min_speed = current['min_speed']
                max_speeds.add(current['max_speed'])

        max_speeds = list(max_speeds)
        max_speeds.sort()
        mses = [data[(min_speed, max_speed)]['mae'] for max_speed in max_speeds]
        ax.plot(max_speeds, mses, label='User {user_id}'.format(user_id=user_ind + 1))
        ax.text(max_speeds[-1] + 25 * (user_ind / len(user_ids)),
                mses[-1],
                str(user_ind + 1),
                fontsize=8,
                horizontalalignment='center',
                verticalalignment='center')
        ax.text(max_speeds[0] - 25 * (user_ind / len(user_ids)),
                mses[0],
                str(user_ind + 1),
                fontsize=8,
                horizontalalignment='center',
                verticalalignment='center')

    plt.title('Users predictions MAE, min speed = {ms}'.format(ms=min_speed))
    plt.savefig(os.path.join(output_dir_path, 'results.png'), bbox_inches='tight')


if __name__ == '__main__':
    for user_id in USER_IDS['1222472']:
        build_user_speed_plots(BOOK_IDS[0][0], user_id)

    # build_all_speed_distribution(BOOK_IDS[0][0])
    # build_plot_several_thresholds_for_users_max(os.path.join('..', 'results', 'results.csv'),
    #                                             USER_IDS['1222472'],
    #                                             os.path.join('..', 'plots', 'article', 'thresholds'))
    # for ind, user_id in enumerate(USER_IDS['1222472']):
    #     build_several_thresholds(os.path.join('..', 'results', 'results.csv'), user_id,
    #                              os.path.join('..', 'plots', 'article', 'thresholds'), ind + 1)
