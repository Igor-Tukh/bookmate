from src.doc2vec_module.src.build_features import *
from enum import Enum
from sklearn.linear_model import Ridge, LinearRegression, Lasso, SGDRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, \
    median_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import os
import numpy as np
import csv

METRICS = [('explained variance', explained_variance_score), ('mean squared error', mean_squared_error),
           ('r2', r2_score)]
EPS = 1e-9


class Split(Enum):
    ORDER = 1
    RANDOM = 2


def upload_features(book_id, user_id):
    sessions = get_user_sessions(book_id, user_id)
    handler = FeaturesHandler(sessions, book_id)
    handler.upload_features(os.path.join('..', 'features', str(user_id) + '.csv'))
    features = handler.get_features()
    features.sort(key=lambda features_dict: features_dict['read_at'])
    return features, handler


def split_features(features, split_type, total_proportion=1.0, split_proportion=0.8):
    total_features_cnt = int(len(features) * total_proportion)
    split_len = int(total_features_cnt * split_proportion)
    features = features[:total_features_cnt]

    if split_type is Split.ORDER:
        return features[:split_len], features[split_len:]
    elif split_type is Split.RANDOM:
        permutation = np.random.permutation(total_features_cnt)
        left = []
        right = []

        for ind in permutation:
            if ind < split_len:
                left.append(features[ind])
            else:
                right.append(features[ind])

        return left, right

    return [], []


def preapare_features(split_type,
                      book_id,
                      user_id,
                      features_names,
                      result_name,
                      features_proportion=1.0,
                      split_proportion=0.8,
                      log=True,
                      min_speed=200.0,
                      max_speed=8000.0,
                      min_words=15):
    features, handler = upload_features(book_id, user_id)
    features = [features_dict for features_dict in features
                if min_speed + EPS <= features_dict['speed'] <= max_speed - EPS and
                features_dict['words_number'] >= min_words]
    train, test = split_features(features, split_type, features_proportion, split_proportion)

    train_X = np.array([[features_dict[key]
                         for key in features_names] for features_dict in train])

    test_X = np.array([[features_dict[key]
                        for key in features_names] for features_dict in test])

    f = np.log if log else lambda x: x
    train_y = np.array([f(features_dict[result_name]) for features_dict in train])
    test_y = np.array([f(features_dict[result_name]) for features_dict in test])

    return train_X, test_X, train_y, test_y


def visualize_sessions(user_id, sessions):
    sessions.sort(key=lambda session: session['read_at'])
    sessions = sessions

    max_speed = -1e18
    for session in sessions:
        max_speed = max(max_speed, session['speed'])

    plt.clf()
    plt.plot(range(len(sessions)), [session['book_from'] for session in sessions])
    plt.savefig(os.path.join('..', 'plots', str(user_id) + '_percent.png'))

    plt.clf()
    plt.plot(range(len(sessions)), [session['speed'] / max_speed * 100 for session in sessions], 'g')
    plt.savefig(os.path.join('..', 'plots', str(user_id) + '_speed.png'))


if __name__ == '__main__':
    # for user_id in USER_IDS[BOOK_IDS[0][1]]:
    #     preapare_features(Split.ORDER, book_id, user_id, {'words_number', 'sentences_number',
    #                                                       'average_word_len', 'average_sentence_len',
    #                                                       'hour', 'distance_from_the_beginning',
    #                                                       'rare_words_count', 'is_weekend',
    #                                                       'verbs_count', 'noun_count'}, 'speed',
    #                       max_speed=8000,
    #                       min_speed=200)
    #     sessions = get_user_sessions(BOOK_IDS[0][0], user_id)
    #     visualize_sessions(user_id, list(sessions))
    #
    # exit(0)

    book_id, document_id = BOOK_IDS[0]
    results = []

    for min_speed in [100.0, 200.0, 400.0, 800.0, 1000.0, 2000.0]:
        for max_speed in [3000.0, 4000.0, 6000.0, 8000.0]:
            for regression_type in [(Ridge(alpha=0.5), 'ridge'), (Lasso(alpha=0.1), 'lasso'),
                                    (LinearRegression(), 'linear regression'),
                                    (SGDRegressor(max_iter=1000), 'sgd regressor')]:
                for user_id in USER_IDS[document_id]:
                    for split_type in [Split.RANDOM, Split.ORDER]:
                        current_results = {}
                        features_names = ['words_number', 'sentences_number',
                                          'average_word_len', 'average_sentence_len',
                                          'hour', 'distance_from_the_beginning',
                                          'rare_words_count', 'is_weekend',
                                          'verbs_count', 'noun_count']

                        train_X, test_X, train_y, test_y = preapare_features(split_type,
                                                                             book_id,
                                                                             user_id,
                                                                             set(features_names),
                                                                             'speed',
                                                                             min_speed=min_speed,
                                                                             max_speed=max_speed)
                        scaler = MinMaxScaler()
                        scaler.fit_transform(train_X)
                        test_X = scaler.transform(test_X)

                        reg, model_name = regression_type
                        reg.fit(train_X, train_y)
                        pred_y = reg.predict(test_X)
                        qulity_y = reg.predict(train_X)

                        current_results['book_id'] = book_id
                        current_results['user_id'] = user_id
                        current_results['split_type'] = split_type
                        current_results['features'] = ', '.join(features_names)
                        current_results['model'] = model_name
                        current_results['min_speed'] = min_speed
                        current_results['max_speed'] = max_speed

                        for name, metric in METRICS:
                            current_results[name] = metric(test_y, pred_y)
                        results.append(current_results)

    with open(os.path.join('..', 'results', 'results.csv'), 'w') as results_file:
        writer = csv.DictWriter(results_file,
                                fieldnames=['book_id', 'user_id', 'split_type', 'features', 'model',
                                            'min_speed', 'max_speed'] + [name for name, _ in METRICS])
        writer.writeheader()

        for session_features in results:
            writer.writerow(session_features)
