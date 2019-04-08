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

# METRICS = [('explained variance', explained_variance_score), ('mean squared error', mean_squared_error),
#            ('r2', r2_score)]

METRICS = [('mean squared error', mean_squared_error), ('mean absolute error', mean_absolute_error)]

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

        left = [features[number] for number in permutation[:split_len]]
        right = [features[number] for number in permutation[split_len:]]

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
    word_lens = [features_dict['words_number']for features_dict in features]
    print(len(word_lens), '+', end=' ')
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
    data = [(session['read_at'], session['book_from'], session['book_to']) for session in sessions[:100]]
    for data_item in data:
        print(data_item[0], data_item[1], data_item[2])
    print()

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
    book_id, document_id = BOOK_IDS[0]
    results = []

    for user_id in USER_IDS[document_id]:
        features, handler = upload_features(book_id, user_id)
        print(user_id, len([features_dict for features_dict in features if
                            800 + EPS <= features_dict['speed'] <= 3000 - EPS]))
    exit(0)

    # for regression_type in [(Ridge(alpha=0.5), 'ridge'), (Lasso(alpha=0.1), 'lasso'),
    #                         (LinearRegression(), 'linear regression'),
    #                         (SGDRegressor(max_iter=1000), 'sgd regressor')]:
    with open(os.path.join('..', 'results', 'results.csv'), 'w') as results_file:
        writer = csv.DictWriter(results_file,
                                fieldnames=['book_id', 'user_id', 'split_type', 'features', 'model',
                                            'min_speed', 'max_speed'] + [name for name, _ in METRICS])
        writer.writeheader()

        text_features_names = ['words_number', 'sentences_number', 'average_word_len', 'average_sentence_len',
                               'rare_words_count', 'verbs_count', 'noun_count']
        context_features_names = ['hour', 'distance_from_the_beginning', 'is_weekend']

        combined_features_names = text_features_names + context_features_names

        for user_id in USER_IDS[document_id]:
            train_X, test_X, train_y, test_y = preapare_features(Split.ORDER,
                                                                 book_id,
                                                                 user_id,
                                                                 set(combined_features_names),
                                                                 'speed',
                                                                 min_speed=800,
                                                                 max_speed=3000,
                                                                 log=False)

        exit(0)

        for features_names, features_set_type in [(combined_features_names, 'combined')]:
            for user_id in USER_IDS[document_id]:
                for split_type in [Split.ORDER]:
                    for min_speed in [800.0]:
                        for max_speed in [2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0]:
                            current_results = {}

                            train_X, test_X, train_y, test_y = preapare_features(split_type,
                                                                                 book_id,
                                                                                 user_id,
                                                                                 set(features_names),
                                                                                 'speed',
                                                                                 min_speed=min_speed,
                                                                                 max_speed=max_speed,
                                                                                 log=False)
                            scaler = MinMaxScaler()
                            scaler.fit_transform(train_X)
                            test_X = scaler.transform(test_X)

                            reg, model_name = Lasso(alpha=0.1), 'lasso'
                            reg.fit(train_X, train_y)
                            pred_y = reg.predict(test_X)

                            current_results['book_id'] = book_id
                            current_results['user_id'] = user_id
                            current_results['split_type'] = split_type
                            current_results['features'] = features_set_type
                            current_results['model'] = model_name
                            current_results['min_speed'] = min_speed
                            current_results['max_speed'] = max_speed

                            for name, metric in METRICS:
                                current_results[name] = metric(test_y, pred_y)
                            results.append(current_results)

                            writer.writerow(current_results)
                            print('One done')
