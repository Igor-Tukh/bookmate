from src.doc2vec_module.src.build_features import *
from enum import Enum
from sklearn.linear_model import Ridge, LinearRegression, Lasso, SGDRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, \
    median_absolute_error, r2_score

import os
import numpy as np
import csv

METRICS = [('explained variance', explained_variance_score), ('mean squared error', mean_squared_error),
           ('r2', r2_score)]


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


def preapare_features(split_type, book_id, user_id, features_names, result_name,
                      features_proportion=1.0, split_proportion=0.8):
    features, handler = upload_features(book_id, user_id)
    train, test = split_features(features, split_type, features_proportion, split_proportion)

    train_X = np.array([[features_dict[key]
                         for key in features_names] for features_dict in train])

    test_X = np.array([[features_dict[key]
                        for key in features_names] for features_dict in test])

    train_y = np.array([np.log(features_dict[result_name]) for features_dict in train])
    test_y = np.array([np.log(features_dict[result_name]) for features_dict in test])

    return train_X, test_X, train_y, test_y


if __name__ == '__main__':
    book_id, document_id = BOOK_IDS[0]
    results = []

    for regression_type in [Ridge(alpha=0.5), Lasso(alpha=0.1), LinearRegression(), SGDRegressor(max_iter=1000)]:
        for user_id in USER_IDS[document_id]:
            for split_type in [Split.RANDOM, Split.ORDER]:
                current_results = {}
                train_X, test_X, train_y, test_y = preapare_features(split_type,
                                                                     book_id,
                                                                     user_id,
                                                                     {'words_number', 'sentences_number',
                                                                      'average_word_len', 'average_sentence_len',
                                                                      'hour', 'distance_from_the_beginning',
                                                                      'rare_words_count', 'is_weekend',
                                                                      'verbs_count', 'noun_count'},
                                                                     'speed')
                reg = regression_type
                reg.fit(train_X, train_y)
                pred_y = reg.predict(test_X)
                qulity_y = reg.predict(train_X)

                current_results['book_id'] = book_id
                current_results['user_id'] = user_id
                current_results['split_type'] = split_type

                for name, metric in METRICS:
                    current_results[name] = metric(test_y, pred_y)
                results.append(current_results)

    with open(os.path.join('..', 'results', 'results.csv'), 'w') as results_file:
        writer = csv.DictWriter(results_file,
                                fieldnames=['book_id', 'user_id', 'split_type'] + [name for name, _ in METRICS])
        writer.writeheader()

        for session_features in results:
            writer.writerow(session_features)
