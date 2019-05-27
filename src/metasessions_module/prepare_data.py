import argparse
import logging
import sys
import os
import csv
import pickle
import numpy as np

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

from src.metasessions_module.text_features_handler import TextFeaturesHandler
from src.metasessions_module.users_clustering import load_clusters
from src.metasessions_module.utils import save_via_pickle, load_from_pickle
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler(os.path.join('logs', 'swipes_prediction.log'), 'a')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
log_step = 100000


def identify_swipe_places(book_id, model_name, target_cluster_index,
                          swipe_percentile=55,
                          swipe_ratio=0.5):
    clusters = load_clusters(book_id, model_name)
    target_cluster = clusters[target_cluster_index][0]
    swipe_speeds = []
    for speeds in target_cluster:
        swipe_speeds.append(np.percentile(speeds, swipe_percentile))
    result = np.zeros(target_cluster.shape[1])
    for position in range(result.shape[0]):
        swipes_count = np.sum(np.array(target_cluster[:, position] > swipe_speeds, dtype=np.int64))
        if swipes_count >= swipe_ratio * target_cluster.shape[0]:
            result[position] = 1
    return result


def save_features(X, book_id, batches_amount):
    with open(os.path.join('resources',
                           'features',
                           '{}_{}.pkl'.format(book_id, batches_amount)), 'wb') as features_file:
        pickle.dump(X, features_file)


def save_features_csv(X, book_id, batches_amount, handler):
    with open(os.path.join('resources',
                           'features',
                           'csv',
                           '{}_{}.csv'.format(book_id, batches_amount)), 'w') as csvfile:
        fieldnames = handler.features_names_list
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in X:
            writer.writerow({feature: row[ind] for ind, feature in enumerate(fieldnames)})


def save_labels(y, book_id, model):
    save_via_pickle(y, os.path.join('resources', 'labels', '{}_{}'.format(book_id, model)))


def load_features(book_id, batches_amount):
    return load_from_pickle(os.path.join('resources',
                                         'features',
                                         '{}_{}.pkl'.format(book_id, batches_amount)))


def load_labels(book_id, model):
    return load_from_pickle(os.path.join('resources', 'labels', '{}_{}.pkl'.format(book_id, model)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare_features', help='Prepare and save features', action='store_true')
    parser.add_argument('--train_book_id', help='Train on book ID', type=int, metavar='ID', required=False)
    parser.add_argument('--test_book_id', help='Test on book ID', type=int, metavar='ID', required=False)
    parser.add_argument('--batches_amount', help='Split train book to N parts', type=int, metavar='N', required=False)
    parser.add_argument('--prepare_labels', help='Prepare and save labels', action='store_true')
    parser.add_argument('--train_model', help='Train on model M', type=str, metavar='M', required=False)
    parser.add_argument('--test_model', help='Test on book M', type=str, metavar='M', required=False)
    parser.add_argument('--train_cluster', help='Train on model C', type=str, metavar='M', required=False)
    parser.add_argument('--test_cluster', help='Test on book C', type=str, metavar='C', required=False)
    parser.add_argument('--predict_swipes', help='Predict the swipes', action='store_true')

    args = parser.parse_args()

    if args.prepare_features:
        train_handler = TextFeaturesHandler(args.train_book_id, batches_amount=args.batches_amount)
        X_train = train_handler.get_features()
        save_features(X_train, args.test_book_id, train_handler.batches_amount)
        save_features_csv(X_train, args.test_book_id, train_handler.batches_amount, train_handler)
        test_handler = TextFeaturesHandler(args.test_book_id, batches_amount=None, batch_size=train_handler.batch_size)
        X_test = test_handler.get_features()
        save_features(X_test, args.test_book_id, test_handler.batches_amount)
        save_features_csv(X_test, args.test_book_id, test_handler.batches_amount, test_handler)

    if args.prepare_labels:
        for book, model, cluster in [(210901, '2_100_agglomerative.pkl', 1),
                                     (210901, '2_100_k_means.pkl', 0),
                                     (210901, '2_100_spectral.pkl', 0),
                                     (215591, '2_114_agglomerative.pkl', 0),
                                     (215591, '2_114_k_means.pkl', 0),
                                     (215591, '2_114_spectral.pkl', 1),
                                     (259222, '2_107_agglomerative.pkl', 0),
                                     (259222, '2_107_k_means.pkl', 1),
                                     (259222, '2_107_spectral.pkl', 1)]:
            logging.info('Book {}, model {}'.format(book, model))
            y = identify_swipe_places(book, model, cluster)
            print(y)
            save_labels(y, book, model)

    if args.predict_swipes:
        X_train = load_features(210901, 100)
        train_models = ['2_100_agglomerative', '2_100_k_means', '2_100_spectral']
        y_train = load_labels(210901, train_models[0])
        X_test = load_features(215591, 114)
        y_test = load_labels(215591, '2_114_agglomerative')
        for clf in [DecisionTreeClassifier(),
                    KNeighborsClassifier(),
                    SVC(gamma='auto'),
                    RandomForestClassifier(n_estimators=1000, criterion='entropy')]:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(list(zip(y_test, y_pred)))
            print('F1: {}, precision: {}, recall: {}'.format(f1_score(y_test, y_pred),
                                                             precision_score(y_test, y_pred),
                                                             recall_score(y_test, y_pred)))
