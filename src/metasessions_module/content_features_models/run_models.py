import os
import sys
import argparse
import numpy as np


sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))

from src.metasessions_module.content_features.feautres_builder import TextFeaturesBuilder
from src.metasessions_module.content_features_models.models_utils import load_regression_markers, update_stats
from src.metasessions_module.config import RANDOM_SEED, BOOK_ID_TO_TITLE
from src.metasessions_module.content_features.bert_embeddings import get_embeddings
from src.metasessions_module.utils import save_via_pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn import tree
from sklearn import svm


from tqdm import tqdm

MODELS = [('Linear Regression', linear_model.LinearRegression, {}),
          ('Ridge', linear_model.Ridge, {'alpha': 0.5}),
          ('Lasso', linear_model.Lasso, {'alpha': 0.1}),
          ('ElasticNet', linear_model.ElasticNet, {'random_state': RANDOM_SEED}),
          ('ElasticNetCV', linear_model.ElasticNetCV, {'cv': 5, 'random_state': RANDOM_SEED}),
          ('LARS', linear_model.Lars, {}),
          ('SGD Regressor', linear_model.SGDRegressor, {'random_state': RANDOM_SEED}),
          ('SVM Regressor', svm.SVR, {'degree': 5}),
          ('K Neighbors Regressor', KNeighborsRegressor, {}),
          ('Decision Tree Regressor', tree.DecisionTreeRegressor, {'random_state': RANDOM_SEED}),
          ('Ada Boost Regressor', AdaBoostRegressor, {'random_state': RANDOM_SEED,
                                                      'base_estimator': tree.DecisionTreeRegressor()}),
          ('Random Forest Regressor', RandomForestRegressor, {'random_state': RANDOM_SEED, 'n_estimators': 1000})]

METRICS = [('Explained Variance Score', explained_variance_score),
           ('Mean Absolute Error', mean_absolute_error),
           (' Mean Squared Error', mean_squared_error),
           ('R2 Score', r2_score)]


def combine_features(fs, es):
    return {'bert': es,
            'features': fs,
            'combined': np.hstack([fs, es])}


def get_models_predictions_path(book, filename):
    output_path = os.path.join('resources', 'models', str(book), 'predictions')
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)


def try_sklearn_models(book, x, all_y, x_embeddings, save_predictions=True):
    for y_description, y in tqdm(all_y.items()):
        results = []
        for shuffle in [False, True]:
            for test_size in [0.5, 0.25]:
                for features_description, f in combine_features(x, x_embeddings).items():
                    x_train, x_test, y_train, y_test = train_test_split(f, y, shuffle=shuffle, random_state=RANDOM_SEED,
                                                                        test_size=test_size)
                    for model_description, model, params in MODELS:
                        current_results = {'Marker': y_description, 'Shuffle': str(shuffle), 'Model': model_description,
                                           'Test size': str(test_size), 'Features': features_description}
                        model = model(**params).fit(x_train, y_train)
                        y_pred = model.predict(x_test)
                        for metrics_description, metrics in METRICS:
                            score = metrics(y_test, y_pred)
                            current_results[metrics_description] = score

                        results.append(current_results)
                        if save_predictions:
                            predictions_name = f'{y_description}_{shuffle}_{model_description}' \
                                               f'_{test_size}_{features_description}.pkl'
                            save_via_pickle((y, model.predict(f)), get_models_predictions_path(book, predictions_name))
            update_stats(book, results, y_description)


def try_sklearn_two_books(book_1, book_2,
                          book_features_1, book_features_2,
                          book_embeddings_1, book_embeddings_2,
                          book_markers_1, book_markers_2):
    for marker_description in tqdm(list(book_markers_1.keys())[4:]):
        results = []
        combined_features_1 = combine_features(book_features_1, book_embeddings_1)
        combined_features_2 = combine_features(book_features_2, book_embeddings_2)
        y_train = book_markers_1[marker_description]
        y_test = book_markers_2[marker_description]

        for features_description in combined_features_1.keys():
            x_train = combined_features_1[features_description]
            x_test = combined_features_2[features_description]

            for model_description, model, params in MODELS:
                current_results = {'Marker': marker_description, 'Model': model_description,
                                   'Features': features_description, 'Train book': BOOK_ID_TO_TITLE[book_1],
                                   'Test book': BOOK_ID_TO_TITLE[book_2]}
                model = model(**params).fit(x_train, y_train)
                y_pred = model.predict(x_test)
                for metrics_description, metrics in METRICS:
                    score = metrics(y_test, y_pred)
                    current_results[metrics_description] = score

                results.append(current_results)
        update_stats(book_1, results, f'{book_2}_{marker_description}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=int, required=True)
    parser.add_argument('--second_book_id', type=int)
    args = parser.parse_args()
    book_id_1 = args.book_id
    book_id_2 = args.second_book_id

    if book_id_2:
        features_1 = TextFeaturesBuilder.load_for_book(book_id_1)
        features_2 = TextFeaturesBuilder.load_for_book(book_id_2)
        markers_1 = load_regression_markers(book_id_1)
        markers_2 = load_regression_markers(book_id_2)
        bert_embeddings_1 = np.array(get_embeddings(book_id_1))
        bert_embeddings_2 = np.array(get_embeddings(book_id_2))

        try_sklearn_two_books(book_id_1, book_id_2, features_1, features_2, bert_embeddings_1, bert_embeddings_2,
                              markers_1, markers_2)
    else:
        features = TextFeaturesBuilder.load_for_book(book_id_1)
        markers = load_regression_markers(book_id_1)
        bert_embeddings = np.array(get_embeddings(book_id_1))

        try_sklearn_models(book_id_1, features, markers, bert_embeddings)
