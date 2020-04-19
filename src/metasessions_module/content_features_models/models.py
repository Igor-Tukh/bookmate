import logging
import csv
import os
import argparse

from sklearn.linear_model import LinearRegression, Ridge
from sklearn import svm
from src.metasessions_module.content_features_models.prepare import prepare_regression_data

log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
root_logger = logging.getLogger()

file_handler = logging.FileHandler(os.path.join('logs', 'interest_markers.log'), 'a')
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)

REGRESSION_MODELS = [(LinearRegression, 'linear regression'), (Ridge, 'ridge'), (svm.SVR, 'SVM')]


def try_regression_model(book, model, model_name):
    logging.info(f'Trying model {model_name}')
    scores = []
    for shuffle in [False, True]:
        for data, description in prepare_regression_data(book, shuffle=shuffle):
            current_model = model()
            current_model.fit(data[0], data[2])
            score = current_model.score(data[1], data[3])
            # 1 - u / v
            scores.append({'model': model_name, 'shuffle': shuffle, 'score': score, 'description': description})
            logging.info(f'Model: {model_name}, shuffle: {shuffle},  score: {score}, description: {description}')

    return scores


def try_regression_models(book, results_path=None, ignore_models_set=None):
    if ignore_models_set is None:
        ignore_models_set = {}

    logging.info('Models estimation started')

    all_results = []
    for model, model_name in REGRESSION_MODELS:
        if model_name not in ignore_models_set:
            all_results.extend(try_regression_model(book, model, model_name))

    logging.info('Models estimation finished')

    if results_path is not None and len(all_results) > 0:
        logging.info(f'Saving estimates to path {results_path} started')
        with open(results_path, 'w+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            for results in all_results:
                writer.writerow(results)
        logging.info(f'Saving estimates to path {results_path} finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--book_id', type=int, required=True)
    args = parser.parse_args()
    book_id = args.book_id
    try_regression_models(book_id, results_path=os.path.join('resources', 'interest_markers_models',
                                                             f'results_book_{book_id}.csv'))
