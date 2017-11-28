import csv
import random
import pandas as pd
import pickle
BOOKS=['210901']


def data_generation_from_features(pages_num, output_label):
    train_x, train_y, test_x, test_y = list(), list(), list(), list()

    for book_id in BOOKS:
        input_reader = csv.reader(open('resources/%s_features.csv' % book_id, 'r'))
        output_reader = csv.DictReader(open('resources/%s_groundtruth.csv' % book_id, 'r'))

        features, groundtruth = list(), list()
        features_lists = list(input_reader)
        for features_list in features_lists:
            if len(features_list) != 0:
                features.append([float(x) for x in features_list])
        for labels in output_reader:
            groundtruth.append(float(labels[output_label]))

        for page_num in range(0, len(features) - pages_num):
            sample = list()
            for i in range(0, pages_num):
                sample.append(features[i])
            if random.randint(1, 100) > 80:
                test_x.append(sample)
                test_y.append([groundtruth[page_num + pages_num]])
            else:
                train_x.append(sample)
                train_y.append([groundtruth[page_num + pages_num]])

    return train_x, train_y, test_x, test_y


def data_generation_from_one_hot(output_label):
    train_x, train_y, test_x, test_y = list(), list(), list(), list()

    for book_id in BOOKS:
        features = pickle.load(open('resources/%s_one_hot.pkl' % book_id, 'rb'))
        output_reader = csv.DictReader(open('resources/%s_groundtruth.csv' % book_id, 'r'))

        groundtruth = list()
        for labels in output_reader:
            groundtruth.append(float(labels[output_label]))

        for i in range(0, len(features)):
            if random.randint(1, 100) > 80:
                test_x.append(features[i])
                test_y.append([groundtruth[i]])
            else:
                train_x.append(features[i])
                train_y.append([groundtruth[i]])

    return train_x, train_y, test_x, test_y


def get_batch(x, y, batch_size, offset):
    if offset + batch_size >= len(y):
        batch_size = len(y) - offset - 1
    indexes = range(offset, offset + batch_size)
    x_batch, y_batch = list(), list()

    for i in indexes:
        x_batch.append(x[i])
        y_batch.append(y[i])

    return x_batch, y_batch


def results_to_csv(predictions, y):
    with open('resources/predictions.csv', 'w') as outfile:
        data = pd.DataFrame()
        predictions_list = [x[0] for x in predictions]
        y_list = [x[0] for x in y]
        data['predictions'] = predictions_list
        data['y'] = y_list
        data.to_csv('resources/predictions.csv')


# data_generation_from_features(5, 'page_speed')
data_generation_from_one_hot('page_speed')

