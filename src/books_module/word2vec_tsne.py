import numpy as np
from sklearn.manifold import TSNE
from pymongo import MongoClient
import matplotlib.pyplot as plt


def connect_to_database_books_collection():
    client = MongoClient('localhost', 27017)
    global db
    db = client.bookmate
    return db


def tsne_for_windows(book_id: str) -> list():
    db = connect_to_database_books_collection()
    table_name = book_id + '_window'
    windows = db[table_name].find()
    model = TSNE(learning_rate=10, n_components=2, perplexity=7, n_iter=10000)
    np.set_printoptions(suppress=True)

    vectors_list = list()
    for window in windows:
        vectors_list.append(window['vector'])

    vectors_list = np.array(vectors_list)
    new_word2vec = model.fit_transform(vectors_list)
    return new_word2vec

def plot_word2vec(word2vec_points: np.array) -> None:
    plt.clf()
    x, y = list(), list()
    for point in word2vec_points:
        x.append(point[0])
        y.append(point[1])
    plt.scatter(x, y)
    plt.savefig('word2vec.png')


book_id = '12193260'
word2vec_low_dimension = tsne_for_windows(book_id)
plot_word2vec(word2vec_low_dimension)
