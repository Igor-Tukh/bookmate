import pickle
import os


def count(key, iterable):
    return len(filter(key, iterable))


def get_top_keys(dictionary, comparator):
    keys = ((x, comparator(x)) for x in dictionary.keys())
    return sorted(keys, reverse=True, key=lambda x:x[1])


def save_top(books_index, top_book_ids, f, measurement_type):
    index = 0
    for top_book_id, key in top_book_ids:
        index += 1
        title = 'unknown'
        authors = 'unknown'
        if top_book_id in books_index:
            title = books_index[top_book_id]['title']
            authors = books_index[top_book_id]['authors']
        f.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(
            index, title, authors, top_book_id, key, measurement_type))


def sort_histogram(histogram):
    return ((x, histogram[x]) for x in
            sorted(histogram.keys(), reverse=True, key=lambda x:histogram[x]))


def load_file(filename):
    with open('dumps/' + filename + '.pk', 'rb') as f:
        obj = pickle.load(f)
    print('{0} is loaded'.format(filename))
    return obj


def check_dir_existance(directory):
    if directory.endswith('/'):
        directory = directory[:-1]
    if not os.path.exists(directory):
        os.makedirs(directory)
