from gensim.models import Doc2Vec
import csv
import sys


model = Doc2Vec.load('../resources/flbst.d2v')


def process_book(book_id):
    tsvfile = open('../resources/books/%s_v.tsv', 'w')
    writer = csv.writer(tsvfile, delimiter='\t', newline='\n')
    with open('../resources/books/%s.csv' % book_id, encoding='utf-8') as f:
        pages = csv.reader(f)
        for page in pages:
            writer.writerow(model.infer_vector(page[0].split(' ')))


def main():
    for book_id in sys.argv[1:]:
        process_book(book_id)


if __name__ == "__main__":
    main()