from pymystem3 import Mystem
from os import listdir
from os.path import isfile, join
import numpy as np
import gensim
import nltk
from stop_words import get_stop_words
import string


mystem = Mystem()
punctuation = string.punctuation
punctuation += '—'
punctuation += '…'
ru_stopwords = get_stop_words('russian')
model = gensim.models.Word2Vec.load_word2vec_format('all.norm-sz500-w10-cb0-it3-min5.w2v', binary = True,
                                                    unicode_errors='ignore')


def process_files():
    books_in = [f for f in listdir('text_in') if isfile(join('text_in', f))]
    books_out = [f for f in listdir('text_out') if isfile(join('text_out', f))]

    for book in books_in:
        if book in books_out:
            continue
        vectors_file = open('vectors_out/%s' % book, 'w+')
        # clean_book_text = open('text_out/%s' % book, 'w+')
        with open('text_in/%s' % book) as file:
            pages = file.readlines()
            pages = [x.strip() for x in pages]
        for page in pages:
            page_out_string = ''
            words_count = 0
            page_vector = np.zeros(model.vector_size)
            lemmas = mystem.lemmatize(page)
            text = ''.join(lemmas)
            words = nltk.word_tokenize(text)
            # clean_book_text.write(words)
            for word in words:
                if word in punctuation:
                    continue
                if word not in ru_stopwords:
                    try:
                        page_vector += model[word]
                        words_count += 1
                    except:
                        print('There is no word %s in word2vec model' % word)
                        pass

            page_vector /= words_count
            page_vector = page_vector.tolist()
            string_vector = ''
            for node in page_vector:
                string_vector += str(node)
                string_vector += '\t'
            string_vector = string_vector[:-2]
            string_vector += '\n'
            vectors_file.write(string_vector)


process_files()

