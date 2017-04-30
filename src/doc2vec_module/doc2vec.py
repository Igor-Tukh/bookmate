from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import glob
import os
import json
from process_fb2 import contains_end_of_sentence
from random import shuffle
import pickle
import sys


def iterate_lemmatized_sentences(parsed_text):
    json_text = json.loads(parsed_text)
    sentence = []
    for token in json_text:
        if token['text'] == '\\s':
            continue
        if 'analysis' in token:
            if len(token['analysis']) != 0 and 'lex' in token['analysis'][0]:
                sentence.append(token['analysis'][0]['lex'])
        else:
            if contains_end_of_sentence(token['text']):
                yield sentence
                sentence = []
    if len(sentence) != 0:
        yield sentence


class FlibustaMystemmedSentences(object):
    def __iter__(self):
        mystem_out_filenames = glob.glob('mystem_out/*.mystem.out')
        step = 100
        index = 0
        for filename in mystem_out_filenames:
            flibusta_id = filename.split('/')[-1].split('.')[0]
            symbols = 0
            max_symbols = 1000
            stop = False
            with open(filename, 'r') as paragraphs:
                for paragraph in paragraphs:
                    paragraph = paragraph[:-1]
                    for sentence in iterate_lemmatized_sentences(paragraph):
                        symbols += sum(len(x) for x in sentence)
                        if symbols <= max_symbols:
                            yield LabeledSentence(sentence, [flibusta_id])
                        else:
                            stop = True
                            break
                    if stop:
                        break
            index += 1
            if index % step == 0:
                print('{0} books processed'.format(index))

    def collect_data(self):
        print('Collecting started')
        self._sentences = []
        for sentence in self:
            self._sentences.append(sentence)
        print('Collecting done')

    @property
    def data(self):
        return self._sentences

    def shuffle(self):
        shuffle(self._sentences)


def collect_and_store_book_stats(sentences):
    sentence_counts = {}
    word_counts = {}
    letter_counts = {}

    for sentence in sentences:
        words = sentence.words
        tag = sentence.tags[0]
        for counts in (word_counts, sentence_counts, letter_counts):
            if tag not in counts:
                counts[tag] = 0
        sentence_counts[tag] += 1
        word_counts[tag] += len(words)
        letter_counts[tag] += sum((len(word) for word in words))

    with open('counts.pkl', 'w') as f:
        pickle.dump((word_counts, letter_counts, sentence_counts), f)


def build_and_store_word_map():
    sentences = FlibustaMystemmedSentences()
    words = set()

    for sentence in sentences:
        for word in sentence[0]:
            words.add(word)

    word_map = {word: i for i, word in enumerate(words)}

    with open('word_map.pkl', 'w') as f:
        pickle.dump(word_map, f)


sentences = FlibustaMystemmedSentences()
sentences.collect_data()

model = Doc2Vec(workers=8, window=10, size=300, negative=5, iter=1)

print('Building vocabulary started')
model.build_vocab(sentences.data)
print('Building vocabulary done')

print('Training started')
for epoch in range(10):
    print('Epoch #{0}'.format(epoch))
    sentences.shuffle()
    model.train(sentences.data)
print('Training done')

model.save('flbst.d2v')
