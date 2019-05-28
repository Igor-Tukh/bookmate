import csv
import os
import nltk
import logging
import numpy as np

from src.metasessions_module.text_utils import load_text
from src.metasessions_module.config import *
from pymystem3 import Mystem
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords


class TextFeaturesHandler(object):
    """
    Class which stores books text and builds features every text batch. Batch is a continues subrange of a text (of the
    same size). Batches amount is an argument and in case if it is None, it is calculated as ratio of text length and
    batch size (in this case batch_size argument is required).
    """

    def __init__(self, book_id, batches_amount=100, batch_size=None):
        logging.info('Creating features handler for book {}'.format(book_id))
        self.book_id = book_id
        self.text = load_text(book_id)
        if batches_amount is None:
            self.batches_amount = round((len(self.text) + batch_size - 1) / batch_size)
        else:
            self.batches_amount = batches_amount

        self.batch_size = 1. * len(self.text) / self.batches_amount
        self.batches_borders = [(round(ind * self.batch_size), round(min((ind + 1) * self.batch_size, len(self.text))))
                                for ind in range(self.batches_amount)]
        logging.info('Batches amount: {}, batch size: {}'.format(self.batches_amount, self.batch_size))
        self.fix_borders()
        self.batches_text = [self.text[first:last] for first, last in self.batches_borders]
        self.mystem = Mystem()
        logging.info('Building lemmatized tokens')
        self.lemmatized_tokens = [self.mystem.lemmatize(self.text[first:last]) for first, last in self.batches_borders]
        logging.info('Building tagged tokens')
        self.tagged_tokens = [pos_tag(word_tokenize(self.text[first:last]), lang='rus')
                              for first, last in self.batches_borders]
        nltk.download("stopwords")

        self.emotional_verbs = set()
        self.load_emotional_verbs()

        self.features_calculators = {'nouns_percent': self.get_nouns_percent,
                                     'verbs_percent': self.get_verbs_percent,
                                     'parts_percent': self.get_parts_percent,
                                     'average_word_len': self.get_average_word_length,
                                     'stopwords_percent': self.get_stopwords_percent,
                                     'average_sentence_len': self.get_average_sentence_length,
                                     'characters_names_percent': self.get_characters_names_percent,
                                     'main_characters_names_percent': self.get_main_characters_names_percent,
                                     'emotional_verbs_percent': self.get_emotional_verbs_percent,
                                     'book_placement_percent': self.get_book_placement_percent,
                                     'bodyparts_percent': self.get_bodyparts_percent}

        self.features_names_list = list(self.features_calculators.keys())

    def get_nouns_percent(self, ind):
        count = 0
        for _, tag in self.tagged_tokens[ind]:
            if tag == 'S':
                count += 1
        return 1. * count / len(self.tagged_tokens[ind])

    def get_verbs_percent(self, ind):
        count = 0
        for _, tag in self.tagged_tokens[ind]:
            if tag == 'V':
                count += 1
        return 1. * count / len(self.tagged_tokens[ind])

    def get_parts_percent(self, ind):
        count = 0
        for _, tag in self.tagged_tokens[ind]:
            if tag == 'PART':
                count += 1
        return 1. * count / len(self.tagged_tokens[ind])

    def get_average_word_length(self, ind):
        return np.mean(np.array([len(word) for word, _ in self.tagged_tokens[ind]]))

    def get_average_sentence_length(self, ind):
        sentences = nltk.sent_tokenize(self.batches_text[ind], language="russian")
        return np.mean(np.array([len(sentence) for sentence in sentences]))

    def get_characters_names_percent(self, ind):
        names, total_amount = self.get_characters_name(ind)
        return 1. * len(names) / total_amount

    def get_main_characters_names_percent(self, ind):
        names = self.get_characters_name(ind)[0]
        if len(names) == 0:
            return 0.0
        main_characters_amount = 0
        main_names = set(map(lambda s: s.lower(), MAIN_CHARACTERS[self.book_id]))
        for name in names:
            main_characters_amount += 1 if name in main_names else 0
        return 1. * main_characters_amount / len(names)

    def get_emotional_verbs_percent(self, ind):
        emotional_verbs_amount = 0
        total_verbs_amount = 0
        for info in self.mystem.analyze(self.batches_text[ind]):
            if 'analysis' not in info or len(info['analysis']) == 0:
                continue
            info = info['analysis'][0]
            if 'gr' in info and info['gr'].split(',')[0] == 'V' and 'lex' in info:
                total_verbs_amount += 1
                emotional_verbs_amount += 1 if info['lex'] in self.emotional_verbs else 0
        return 1. * emotional_verbs_amount / total_verbs_amount

    def get_bodyparts_percent(self, ind):
        bodyparts_amount = 0
        total_amount = 0
        for info in self.mystem.analyze(self.batches_text[ind]):
            if 'analysis' not in info or len(info['analysis']) == 0:
                continue
            total_amount += 1
            info = info['analysis'][0]
            if 'lex' in info:
                bodyparts_amount += 1 if info['lex'].lower() in BODYPARTS else 0

        return 1. * bodyparts_amount / total_amount

    def get_characters_name(self, ind):
        names = []
        characters = set(map(lambda s: s.lower(), CHARACTERS[self.book_id]))
        total_amount = 0
        for word_info in self.mystem.analyze(self.batches_text[ind]):
            total_amount += 1
            if 'analysis' not in word_info or len(word_info['analysis']) == 0:
                continue
            info = word_info['analysis'][0]
            if info['lex'] in characters:
                names.append(info['lex'])
        return names, total_amount

    def get_stopwords_percent(self, ind):
        count = 0
        for word in self.lemmatized_tokens[ind]:
            if word in stopwords.words('russian'):
                count += 1
        return 1. * count / len(self.tagged_tokens[ind])

    def get_book_placement_percent(self, ind):
        return 1. * ind / self.batches_amount

    def fix_borders(self):
        """
        Moves borders (left to the back and right to the right) in a purpose put first and last words to batch text
        completely.
        :return: nothing
        """
        for ind, (first, last) in enumerate(self.batches_borders):
            while self.text[first] != ' ' and first > 0:
                first -= 1
            while last < len(self.text) and self.text[last] != ' ':
                last += 1
            self.batches_borders[ind] = (first, last)

    def load_emotional_verbs(self):
        logging.info('Loading emotional verbs')
        with open(os.path.join('resources', 'vocabulary', 'verbs', 'verbs_emotional.csv')) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['emotional'] == '+':
                    self.emotional_verbs.add(row['verb'])

    def get_features(self, features_names=None):
        if features_names is None:
            features_names = self.features_names_list
        return np.array([[self.features_calculators[feature_name](ind) for feature_name in features_names]
                         for ind in range(self.batches_amount)])
