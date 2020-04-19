import csv
import os
import numpy as np
import stanfordnlp
import warnings
import pickle

from pymystem3 import Mystem
from tqdm import tqdm
from nltk import pos_tag, word_tokenize
from collections import defaultdict

from src.metasessions_module.config import BODYPARTS, CHARACTERS, MAIN_CHARACTERS, PERSONAL_PRONOUNS
from src.metasessions_module.content_features.sentiment_utils import get_word_sentiment, load_words_sentiment_dict
from src.metasessions_module.utils import load_from_pickle, save_via_pickle


def _get_sentence_depth(sentence):
    def dfs(v, children, used):
        used[v] = 1

        max_depth = 0
        for child in children:
            if not used[child]:
                max_depth = max(max_depth, dfs(child, children, used))

        return max_depth + 1

    children = defaultdict(lambda: [])
    used = defaultdict(lambda: False)
    for word in sentence.words:
        children[word.governor].append(int(word.index))

    return dfs(0, children, used)


class TextFeaturesBuilder(object):
    FEATURES_NAMES = ['get_bodyparts_percent', 'get_characters_names_percent',
                      'get_main_characters_names_percent', 'get_emotional_verbs_percent',
                      'get_sentiment', 'get_average_word_len', 'get_average_word_len',
                      'get_personal_pronouns_percent', 'get_nouns_percent', 'get_verbs_percent',
                      'get_adjectives_percent', 'get_average_dependency_tree_depth']

    def __init__(self, texts_path, book_id=210901):
        texts_path = os.path.join(texts_path, str(book_id))
        warnings.filterwarnings("ignore")
        self.texts = []
        for ind in range(len(os.listdir(texts_path))):
            text_filename = f'text_{ind}.txt'
            with open(os.path.join(texts_path, text_filename), 'r') as text_file:
                self.texts.append(text_file.read())

        self.pos = []
        self.tokens = []
        for text in self.texts:
            self.tokens.append(word_tokenize(text.lower()))
            self.pos.append(pos_tag(self.tokens[-1], lang='rus'))

        self.n_texts = len(self.texts)

        self.emotional_verbs = set()
        with open(os.path.join('resources', 'vocabulary', 'verbs', 'verbs_emotional.csv')) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['emotional'] == '+':
                    self.emotional_verbs.add(row['verb'])
        self.mystem = Mystem()
        self.words_sentiment_dict = load_words_sentiment_dict()

        self.POSES = ['noun', 'verb', 'adjective']
        self.features_number = len(self.FEATURES_NAMES)
        self.features = np.zeros((self.n_texts, self.features_number), dtype=np.float32)
        self.nlp = stanfordnlp.Pipeline(lang='ru')
        # self.docs = [self.nlp(text) for text in self.texts]
        self.docs = []
        for text in tqdm(self.texts):
            self.docs.append(self.nlp(text))

    @staticmethod
    def get_default_path():
        return os.path.join('resources', 'saved_features', 'features.pkl')

    def save_features(self, path=None):
        if path is None:
            path = TextFeaturesBuilder.get_default_path()
        save_via_pickle(self.features, path)

    @staticmethod
    def load_features(path=None):
        if path is None:
            path = TextFeaturesBuilder.get_default_path()
        return load_from_pickle(path)

    def build(self):
        for ind, feature in enumerate(self.FEATURES_NAMES):
            try:
                method = getattr(self, feature)
                self.features[:, ind] = method()
            except ArithmeticError:
                pass

        return self.features

    def get_average_dependency_tree_depth(self):
        depths = np.zeros(self.n_texts, dtype=np.float32)
        for doc_ind, doc in enumerate(self.docs):
            depths[doc_ind] = np.mean([_get_sentence_depth(sentence) for sentence in doc.sentences])
        return depths

    def get_nouns_percent(self):
        return self.get_pos_percent('noun')

    def get_verbs_percent(self):
        return self.get_pos_percent('verb')

    def get_adjectives_percent(self):
        return self.get_pos_percent('adjective')

    def get_important_fragments(self, feature_name, window_size=7, feature_params=None):
        if feature_params is None:
            feature_params = []
        important_texts = []
        try:
            method = getattr(self, feature_name)
            feature = method(*feature_params)
            for ind, value in enumerate(feature):
                if value == np.max(feature[max(ind - window_size, 0): min(window_size + ind + 1, len(feature))]):
                    important_texts.append((self.texts[ind], 'max', value, ind))
                if value == np.min(feature[max(ind - window_size, 0): min(window_size + ind + 1, len(feature))]):
                    important_texts.append((self.texts[ind], 'min', value, ind))
        except AttributeError:
            pass

        return important_texts

    def get_average_word_len(self):
        avg_len = np.zeros(self.n_texts, dtype=np.float32)
        for text_ind in range(self.n_texts):
            avg_len[text_ind] = np.mean(list(map(len, self.tokens[text_ind])))
        return avg_len

    def get_sentiment(self):
        percent = np.zeros(self.n_texts, dtype=np.float32)

        for text_ind in tqdm(range(self.n_texts)):
            sentiments = []
            avg_sentiments = []
            lemmas = self.mystem.lemmatize(self.texts[text_ind])
            for lemma_ind, lemma in enumerate(lemmas):
                sentiments.append(get_word_sentiment(self.words_sentiment_dict, lemma))

            for lemma_ind, lemma in enumerate(lemmas):
                avg_sentiments.append(sentiments[lemma_ind])
                # avg_sentiments.append(np.mean(sentiments[max(lemma_ind - 1, 0): min(lemma_ind + 2, len(sentiments))]))

            percent[text_ind] = np.mean(avg_sentiments)

        return percent

    def get_emotional_verbs_percent(self):
        percent = np.zeros(self.n_texts, dtype=np.float32)
        for text_ind in range(self.n_texts):
            lemmas = self.mystem.lemmatize(self.texts[text_ind])
            emotional_verbs_number = 0
            for lemma in lemmas:
                if lemma in self.emotional_verbs:
                    emotional_verbs_number += 1

            percent[text_ind] = 1. * emotional_verbs_number / len(lemmas)
        return percent

    def get_main_characters_names_percent(self):
        percent = np.zeros(self.n_texts, dtype=np.float32)
        for text_ind in range(self.n_texts):
            percent[text_ind] = 1. * self._get_word_from_list_number(
                text_ind, MAIN_CHARACTERS[210901]) / len(self.tokens[text_ind])
        return percent

    def get_characters_names_percent(self):
        percent = np.zeros(self.n_texts, dtype=np.float32)
        for text_ind in range(self.n_texts):
            percent[text_ind] = 1. * self._get_word_from_list_number(text_ind,
                                                                     CHARACTERS[210901]) / len(self.tokens[text_ind])
        return percent

    def get_personal_pronouns_percent(self):
        percent = np.zeros(self.n_texts, dtype=np.float32)
        for text_ind in range(self.n_texts):
            percent[text_ind] = 1. * self._get_word_from_list_number(text_ind,
                                                                     PERSONAL_PRONOUNS) / len(self.tokens[text_ind])
        return percent

    def get_bodyparts_percent(self):
        percent = np.zeros(self.n_texts, dtype=np.float32)
        for text_ind in range(self.n_texts):
            percent[text_ind] = 1. * self._get_word_from_list_number(text_ind, BODYPARTS) / len(self.tokens[text_ind])
        return percent

    def _get_word_from_list_number(self, text_ind, words_list):
        words = set([word.lower() for word in words_list])
        words_number = 0
        for token in self.tokens[text_ind]:
            if token in words:
                words_number += 1
        return words_number

    def get_pos_percent(self, pos=''):
        percent = np.zeros(self.n_texts, dtype=np.float32)
        try:
            method = getattr(self, f'_get_{pos}s_percent')
            for ind in tqdm(range(self.n_texts)):
                percent[ind] = method(ind)
        except AttributeError:
            pass
        return percent

    def _get_nouns_percent(self, text_ind):
        return self._get_pos_tag_percent(text_ind, 'S')

    def _get_verbs_percent(self, text_ind):
        return self._get_pos_tag_percent(text_ind, 'V')

    def _get_adjectives_percent(self, text_ind):
        return self._get_pos_tag_percent(text_ind, 'A')

    def _get_pos_tag_percent(self, text_ind, target_tag):
        words_with_tag_number = len([tag for _, tag in self.pos[text_ind] if tag == target_tag or tag[0] == target_tag])

        return 1. * words_with_tag_number / len(self.pos[text_ind])


if __name__ == '__main__':
    builder = TextFeaturesBuilder(os.path.join('resources', 'texts'))
    builder.build()
    builder.save_features()
    # print(TextFeaturesBuilder.load_features().shape)

