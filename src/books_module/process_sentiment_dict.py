import json
import gensim


sentiment_dictionary_txt_file = '../resources/sentiment_dictionary.txt'
sentiment_dictionary_json_file = '../resources/sentiment_dictionary.json'


def extend_dictionary_with_word2vec(sentiment_dictionaty):
    word2vec_model = gensim.models.Word2Vec.load_word2vec_format('../../ruscorpora_russe.model.bin', binary=True,
                                                              unicode_errors='ignore')
    similar_words_dictionary = dict()
    skipped_words_number = 0
    sentiment_dictionary_size = len(sentiment_dictionary)
    for word in sentiment_dictionary:
        value = sentiment_dictionary[word]
        try:
            most_similar_words = word2vec_model.most_similar(word, topn = 10)
            for similar_word in most_similar_words:
                similar_words_dictionary[similar_word[0]] = value
        except:
            skipped_words_number += 1
            print ('Already skipped %s/%s words' % (str(skipped_words_number), str(sentiment_dictionary_size)))
            print ('Cannot find word %s in vocabulary, skipping' % (word))
            continue

    sentiment_dictionary.update(similar_words_dictionary)


with open(sentiment_dictionary_txt_file) as f:
    content = f.readlines()

sentiment_dictionary = dict()
for line in content:
    split_line = line.split()
    sentiment_dictionary[split_line[0].lower()] = split_line[1]


extend_dictionary_with_word2vec(sentiment_dictionary)

with open(sentiment_dictionary_json_file, 'w') as f:
    json.dump(sentiment_dictionary, f, ensure_ascii=False)