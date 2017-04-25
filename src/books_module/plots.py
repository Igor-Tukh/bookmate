from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn
import gensim

client = MongoClient('localhost', 27017)
db = client.bookmate
save_dir = '../results/'
# model = gensim.models.Word2Vec.load_word2vec_format('../resources/models/169962', binary = True,
#                                                     unicode_errors='ignore')

def create_folder(book_id):
    folder_name = db.books.find_one({'_id': book_id})['title']
    # TODO create folder for book here if it isn't exists
    try:
        os.mkdir(save_dir + folder_name)
    except:
        pass
    return folder_name


def smooth_points(Y, N = 10):
    new_Y = []
    for i in range(0, len(Y)):
        smooth_N = N
        if i - N < 0:
            smooth_N = i
            # new_Y.append(Y[i])
            # continue
        elif i + N >= len(Y):
            smooth_N = len(Y) - i - 1

        sum = 0
        for j in range(-smooth_N, smooth_N):
            sum += Y[i + j]
        sum /= ((2 * smooth_N) + 1)
        new_Y.append(sum)

    return new_Y


def params_plot_for_paragraphs(book_id, params, plot_name, smooth = True, smooth_N = 30):
    plt.clf()
    folder_name = create_folder(book_id)
    table = book_id
    X = np.arange(0, db[table].find().count())

    for param in params:
        Y = []
        items = db[table].find()
        for item in items:
            Y.append(item[param])
        if smooth:
            Y = np.array(smooth_points(Y, N = smooth_N))
        else:
            Y = np.array(Y)
        plt.plot(X, Y)

    plt.legend(params, loc = 'upper left')
    plt.tight_layout()

    if smooth:
        plot_name += '_smooth_' + str(smooth_N)
    plt.savefig(save_dir + folder_name + '/' + plot_name, bbox_inches='tight')


def params_plot_for_pages(book_id, params, plot_name, smooth = True, smooth_N = 30):
    plt.clf()
    folder_name = create_folder(book_id)
    windows_id = db['books'].find_one({'_id': book_id})['disjoint_windows_id']
    X = np.arange(0, len(windows_id))

    for param in params:
        Y = []
        for i in range(0, len(windows_id)):
            window = db[book_id + '_window'].find_one({'_id': windows_id[i]})
            Y.append(window[param])
        if smooth:
            Y = np.array(smooth_points(Y, N = smooth_N))
        else:
            Y = np.array(Y)
        plt.plot(X, Y)

    plt.legend(params, loc='upper left')
    plt.tight_layout()

    plot_name += ' _window'
    if smooth:
        plot_name += '_smooth_' + str(smooth_N)
    plt.savefig(save_dir + folder_name + '/' + plot_name, bbox_inches='tight')


def corr_window(book_id):
    plt.clf()
    table = book_id + '_window'
    folder_name = create_folder(book_id)
    print(str(db[table].find().count()) + ' items in book')

    dialoques_number, sentences_number, window_length_in_words, person_verbs_num = [], [], [], []
    person_pronouns_num, avr_word_length, avr_sentence_length = [], [], []
    new_words_count = []
    #     similarity_with_previous_window_all, similarity_with_previous_window_main = [], []


    for i in range(1, db[table].find().count()):
        item = db[table].find_one({'_id': i})
        dialoques_number.append(item['dialoques_number'])
        sentences_number.append(item['sentences_number'])
        window_length_in_words.append(item['window_length_in_words'])
        person_verbs_num.append(item['person_verbs_num'])
        person_pronouns_num.append(item['person_pronouns_num'])
        avr_word_length.append(item['average_word_length'])
        avr_sentence_length.append(item['avr_sentence_length'])
        new_words_count.append(item['new_words_count'])
    # similarity_with_previous_window_all.append(item['similarity_with_previous_window_by_main_words'])
    #         similarity_with_previous_window_main.append(item['similarity_with_previous_window_by_all_words'])

    data = pd.DataFrame()
    data['dialogues_number'] = np.array(dialoques_number)
    data['sentences_number'] = np.array(sentences_number)
    data['window_length_in_words'] = np.array(window_length_in_words)
    data['person_verbs_num'] = np.array(person_verbs_num)
    data['person_pronouns_num'] = np.array(person_pronouns_num)
    data['avr_word_length'] = np.array(avr_word_length)
    data['avr_sentence_length'] = np.array(avr_sentence_length)
    data['new_words_count'] = np.array(new_words_count)
    #     data['similarity_with_previous_window_all'] = np.array(similarity_with_previous_window_all)
    #     data['similarity_with_previous_window_main'] = np.array(similarity_with_previous_window_main)

    corr_df = data.corr(method='pearson')
    print("--------------- BOOK ---------------")
    print(db.books.find_one({'_id': book_id})['title'])
    print("--------------- WINDOW MODE CORRELATIONS ---------------")
    # print(corr_df.head(len(data)))
    # print("--------------- CREATE A HEATMAP ---------------")
    # Create a mask to display only the lower triangle of the matrix (since it's mirrored around its
    # top-left to bottom-right diagonal).
    mask = np.zeros_like(corr_df)
    mask[np.triu_indices_from(mask)] = True
    # Create the heatmap using seaborn library.
    # List if colormaps (parameter 'cmap') is available here: http://matplotlib.org/examples/color/colormaps_reference.html
    seaborn.heatmap(corr_df, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0, mask=mask, linewidths=2.5)

    # Show the plot we reorient the labels for each column and row to make them easier to read.
    plt.title("Correlation for windows(undisjoint)")
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.savefig(save_dir + folder_name + '/' + 'correlation_plot_for_windows(undisjoint).png', bbox_inches='tight')


def corr_paragraph(book_id):
    plt.clf()
    table = book_id
    folder_name = create_folder(book_id)
    person_pronouns_num, num_of_symbols, is_dialogue, num_of_sentences, person_verbs_num, num_of_words = [], [], [], [], [], []
    new_words_count = []
    for i in range(1, db[table].find().count() + 1):
        item = db[table].find_one({'_id': i})
        person_pronouns_num.append(item['person_pronouns_num'])
        num_of_symbols.append(item['num_of_symbols'])
        is_dialogue.append(item['is_dialogue'])
        num_of_sentences.append(item['num_of_sentences'])
        person_verbs_num.append(item['person_verbs_num'])
        num_of_words.append(item['num_of_words'])
        new_words_count.append(item['new_words_count'])

    data = pd.DataFrame()
    data['person_pronouns_num'] = np.array(person_pronouns_num)
    data['num_of_symbols'] = np.array(num_of_symbols)
    data['is_dialogue'] = np.array(is_dialogue)
    data['num_of_sentences'] = np.array(num_of_sentences)
    data['person_verbs_num'] = np.array(person_verbs_num)
    data['num_of_words'] = np.array(num_of_words)
    data['new_words_count'] = np.array(new_words_count)

    corr_df = data.corr(method='pearson')
    print("--------------- BOOK ---------------")
    print(db.books.find_one({'_id': book_id})['title'])
    print("--------------- PARAGRAPH MODE CORRELATIONS ---------------")
    # print(corr_df.head(len(data)))
    # print("--------------- CREATE A HEATMAP ---------------")
    # Create a mask to display only the lower triangle of the matrix (since it's mirrored around its
    # top-left to bottom-right diagonal).
    mask = np.zeros_like(corr_df)
    mask[np.triu_indices_from(mask)] = True
    # Create the heatmap using seaborn library.
    # List if colormaps (parameter 'cmap') is available here: http://matplotlib.org/examples/color/colormaps_reference.html
    seaborn.heatmap(corr_df, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0, mask=mask, linewidths=2.5)

    # Show the plot we reorient the labels for each column and row to make them easier to read.
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title("Correlation for paragraphs")
    plt.savefig(save_dir + folder_name + '/' + 'correlation_plot_for_paragrsphs.png', bbox_inches='tight')


def corr_disjoint_window(book_id):
    plt.clf()
    table = book_id + '_window'
    folder_name = create_folder(book_id)
    X = []
    i = 1
    while i < db[table].find().count():
        # print ('Begin: ' + str(i))
        X.append(db[table].find_one({'_id': i})['_id'])
        i = db[table].find_one({'_id': i})['end_id'] + 1
        # print ('End: ' + str(i))

    dialogues_number, sentences_number, window_length_in_words, person_verbs_num = [], [], [], []
    person_pronouns_num, avr_word_length, avr_sentence_length = [], [], []
    #     similarity_with_previous_window_all, similarity_with_previous_window_main = [], []


    for i in range(0, len(X)):
        item = db[table].find_one({'_id': X[i]})
        dialogues_number.append(item['dialoques_number'])
        sentences_number.append(item['sentences_number'])
        window_length_in_words.append(item['window_length_in_words'])
        person_verbs_num.append(item['person_verbs_num'])
        person_pronouns_num.append(item['person_pronouns_num'])
        avr_word_length.append(item['average_word_length'])
        avr_sentence_length.append(item['avr_sentence_length'])
    # similarity_with_previous_window_all.append(item['similarity_with_previous_window_by_main_words'])
    #         similarity_with_previous_window_main.append(item['similarity_with_previous_window_by_all_words'])

    data = pd.DataFrame()
    data['dialogues_number'] = np.array(dialogues_number)
    data['sentences_number'] = np.array(sentences_number)
    data['window_length_in_words'] = np.array(window_length_in_words)
    data['person_verbs_num'] = np.array(person_verbs_num)
    data['person_pronouns_num'] = np.array(person_pronouns_num)
    data['avr_word_length'] = np.array(avr_word_length)
    data['avr_sentence_length'] = np.array(avr_sentence_length)
    data['new_words_count'] = np.array(db[book_id + '_window'].find_one({"_id": "new_dictionary"})['new_words_count'])
    #     data['similarity_with_previous_window_all'] = np.array(similarity_with_previous_window_all)
    #     data['similarity_with_previous_window_main'] = np.array(similarity_with_previous_window_main)

    corr_df = data.corr(method='pearson')
    print("--------------- BOOK ---------------")
    print(db.books.find_one({'_id': book_id})['title'])
    print("--------------- DISJOINT WINDOW MODE CORRELATIONS ---------------")
    # print(corr_df.head(len(data)))
    # print("--------------- CREATE A HEATMAP ---------------")
    # Create a mask to display only the lower triangle of the matrix (since it's mirrored around its
    # top-left to bottom-right diagonal).
    mask = np.zeros_like(corr_df)
    mask[np.triu_indices_from(mask)] = True
    # Create the heatmap using seaborn library.
    # List if colormaps (parameter 'cmap') is available here: http://matplotlib.org/examples/color/colormaps_reference.html
    seaborn.heatmap(corr_df, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0, mask=mask, linewidths=2.5)

    # Show the plot we reorient the labels for each column and row to make them easier to read.
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title("Correlation for windows(disjoint)")
    plt.savefig(save_dir + folder_name + '/' + 'correlation_plot_for_windows(disjoint).png', bbox_inches='tight')



books = db.books.find()
for book in books:
    params_plot_for_paragraphs(book['_id'], ['person_pronouns_portion', 'person_verbs_portion', 'sentiment_words_portion', 'labels_portion'],
                               plot_name = '1')
    params_plot_for_paragraphs(book['_id'], ['avr_sentence_length', 'avr_word_length', 'words_with_labels'],
                               plot_name = '2')

    params_plot_for_pages(book['_id'], ['sentences_number', 'sum_sentiment', 'words_with_labels', 'avr_sentence_length', 'new_words_count'],
                               plot_name = '1', smooth_N=10)
    params_plot_for_pages(book['_id'], ['avr_person_pronouns_part', 'avr_person_verbs_part', 'sentiment_words_portion', 'labels_portion', \
                                        'avr_dialogues_part'],
                               plot_name = '2', smooth_N=10)

