import json

sentiment_dictionary_txt_file = '../../resources/words_all_full_rating.csv'
sentiment_dictionary_json_file = '../../resources/sentiment_dictionary_1.json'

with open(sentiment_dictionary_txt_file, encoding='utf-8') as f:
    content = f.readlines()

sentiment_dictionary = dict()
for line in content[1:]:
    split_line = line.split(';')
    word = split_line[0][1:-1]
    sentiment = split_line[3][1:-1]
    if int(sentiment) != 0:
        sentiment_dictionary[word] = int(sentiment)

with open(sentiment_dictionary_json_file, 'w', encoding='utf-8') as f:
    json.dump(sentiment_dictionary, f, ensure_ascii=False)