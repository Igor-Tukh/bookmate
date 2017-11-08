import pickle
import bson
import csv


__index_names = ['books', 'subjects', 'users']

__source_folder = 'meta'
__dump_folder = 'dumps'

genres = [
    'Love & Romance', # 0
    'Mystery & Detective', # 1
    'Science Fiction & Fantasy', # 2
    'Crime & Thrillers', # 3
    'Biography', # 4
    'Kids', # 5
    'Psychology', # 6
    'Business', # 7
    'Personal Growth', # 8
    'Art & Culture', # 9
    'Technology & Science', # 10
    'Politics & Society', # 11
    'History', # 12
    'Religion', # 13
    'Sports & Health', # 14
    'Classics', # 15
    'Cooking & Food',  # 16
    'Modern Fiction', # 17
    'Others', # 18
    'Young Adult', # 19
    'Read for free!', # 20
    'Poetry' # 21
]

genre_is_fiction = [
    True, # 0
    True, # 1
    True, # 2
    True, # 3
    False, # 4
    None, # 5
    False, # 6
    False, # 7
    False, # 8
    False, # 9
    False, # 10
    False, # 11
    False, # 12
    False, # 13
    False, # 14
    True, # 15
    False, # 16
    True, # 17
    None, # 18
    None, # 19
    None, # 20
    True # 21
]

__genre_mapping = {
    61:0,
    62:1,
    63:2,
    64:3,
    65:4,
    66:5,
    67:6,
    68:7,
    69:8,
    70:9,
    71:10,
    72:11,
    73:12,
    74:13,
    75:14,
    76:15,
    77:16,
    78:17,
    79:18,
    81:19,
    101:20,
    102:20,
    103:19,
    104:0,
    105:1,
    106:2,
    107:3,
    108:4,
    109:5,
    110:6,
    111:7,
    112:8,
    113:9,
    114:10,
    115:11,
    116:12,
    117:13,
    118:14,
    119:15,
    120:16,
    121:17,
    122:18,
    123:20,
    124:0,
    125:1,
    126:2,
    127:3,
    128:4,
    129:5,
    130:6,
    131:7,
    132:8,
    133:9,
    134:10,
    135:11,
    136:12,
    137:13,
    138:14,
    139:15,
    140:16,
    141:17,
    142:18,
    144:17, # ????
    145:12,
    146:21,
    147:9,
    148:0,
    175:7,
    176:8,
    177:10,
    178:11,
    179:12,
    180:13,
    181:15,
    182:16,
    183:17,
    184:18,
    185:19,
    186:20,
    187:0,
    188:4,
    189:1,
    190:2,
    191:10,
    192:5,
    193:3,
    194:6,
    195:9,
    196:21,
    197:17,
    224:20,
    225:3,
    226:19,
    227:0,
    228:8,
    229:10,
    230:14,
    231:6,
    232:5,
    233:17,
    234:16,
    235:4,
    236:3,
    237:2,
    238:9,
    239:13,
    240:18,
    241:12,
    242:11,
    243:15,
    244:7,
    245:17, # ????
    246:18, # ????
    247:18, # ????
    248:9   # ?????
}

def __get_dump_fname(index_name):
    return __dump_folder + '/' + index_name + '_index.pk'

def __build_index(index_name):
    index = {}
    with open(__source_folder + '/' + index_name + '.csv', 'r') as f:
        rows = csv.DictReader(f, delimiter=',', quotechar='"')
        for row in rows:
            row_id = int(row['id'])
            row.pop('id')
            index[row_id] = row
    with open(__get_dump_fname(index_name), 'wb') as f:
        pickle.dump(index, f)
    print('{0} built'.format(index_name))

def load():
    result = []
    for index_name in __index_names:
        with open(__get_dump_fname(index_name), 'rb') as f:
            result.append(pickle.load(f))
        print('{0} index loaded'.format(index_name))
    return result[0], result[1], result[2]

def load_books():
    with open(__get_dump_fname(__index_names[0]), 'rb') as f:
        result = pickle.load(f, encoding='utf-8')
    print('books index loaded')
    return result

def get_genres(book_id, books_index):
    if book_id not in books_index or 'topics' not in books_index[book_id]:
        return []
    topics_string = books_index[book_id]['topics']
    if topics_string == 'NULL':
        return []
    return list({__genre_mapping[int(x)] for x in topics_string.split(' ')})

def is_fiction(book_id, books_index):
    if book_id is None:
        return None
    genres = get_genres(book_id, books_index)
    if any((genre_is_fiction[x] is not None for x in genres)) == 0:
        return None
    return any((genre_is_fiction[x] for x in genres))

def get_field_value(field, value_transform, index, object_id):
    if object_id not in index or field not in index[object_id] or \
            index[object_id][field] == '' or index[object_id][field] == 'NULL':
        return None
    return value_transform(index[object_id][field])

def load_lcid_bid():
    result = {}
    with open('meta/lcid_bid.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            result[row[0]] = row[1]
    return result

if __name__ == '__main__':
    for index_name in __index_names:
        __build_index(index_name)
