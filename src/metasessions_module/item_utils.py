import logging


from metasessions_module.utils import connect_to_mongo_database


def save_items(document_id):
    db_work = connect_to_mongo_database('bookmate_work')
    db = connect_to_mongo_database('bookmate')
    collection_name = '{}_items'.format(document_id)
    if collection_name in db_work.collection_names():
        return
    logging.info('Processing items for {}'.format(document_id))
    items = list(db['items'].find({'document_id': document_id}))
    items.sort(key=lambda item: item['position'])
    total_size = 0
    for item in items:
        total_size += item['media_file_size']
    current_size = 0
    for item in items:
        item['_from'] = 100.0 * current_size / total_size
        current_size += item['media_file_size']
        item['_to'] = 100.0 * current_size / total_size
    db_work[collection_name].insert_many(items)
    logging.info('Items for document {} processed'.format(document_id))


def get_items(document_id):
    save_items(document_id)
    db_work = connect_to_mongo_database('bookmate_work')
    items = list(db_work['{}_items'.format(document_id)].find({}))
    items.sort(key=lambda item: item['position'])
    return items
