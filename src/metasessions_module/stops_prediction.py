import argparse
import numpy as np

from src.metasessions_module.text_features_handler import TextFeaturesHandler
from src.metasessions_module.users_clustering import load_clusters


def identify_swipe_places(book_id, model_name, target_cluster_index,
                          swipe_percentile=50,
                          swipe_ratio=0.5):
    clusters = load_clusters(book_id, model_name)
    target_cluster = clusters[target_cluster_index][0]
    print(target_cluster.shape)
    swipe_speeds = []
    for speeds in target_cluster:
        swipe_speeds.append(np.percentile(speeds, swipe_percentile))
    result = np.zeros(target_cluster.shape[1])
    for position in range(result.shape[0]):
        swipes_count = np.sum(np.array(target_cluster[:, position] > swipe_speeds, dtype=np.int64))
        if swipes_count >= swipe_ratio * target_cluster.shape[0]:
            result[position] = 1
    return result


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train_book_id', help='Train on book ID', type=str, metavar='ID', required=True)
    # parser.add_argument('--test_book_id', help='Test on book ID', type=str, metavar='ID', required=True)
    # parser.add_argument('--batches_amount', help='Split train book to N parts', type=int, metavar='N', required=True)
    # parser.add_argument('--target_cluster_index', help='Index of cluster to identify swipes', type=int,
    #                     metavar='N', required=True)
    # parser.add_argument('--test_model_name', help='Use model M to identify test users clusters',
    #                     type=str, metavar='M', required=True)
    # args = parser.parse_args()
    #
    # train_book_id = args.train_book_id
    # test_book_id = args.test_book_id
    # batches_amount = args.batches_amount
    #
    # train_handler = TextFeaturesHandler(train_book_id, batches_amount=batches_amount)
    # X_train = train_handler.get_features()
    # test_handler = TextFeaturesHandler(test_book_id, batches_amount=batches_amount)
    # X_test = test_handler.get_features()
    print(identify_swipe_places('210901', '2_100_agglomerative.pkl', 1))
