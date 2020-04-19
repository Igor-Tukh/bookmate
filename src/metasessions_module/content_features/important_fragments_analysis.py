import os

from src.metasessions_module.content_features.feautres_builder import TextFeaturesBuilder
from tqdm import tqdm


def get_important_fragments_path():
    return os.path.join('resources', 'important_fragments')


def process_important_texts(full_feature_name, important_fragments):
    path_to_save = os.path.join(get_important_fragments_path(), full_feature_name)
    os.makedirs(os.path.join(path_to_save, 'max'), exist_ok=True)
    os.makedirs(os.path.join(path_to_save, 'min'), exist_ok=True)
    for text, text_type, text_value, text_ind in important_fragments:
        with open(os.path.join(path_to_save, text_type, f'{text_ind}_{text_value}.txt'), 'w') as text_file:
            text_file.write(text)


if __name__ == '__main__':
    builder = TextFeaturesBuilder(os.path.join('resources', 'texts'))

    for feature_name in tqdm(builder.FEATURES_NAMES):
        important_texts = builder.get_important_fragments(feature_name, window_size=10)
        process_important_texts(feature_name, important_texts)
