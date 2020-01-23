# Generic/Built-in
import os, re

# Other Libs
import tensorflow_datasets as tfds

# Own modules
from utils import import_data, clean_data

__author__ = "Ming Gao"
__credits__ = ["Ming Gao"]
__license__ = ""
__version__ = "0.1.0"
__maintainer__ = "Ming Gao"
__email__ = "ming_gao@outlook.com"
__status__ = "Dev"

'''
build subwords vocabulary list
'''


def build_subwords_vocab(target_vocab_size=10000):
    train_neg = import_data('./train/neg')
    train_pos = import_data('./train/pos')
    train_raw = train_neg + train_pos
    train_clean = [clean_data(t) for t in train_raw]

    vocab_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(train_clean, target_vocab_size)
    vocab_encoder.save_to_file('vocab')

    print(vocab_encoder.vocab_size)

if __name__ == '__main__':
    build_subwords_vocab()