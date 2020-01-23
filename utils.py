# Generic/Built-in
import os, re, pickle

# Other Libs
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Own modules


__author__ = "Ming Gao"
__credits__ = ["Ming Gao"]
__license__ = ""
__version__ = "0.1.0"
__maintainer__ = "Ming Gao"
__email__ = "ming_gao@outlook.com"
__status__ = "Dev"

'''
utility functions
''' 

def clean_data(text):
    text = re.sub('<.*>', ' ', text.lower())
    words = text.split()
    tokens = [re.sub('[^a-zA-Z]', '', w) for w in words]
    return ' '.join(t for t in tokens)


def import_data(directory):
    reviews = []
    for filename in os.scandir(directory):
        with open(filename, 'r') as f:
            raw = f.read()
            clean = clean_data(raw)
            reviews.append(clean)
    return reviews


def encode_review(review, label):
    review_token = []
    vocab_encoder = tfds.features.text.SubwordTextEncoder.load_from_file('vocab')
    # review_clean = clean_data(review.numpy().decode('utf-8'))
    review_token.extend(vocab_encoder.encode(review.numpy().decode('utf-8')))
    return review_token, label


def map_fn_encode_data(review, label):
    return tf.py_function(encode_review, inp=[review, label], Tout=(tf.int64, tf.int64))


def prepare_tf_dataset(train_test):
    neg = import_data('./'+train_test+'/neg/')
    neg_dataset = tf.data.Dataset.from_tensor_slices((neg, np.zeros(len(neg), dtype=int)))
    pos = import_data('./'+train_test+'/pos/')
    pos_dataset = tf.data.Dataset.from_tensor_slices((pos, np.ones(len(pos), dtype=int)))

    dataset = neg_dataset.concatenate(pos_dataset).shuffle(buffer_size=len(neg)+len(pos), seed=23, reshuffle_each_iteration=False)
    dataset = dataset.map(map_fn_encode_data)
    return dataset


if __name__ == '__main__':
    train = prepare_tf_dataset('train')
    test = prepare_tf_dataset('test')