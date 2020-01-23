# Generic/Built-in
import json

# Other Libs
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy
import requests

# Own modules
from utils import clean_data


__author__ = "Ming Gao"
__credits__ = ["Ming Gao"]
__license__ = ""
__version__ = "0.1.0"
__maintainer__ = "Ming Gao"
__email__ = "ming_gao@outlook.com"
__status__ = "Dev"

'''
Apply the trained GRU model
'''

MODEL_DIR = "./saved_model/v1"

def prepare_input(raw_input):
    clean_input = clean_data(raw_input)
    vocab_encoder = tfds.features.text.SubwordTextEncoder.load_from_file('vocab')
    return vocab_encoder.encode(clean_input)


def send_request(raw_input):
    encoded_input = prepare_input(raw_input)
    data = json.dumps({"signature_name": "serving_default",
                       "instances": [encoded_input]})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/testing:predict',
                                data=data, headers=headers)

    return json_response.text


if __name__ == '__main__':
    send_request(raw_input="This is a good movie")


