"""
This module contains the TensorFlow model itself, as well as the logic for training and testing
it in strongly supervised and weakly supervised frameworks.
"""
import sys

sys.path.append('../')
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import pickle
import numpy as np
import os
import sys
import time
import definitions
from definitions import LOGICAL_TOKENS_MAPPING_PATH
from seq2seqModel.utils import *
# from seq2seqModel.partial_program import *
from data_manager import CNLVRDataSet, DataSetForSupervised, DataSet, load_functions
# from seq2seqModel.beam_search import *
from seq2seqModel.hyper_params import *
from general_utils import increment_count, union_dicts
# from seq2seqModel.beam_classification import *
# from seq2seqModel.similarity_model import run_similarity_model
import matplotlib.pyplot as plt
tf.set_random_seed(1)
np.random.seed(1)

def load_meta_data():
    # load word embeddings
    embeddings_file = open(WORD_EMBEDDINGS_PATH, 'rb')
    embeddings_dict = pickle.load(embeddings_file)
    embeddings_file.close()
    assert WORD_EMB_SIZE == np.size(embeddings_dict['blue'])
    vocab_size = len(embeddings_dict)
    vocab_list = [k for k in sorted(embeddings_dict.keys())]
    one_hot_dict = {w: one_hot(vocab_size, i) for i, w in enumerate(vocab_list)}
    embeddings_matrix = np.stack([embeddings_dict[k] for k in vocab_list])

    # load logical tokens inventory
    logical_tokens_mapping = load_functions(LOGICAL_TOKENS_MAPPING_PATH)
    logical_tokens = pickle.load(open(LOGICAL_TOKENS_LIST, 'rb'))
    assert set(logical_tokens) == set(logical_tokens_mapping.keys())

    for var in "xyzwuv":
        logical_tokens.extend([var, 'lambda_{}_:'.format(var)])
    logical_tokens.extend(['<s>', '<EOS>'])
    logical_tokens_ids = {lt: i for i, lt in enumerate(logical_tokens)}
    return logical_tokens_ids, logical_tokens_mapping, embeddings_dict, one_hot_dict, embeddings_matrix


logical_tokens_ids, logical_tokens_mapping, embeddings_dict, one_hot_dict, embeddings_matrix = load_meta_data()

print(logical_tokens_ids, '\n', logical_tokens_mapping, '\n')

if definitions.MANUAL_REPLACEMENTS:
    words_to_tokens = pickle.load(
        open(os.path.join(definitions.DATA_DIR, 'logical forms', 'words_to_tokens'), 'rb'))
else:
    words_to_tokens = pickle.load(
        open(os.path.join(definitions.DATA_DIR, 'logical forms', 'new_words_to_tokens'), 'rb'))

n_logical_tokens = len(logical_tokens_ids)

print(words_to_tokens)
print(n_logical_tokens)

if USE_BOW_HISTORY:
    HISTORY_EMB_SIZE += n_logical_tokens