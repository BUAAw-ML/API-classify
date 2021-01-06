import pickle as pk

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from keras_utils.dictionary import GloveDictionary


def convert_to_one_hot(Y, C):
    one_hot = np.eye(C, dtype=np.int32)
    one_hot_Y = []
    for tags in Y:
        one_hot_Y.append(one_hot[tags].sum(axis=0))
    return np.stack(one_hot_Y, axis=0)


def load_data(data_file, glove_file, index_to_tag_file, tag_to_index_file, max_len=None):
    data = pk.load(open(data_file, "rb"))
    index_to_tag = pk.load(open(index_to_tag_file, "rb"))
    tag_to_index = pk.load(open(tag_to_index_file, "rb"))

    word_to_index, index_to_word, word_to_vec_map = GloveDictionary.read_glove_vecs(glove_file)
    glove_dict = GloveDictionary(word_to_index, index_to_word, word_to_vec_map)

    X = []
    Y = []
    
    _max_len = 0
    for d in data:
        words = text_to_word_sequence(d['descr'])

        x = []
        y = []
        for w in words:
            if w in word_to_index:
                x.append(word_to_index[w])
        for t in d['tags']:
            y.append(tag_to_index[t])
        _max_len = max(_max_len, len(x))
        X.append(x)
        Y.append(y)
    max_len = max_len if max_len is not None else _max_len
    X = pad_sequences(X, maxlen=max_len, padding="post")
    Y = convert_to_one_hot(Y, len(tag_to_index))
    return X, Y
