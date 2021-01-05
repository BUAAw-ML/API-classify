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
    one_hot_Y =  np.stack(one_hot_Y, axis=0)
    one_hot_Y[one_hot_Y > 1] = 1
    return one_hot_Y


def convert(dataset, glove_file):
    word_to_index, index_to_word, word_to_vec_map = GloveDictionary.read_glove_vecs(glove_file)
    glove_dict = GloveDictionary(word_to_index, index_to_word, word_to_vec_map)
    tag_to_index = dataset.tag2id

    X = []
    Y = []

    test_X = []
    test_Y = []
    
    _max_len = 0
    for d in dataset.train_data:
        words = text_to_word_sequence(d['dscp'].strip())

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
    
    for d in dataset.test_data:
        words = text_to_word_sequence(d['dscp'].strip())

        x = []
        y = []
        for w in words:
            if w in word_to_index:
                x.append(word_to_index[w])
        for t in d['tags']:
            y.append(tag_to_index[t])
        _max_len = max(_max_len, len(x))
        test_X.append(x)
        test_Y.append(y)
    max_len = _max_len
    X = pad_sequences(X, maxlen=max_len, padding="post")
    Y = convert_to_one_hot(Y, len(tag_to_index))
    test_X = pad_sequences(test_X, maxlen=max_len, padding="post")
    test_Y = convert_to_one_hot(test_Y, len(tag_to_index))
    return X, Y, test_X, test_Y
