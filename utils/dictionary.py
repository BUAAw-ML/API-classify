import os
import numpy as np
import pickle as pk


class GloveDictionary(object):

    def __init__(self, word_to_index, index_to_word, word_to_vec_map):
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.word_to_vec_map = word_to_vec_map

    def word_sequence_to_index(self, words):
        index = []
        for w in words:
            if w in self.word_to_index:
                index.append(self.word_to_index[w])
        return index

    def build_emb_matrix(self):
        vocab_len = len(self.word_to_index) + 1
        emb_dim = self.word_to_vec_map[self.index_to_word[1]].shape[0]
        
        emb_matrix = np.zeros((vocab_len, emb_dim))
        
        for word, index in self.word_to_index.items():
            embedding_vector = self.word_to_vec_map.get(word)
            if embedding_vector is not None:
                emb_matrix[index, :] = embedding_vector

        return emb_matrix
    
    @classmethod
    def read_glove_vecs(cls, glove_file):
        if (os.path.exists("data/word_to_index.pkl")
            and os.path.exists("data/index_to_word.pkl")
            and os.path.exists("data/word_to_vec_map.pkl")):
            word_to_index = pk.load(open("data/word_to_index.pkl", "rb"))
            index_to_word = pk.load(open("data/index_to_word.pkl", "rb"))
            word_to_vec_map = pk.load(open("data/word_to_vec_map.pkl", "rb"))
            return word_to_index, index_to_word, word_to_vec_map
            
        with open(glove_file, 'r', encoding="utf-8") as f:
            words = set()
            word_to_vec_map = {}
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
            i = 1
            word_to_index = {}
            index_to_word = {}
            for w in sorted(words):
                word_to_index[w] = i
                index_to_word[i] = w
                i = i + 1
        pk.dump(word_to_index, open("data/word_to_index.pkl", "wb"))
        pk.dump(index_to_word, open("data/index_to_word.pkl", "wb"))
        pk.dump(word_to_vec_map, open("data/word_to_vec_map.pkl", "wb"))
        return word_to_index, index_to_word, word_to_vec_map