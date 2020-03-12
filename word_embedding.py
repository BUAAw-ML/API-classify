# import re
# import os
# import json
# import numpy as np
# import pickle as pkl
#
# """
# for extracting word embedding yourself, please download pretrained model from one of the following links.
# """
#
# url = {'glove': 'http://nlp.stanford.edu/data/glove.6B.zip',
#        'google': 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing',
#        'fasttext': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip'}
#
# data_dir = '../data/'
# feat_len = 300
#
#
# def embed_text_file(text_file, word_vectors, get_vector, save_file):
#     with open(text_file) as fp:
#         text_list = json.load(fp)
#
#     all_feats = []
#
#     has = 0
#     cnt_missed = 0
#     missed_list = []
#     for i in range(len(text_list)):
#         class_name = text_list[i].lower()
#         if i % 500 == 0:
#             print('%d / %d : %s' % (i, len(text_list), class_name))
#         feat = np.zeros(feat_len)
#
#         options = class_name.split(',')
#         cnt_word = 0
#         for j in range(len(options)):
#             now_feat = get_embedding(options[j].strip(), word_vectors, get_vector)
#             if np.abs(now_feat.sum()) > 0:
#                 cnt_word += 1
#                 feat += now_feat
#         if cnt_word > 0:
#             feat = feat / cnt_word
#
#         if np.abs(feat.sum()) == 0:
#             print('cannot find word ' + class_name)
#             cnt_missed = cnt_missed + 1
#             missed_list.append(class_name)
#         else:
#             has += 1
#             feat = feat / (np.linalg.norm(feat) + 1e-6)
#
#         all_feats.append(feat)
#
#     all_feats = np.array(all_feats)
#
#     for each in missed_list:
#         print(each)
#     print('does not have semantic embedding: ', cnt_missed, 'has: ', has)
#
#     if not os.path.exists(os.path.dirname(save_file)):
#         os.makedirs(os.path.dirname(save_file))
#         print('## Make Directory: %s' % save_file)
#     with open(save_file, 'wb') as fp:
#         pkl.dump(all_feats, fp)
#     print('save to : %s' % save_file)
#
# def get_glove_dict(txt_dir):
#     print('load glove word embedding')
#     txt_file = os.path.join(txt_dir, 'glove.6B.300d.txt')
#     word_dict = {}
#     feat = np.zeros(feat_len)
#     with open(txt_file) as fp:
#         for line in fp:
#             words = line.split()
#             assert len(words) - 1 == feat_len
#             for i in range(feat_len):
#                 feat[i] = float(words[i+1])
#             feat = np.array(feat)
#             word_dict[words[0]] = feat
#     print('loaded to dict!')
#     return word_dict
#
#
# def glove_google(word_vectors, word):
#     return word_vectors[word]
#
#
# def obtain_word_embedding(wv='glove', path='data/glove'):
#
#     text_file = os.path.join('data', 'invdict_wordntext.json')
#
#
#     if not os.path.exists(save_file):
#         print('obtain semantic word embedding', save_file)
#         embed_text_file(text_file, word_vectors, get_vector, save_file)
#     else:
#         print('Embedding existed :', save_file, 'Skip!!!')
#
#     model_path = path
#     if wv == 'glove':
#         save_file = os.path.join(data_dir, 'word_embedding_model', 'glove_word2vec_wordnet.pkl')
#         if not os.path.exists(save_file):
#             word_vectors = get_glove_dict(model_path)
#             get_vector = glove_google
#     else:
#         raise NotImplementedError
#
