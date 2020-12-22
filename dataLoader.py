import csv
import copy
import os
import sys
from random import shuffle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from word_embedding import *

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
token_table = {'ecommerce': 'electronic commerce'}


def load_data(data_config, data_path=None, data_type='allData', use_previousData=False):
    cache_file_head = data_path.split("/")[-1]

    if use_previousData:

        print("load dataset from cache")
        dataset = dataEngine.from_dict(torch.load(os.path.join('cache', cache_file_head + '.dataset')))
        encoded_tag, tag_mask = torch.load(os.path.join('cache', cache_file_head + '.encoded_tag')), \
                                torch.load(os.path.join('cache', cache_file_head + '.tag_mask'))

    else:
        print("build dataset")
        if not os.path.exists('cache'):
            os.makedirs('cache')

        dataset = dataEngine(data_config=data_config)

        if data_type == 'All':

            data = dataset.load_programWeb_AAPD(data_path)

            data = np.array(data)
            ind = np.random.RandomState(seed=10).permutation(len(data))

            split = int(len(data) * data_config['data_split'])
            split2 = int(len(data) * 0.7)
            split3 = int(len(data) * 1)

            dataset.train_data = data[ind[:split]].tolist()
            dataset.unlabeled_train_data = data[ind[split:split*4]].tolist()
            dataset.test_data = data[ind[split2:split3]].tolist()

        elif data_type == 'TrainTest_ganBert':

            file = os.path.join(data_path, 'labeled.tsv')
            dataset.train_data = dataset.load_ganBert(file)
            file = os.path.join(data_path, 'unlabeled.tsv')
            dataset.unlabeled_train_data = dataset.load_ganBert(file)
            file = os.path.join(data_path, 'test.tsv')
            dataset.test_data = dataset.load_ganBert(file)

        elif data_type == 'TrainTest_agNews':

            file = os.path.join(data_path, 'train.csv')
            data = dataset.load_agNews(file)

            data = np.array(data)
            ind = np.random.RandomState(seed=10).permutation(len(data))
            split = int(len(data) * data_config['data_split'])
            split2 = int(len(data) * 0.3)

            dataset.train_data = data[ind[:split]].tolist()
            dataset.unlabeled_train_data = data[ind[split:split2]].tolist()

            file = os.path.join(data_path, 'test.csv')
            dataset.test_data = dataset.load_agNews(file)

        elif data_type == 'TrainTestTextTag':

            file1 = os.path.join(data_path, 'train_texts.txt')
            file2 = os.path.join(data_path, 'train_labels.txt')
            dataset.filter_tags(file2)
            data = dataset.load_EurLex_RCV2(file1, file2)

            data = np.array(data)
            ind = np.random.RandomState(seed=10).permutation(len(data))
            split = int(len(data) * data_config['data_split'])
            split2 = int(len(data) * 0.5)

            dataset.train_data = data[ind[:split]].tolist()
            dataset.unlabeled_train_data = data[ind[split:]].tolist()

            file1 = os.path.join(data_path, 'valid+test_texts.txt')
            file2 = os.path.join(data_path, 'valid+test_labels.txt')
            dataset.test_data = dataset.load_EurLex_RCV2(file1, file2)

        torch.save(dataset.to_dict(), os.path.join('cache', cache_file_head + '.dataset'))
        encoded_tag, tag_mask = dataset.encode_tag()
        torch.save(encoded_tag, os.path.join('cache', cache_file_head + '.encoded_tag'))
        torch.save(tag_mask, os.path.join('cache', cache_file_head + '.tag_mask'))

    return dataset, encoded_tag, tag_mask


class dataEngine(Dataset):
    def __init__(self, train_data=None, unlabeled_train_data=None, test_data=None,
                    tag2id={}, id2tag={}, co_occur_mat=None, tfidf_dict=None, data_config={}):
        self.train_data = train_data
        self.unlabeled_train_data = unlabeled_train_data
        self.test_data = test_data

        self.tag2id = tag2id
        self.id2tag = id2tag

        self.use_tags = set()

        self.co_occur_mat = co_occur_mat
        self.tfidf_dict = tfidf_dict

        self.data_config = data_config


    @classmethod
    def from_dict(cls, data_dict):
        return dataEngine(data_dict.get('train_data'),
                       data_dict.get('unlabeled_train_data'),
                       data_dict.get('test_data'),
                       data_dict.get('tag2id'),
                       data_dict.get('id2tag'),
                       data_dict.get('co_occur_mat'),
                       data_dict.get('tfidf_dict'))

    def to_dict(self):
        data_dict = {
            'train_data': self.train_data,
            'unlabeled_train_data': self.unlabeled_train_data,
            'test_data': self.test_data,
            'tag2id': self.tag2id,
            'id2tag': self.id2tag,
            'co_occur_mat': self.co_occur_mat,
            'tfidf_dict': self.tfidf_dict
        }
        return data_dict

    def stat_cooccurence(cls, data, tags_num):

        co_occur_mat = torch.zeros(size=(tags_num, tags_num))
        for i in range(len(data)):
            tag_ids = data[i]['tag_ids']
            for t1 in range(len(tag_ids)):
                for t2 in range(len(tag_ids)):
                    #if tag_ids[t1] != tag_ids[t2]:
                    co_occur_mat[tag_ids[t1], tag_ids[t2]] += 1

        return co_occur_mat

    #directly input relations between tags
    def similar_net(cls, csvfile, tag2id):

        tags_num = len(tag2id)
        co_occur_mat = torch.zeros(size=(tags_num, tags_num))
        i = 0
        with open(csvfile, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for row in reader:
                if len(row) != 3:
                    continue
                tag1, similar, tag2 = row

                if tag1 not in tag2id or tag2 not in tag2id:
                    i += 1
                    continue

                tag1 = tag1.strip()
                tag2 = tag2.strip()

                co_occur_mat[tag2id[tag1], tag2id[tag2]] += float(similar)

        print(i)
        return co_occur_mat

    def get_tags_num(self):
        return len(self.tag2id)

    def encode_tag(self):
        tag_ids = []
        tag_token_num = []
        for i in range(self.get_tags_num()):
            tag = self.id2tag[i]
            tokens = tokenizer.tokenize(tag)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            tag_ids.append(token_ids)
            tag_token_num.append(len(tokens))
        max_num = max(tag_token_num)
        padded_tag_ids = torch.zeros((self.get_tags_num(), max_num), dtype=torch.long)
        mask = torch.zeros((self.get_tags_num(), max_num))
        for i in range(self.get_tags_num()):
            mask[i, :len(tag_ids[i])] = 1.
            padded_tag_ids[i, :len(tag_ids[i])] = torch.tensor(tag_ids[i])
        return padded_tag_ids, mask

    #directly obtain pretrained-embeddings for tags
    def obtain_tag_embedding(self, wv='glove', model_path='data'):

        if wv == 'glove':
            save_file = os.path.join('data', 'word_embedding_model', 'glove_word2vec_wordnet.pkl')
            if not os.path.exists(save_file):
                word_vectors = get_glove_dict(model_path)
        else:
            raise NotImplementedError

        if not os.path.exists(save_file):
            tag_list = []
            for i in range(self.get_tags_num()):
                tag = self.id2tag[i]
                tag_list.append(tag)
            print(tag_list)

            print('obtain semantic word embedding', save_file)
            embed_text_file(tag_list, word_vectors, save_file)
        else:
            print('Embedding existed :', save_file, 'Skip!!!')

        return save_file

    def collate_fn(self, batch):
        # construct input
        inputs = [e['dscp_ids'] for e in batch]  #e['title_ids'] +

        lengths = np.array([len(e) for e in inputs])
        max_len = np.max(lengths)  #_to_max_length=True , truncation=True
        inputs = [tokenizer.prepare_for_model(e, max_length=max_len+2, pad_to_max_length=True) for e in inputs]

        ids = torch.LongTensor([e['input_ids'] for e in inputs])
        token_type_ids = torch.LongTensor([e['token_type_ids'] for e in inputs])
        attention_mask = torch.FloatTensor([e['attention_mask'] for e in inputs])
        # construct tag
        tags = torch.zeros(size=(len(batch), self.get_tags_num()))
        for i in range(len(batch)):
            tags[i, batch[i]['tag_ids']] = 1.

        dscp = [e['dscp'] for e in batch]

        return (ids, token_type_ids, attention_mask), tags, dscp

    @classmethod
    def get_tfidf_dict(cls, document):
        tfidf_dict = {}
        tfidf_model = TfidfVectorizer(sublinear_tf=True,
                                        strip_accents='unicode',
                                        analyzer='word',
                                        token_pattern=r'\w{1,}',
                                        stop_words='english',
                                        ngram_range=(1, 1),
                                        max_features=10000).fit(document)
        for item in tfidf_model.vocabulary_:
            tfidf_dict[item] = tfidf_model.idf_[tfidf_model.vocabulary_[item]]

        return tfidf_dict

    def load_programWeb_AAPD(self, f):
        data = []

        document = []
        tag_occurance = {}
        # csv.field_size_limit(sys.maxsize)
        with open(f, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:

                if len(row) != 4:
                    continue
                _, _, _, tag = row

                tag = tag.strip().split('###')
                tag = [t for t in tag if t != '']

                for t in tag:
                    if t not in tag_occurance:
                        tag_occurance[t] = 1
                    tag_occurance[t] += 1

        # ignored_tags = set(['Tools','Applications','Other', 'API', 'Software-as-a-Service','Platform-as-a-Service',
        # 'Data-as-a-Service'])  #
        for tag in tag_occurance:
            if self.data_config['min_tagFrequence'] <= tag_occurance[tag] <= self.data_config['max_tagFrequence']:
                self.use_tags.add(tag)

        print('Total number of tags: {}'.format(len(tag_occurance)))
        print(sorted(tag_occurance.items(), key=lambda x: x[1], reverse=True))

        with open(f, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:

                if len(row) != 4:
                    continue
                id, title, dscp, tag = row

                title_tokens = tokenizer.tokenize(title.strip())
                dscp_tokens = title_tokens + tokenizer.tokenize(dscp.strip())

                if len(dscp_tokens) > 510:
                    if self.data_config['overlength_handle'] == 'truncation':
                        dscp_tokens = dscp_tokens[:510]
                    else:
                        continue

                dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)

                tag = tag.strip().split('###')
                tag = [t for t in tag if t != '']

                if self.use_tags is not None:
                    tag = [t for t in tag if t in self.use_tags]

                # if len(set(tag)) < 2:
                #     continue

                if len(tag) == 0:
                    continue

                for t in tag:
                    if t not in self.tag2id:
                        tag_id = len(self.tag2id)
                        self.tag2id[t] = tag_id
                        self.id2tag[tag_id] = t

                tag_ids = [self.tag2id[t] for t in tag]

                data.append({
                    'id': int(id),
                    'dscp_ids': dscp_ids,
                    'dscp_tokens': dscp_tokens,
                    'tag_ids': tag_ids,
                    'dscp': dscp
                })

        print("The number of tags for training: {}".format(len(self.tag2id)))
        # os.makedirs('cache', exist_ok=True)

        return data

    def load_agNews(self, file):
        data = []
        document = []

        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            # next(reader)
            for row in reader:

                if len(row) != 3:
                    continue
                tag, title, dscp = row

                title_tokens = tokenizer.tokenize(title.strip())
                dscp_tokens = title_tokens + tokenizer.tokenize(dscp.strip())

                if len(dscp_tokens) > 510:
                    if self.data_config['overlength_handle'] == 'truncation':
                        dscp_tokens = dscp_tokens[:510]
                    else:
                        continue

                dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)

                tag = tag.strip()
                tag = [t for t in tag if t != '']

                for t in tag:
                    if t not in self.tag2id:
                        tag_id = len(self.tag2id)
                        self.tag2id[t] = tag_id
                        self.id2tag[tag_id] = t

                tag_ids = [self.tag2id[t] for t in tag]

                data.append({
                    'id': int(0),
                    'dscp_ids': dscp_ids,
                    'dscp_tokens': dscp_tokens,
                    'tag_ids': tag_ids,
                    'dscp': dscp
                })

        print("The number of tags for training: {}".format(len(self.tag2id)))

        return data

    def load_ganBert(self, file):
        data = []
        document = []

        with open(file, 'r') as f:
            contents = f.read()
            file_as_list = contents.splitlines()
            for line in file_as_list[1:]:
                split = line.split(" ")
                dscp = ' '.join(split[1:])

                inn_split = split[0].split(":")
                tag = inn_split[0] + "_" + inn_split[1]

                dscp_tokens = tokenizer.tokenize(dscp.strip())
                if len(dscp_tokens) > 510:
                    if self.data_config['overlength_handle'] == 'truncation':
                        dscp_tokens = dscp_tokens[:510]
                    else:
                        continue

                document.append(" ".join(dscp_tokens))

                dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)

                if tag in self.tag2id:
                    tag_id = self.tag2id[tag]
                elif tag == 'UNK_UNK':
                    tag_id = 0
                else:
                    tag_id = len(self.tag2id)
                    self.tag2id[tag] = tag_id
                    self.id2tag[tag_id] = tag

                data.append({
                    'id': 0,
                    'dscp_ids': dscp_ids,
                    'dscp_tokens': dscp_tokens,
                    'tag_ids': tag_id,
                    'dscp': dscp
                })

        print("The number of tags for training: {}".format(len(self.tag2id)))
        os.makedirs('cache', exist_ok=True)
        print(self.tag2id.keys())

        return data

    def filter_tags(self, file):
        tag_occurance = {}

        with open(file, 'r') as f_tag:
            tags = f_tag.readlines()
            for tag in tags:
                tag = tag.strip().split()
                tag = [t for t in tag if t != '']

                for t in tag:
                    if t not in tag_occurance:
                        tag_occurance[t] = 1
                    tag_occurance[t] += 1

        for tag in tag_occurance:
            if self.data_config['min_tagFrequence'] <= tag_occurance[tag] <= self.data_config['max_tagFrequence']:
                self.use_tags.add(tag)
        print('Total number of tags: {}'.format(len(tag_occurance)))
        print(sorted(tag_occurance.items(), key=lambda x: x[1], reverse=True))

    def load_EurLex_RCV2(self, file1, file2):
        data = []

        f_text = open(file1, 'r')
        texts = f_text.readlines()
        f_tag = open(file2, 'r')
        tags = f_tag.readlines()

        instanceCount = 0
        for text, tag in zip(texts, tags):

            dscp_tokens = tokenizer.tokenize(text.strip())
            if len(dscp_tokens) > 510:
                if self.data_config['overlength_handle'] == 'truncation':
                    dscp_tokens = dscp_tokens[:510]
                else:
                    continue

            dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)

            tag = tag.strip().split()
            tag = [t for t in tag if t != '']

            if self.use_tags is not None:
                tag = [t for t in tag if t in self.use_tags]

            if len(tag) == 0:
                continue

            if instanceCount > self.data_config['intanceNum_limit']:
                break
            instanceCount += 1

            for t in tag:
                if t not in self.tag2id:
                    tag_id = len(self.tag2id)
                    self.tag2id[t] = tag_id
                    self.id2tag[tag_id] = t

            tag_ids = [self.tag2id[t] for t in tag]

            data.append({
                'id': 0,
                'dscp_ids': dscp_ids,
                'dscp_tokens': dscp_tokens,
                'tag_ids': tag_ids,
                'dscp': text
            })

        print("The number of tags for training: {}".format(len(self.tag2id)))

        f_text.close()
        f_tag.close()

        return data