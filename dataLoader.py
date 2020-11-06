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

# All data in one excel
class allData(Dataset):
    tag_weight = []
    def __init__(self, train_data=None,unlabeled_train_data=None, test_data=None, co_occur_mat=None, tag2id=None, id2tag=None, tfidf_dict=None):
        self.train_data = train_data
        self.unlabeled_train_data = unlabeled_train_data
        self.test_data = test_data
        self.co_occur_mat = co_occur_mat
        self.tag2id = tag2id
        if id2tag is None:
            id2tag = {v: k for k, v in tag2id.items()}
        self.id2tag = id2tag
        self.tfidf_dict = tfidf_dict

    @classmethod
    def from_dict(cls, data_dict):
        return allData(data_dict.get('train_data'),
                                 data_dict.get('test_data'),
                                 data_dict.get('co_occur_mat'),
                                 data_dict.get('tag2id'),
                                 data_dict.get('id2tag'))

    def to_dict(self):
        data_dict = {
            'train_data': self.train_data,
            'test_data': self.test_data,
            'co_occur_mat': self.co_occur_mat,
            'tag2id': self.tag2id,
            'id2tag': self.id2tag
        }
        return data_dict

    @classmethod
    def from_csv(cls, data_path):
        data, tag2id, id2tag, document = allData.load_programWeb(data_path)
        # data, tag2id, id2tag, document = allData.load_news_group20(data_path)

        data = np.array(data)
        ind = np.random.RandomState(seed=10).permutation(len(data))
        split = int(len(data) * 0.05)
        split2 = int(len(data) * 0.9)
        train_data = data[ind[:split]].tolist()
        unlabeled_train_data = data[ind[split:split2]].tolist()
        test_data = data[ind[split2:]].tolist()

        co_occur_mat = allData.stat_cooccurence(data, len(tag2id))
        tfidf_dict = allData.get_tfidf_dict(document)

        return allData(train_data, unlabeled_train_data, test_data, co_occur_mat, tag2id, id2tag, tfidf_dict), allData.tag_weight

    @classmethod
    def load_news_group20(cls, f):
        data = []
        tag2id = {}
        id2tag = {}

        document = []

        #csv.field_size_limit(500 * 1024 * 1024)
        csv.field_size_limit(sys.maxsize)
        with open(f, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if len(row) != 4:
                    continue
                id, index, tag, dscp = row

                dscp_tokens = tokenizer.tokenize(dscp.strip())
                if len(dscp_tokens) > 510:
                    continue

                document.append(" ".join(dscp_tokens))
                dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)

                tag = tag.strip()
                tag = [t for t in tag if t != '']

                if len(tag) == 0:
                    continue

                for t in tag:
                    if t not in tag2id:
                        tag_id = len(tag2id)
                        tag2id[t] = tag_id
                        id2tag[tag_id] = t

                tag_ids = [tag2id[t] for t in tag]

                data.append({
                    'id': int(id),
                    'dscp_ids': dscp_ids,
                    'dscp_tokens': dscp_tokens,
                    'tag_ids': tag_ids,
                    'dscp': dscp
                })

        print("The number of tags for training: {}".format(len(tag2id)))
        os.makedirs('cache', exist_ok=True)

        return data, tag2id, id2tag, document



    @classmethod
    def load_programWeb(cls, f):
        data = []
        zeroshot_data = []
        tag2id = {}
        id2tag = {}

        document = []
        tag_occurance = {}

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

        ignored_tags = set()
        # ignored_tags = set(['Tools','Applications','Other', 'API', 'Software-as-a-Service','Platform-as-a-Service','Data-as-a-Service'])  #
        for tag in tag_occurance:
            if tag_occurance[tag] > 2000:
                ignored_tags.add(tag)

        print(ignored_tags)

        with open(f, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                zeroshot_data = False
                if len(row) != 4:
                    continue
                id, title, dscp, tag = row

                title_tokens = tokenizer.tokenize(title.strip())
                dscp_tokens = tokenizer.tokenize(dscp.strip())
                if len(title_tokens) + len(dscp_tokens) > 510:
                    continue

                document.append(" ".join(title_tokens) + " ".join(dscp_tokens))

                title_ids = tokenizer.convert_tokens_to_ids(title_tokens)
                dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)

                tag = tag.strip().split('###')
                tag = [t for t in tag if t != '']

                if ignored_tags is not None:
                    tag = [t for t in tag if t not in ignored_tags]

                # if len(set(tag)) < 2:
                #     continue

                if len(tag) == 0:
                    continue

                for t in tag:
                    if t not in tag2id:
                        tag_id = len(tag2id)
                        tag2id[t] = tag_id
                        id2tag[tag_id] = t

                tag_ids = [tag2id[t] for t in tag]

                data.append({
                    'id': int(id),
                    'dscp_ids': title_ids + dscp_ids,
                    'dscp_tokens': title_tokens + dscp_tokens,
                    'tag_ids': tag_ids,
                    'dscp': dscp
                })

        print("The number of tags for training: {}".format(len(tag2id)))
        os.makedirs('cache', exist_ok=True)

        return data, tag2id, id2tag, document

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

    @classmethod
    def stat_cooccurence(cls, data, tags_num):

        co_occur_mat = torch.zeros(size=(tags_num, tags_num))
        for i in range(len(data)):
            tag_ids = data[i]['tag_ids']
            for t1 in range(len(tag_ids)):
                for t2 in range(len(tag_ids)):
                    #if tag_ids[t1] != tag_ids[t2]:
                    co_occur_mat[tag_ids[t1], tag_ids[t2]] += 1

        return co_occur_mat

    @classmethod
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
        result = {}
        # construct input
        inputs = [e['dscp_ids'] for e in batch]  #e['title_ids'] +

        lengths = np.array([len(e) for e in inputs])
        max_len = np.max(lengths)
        inputs = [tokenizer.prepare_for_model(e, max_length=max_len+2, pad_to_max_length=True) for e in inputs]

        ids = torch.LongTensor([e['input_ids'] for e in inputs])
        token_type_ids = torch.LongTensor([e['token_type_ids'] for e in inputs])
        attention_mask = torch.FloatTensor([e['attention_mask'] for e in inputs])
        # construct tag
        tags = torch.zeros(size=(len(batch), self.get_tags_num()))
        for i in range(len(batch)):
            tags[i, batch[i]['tag_ids']] = 1.
            # tags[i] *= torch.from_numpy(np.array(allData.tag_weight)).float()

        dscp = [e['dscp'] for e in batch]

        return (ids, token_type_ids, attention_mask), tags, dscp


def load_allData(data_path=None):

    cache_file_head = data_path.split("/")[-1]

    if os.path.isfile(os.path.join('cache', cache_file_head + '.dataset')) \
            and os.path.isfile(os.path.join('cache', cache_file_head + '.encoded_tag')) \
            and os.path.isfile(os.path.join('cache', cache_file_head + '.tag_mask')) and False:

        print("load dataset from cache")

        dataset = TrainTestData.from_dict(torch.load(os.path.join('cache', cache_file_head + '.dataset')))

        encoded_tag, tag_mask = torch.load(os.path.join('cache', cache_file_head + '.encoded_tag')), \
                                torch.load(os.path.join('cache', cache_file_head + '.tag_mask'))
        tag_weight = 0

    else:

        print("build dataset")
        if not os.path.exists('cache'):
            os.makedirs('cache')

        dataset, tag_weight = allData.from_csv(data_path)
        torch.save(dataset.to_dict(), os.path.join('cache', cache_file_head + '.dataset'))

        # dataset.stat_cooccurence()

        encoded_tag, tag_mask = dataset.encode_tag()
        torch.save(encoded_tag, os.path.join('cache', cache_file_head + '.encoded_tag'))
        torch.save(tag_mask, os.path.join('cache', cache_file_head + '.tag_mask'))

    print("train_data_size: {}".format(len(dataset.train_data)))
    print("unlabeled_train_data: {}".format(len(dataset.unlabeled_train_data)))
    print("val_data_size: {}".format(len(dataset.test_data)))

    # tag_embedding_file = dataset.obtain_tag_embedding()

    return dataset, encoded_tag, tag_mask, tag_weight
