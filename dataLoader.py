import csv
import copy
import os

from random import shuffle

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

token_table = {'ecommerce': 'electronic commerce'}


class ProgramWebDataset(Dataset):
    def __init__(self, data, co_occur_mat, tag2id, id2tag=None, tfidf_dict=None):
        self.data = data
        self.co_occur_mat = co_occur_mat
        self.tag2id = tag2id
        if id2tag is None:
            id2tag = {v: k for k, v in tag2id.items()}
        self.id2tag = id2tag
        self.tfidf_dict = tfidf_dict

    @classmethod
    def from_dict(cls, data_dict):
        return ProgramWebDataset(data_dict.get('data'),
                                 data_dict.get('co_occur_mat'),
                                 data_dict.get('tag2id'),
                                 data_dict.get('id2tag'))

    @classmethod
    def from_csv(cls, api_csvfile, net_csvfile):
        data, tag2id, id2tag, document = ProgramWebDataset.load(api_csvfile)
        co_occur_mat = ProgramWebDataset.stat_cooccurence(data, len(tag2id))
        #co_occur_mat = ProgramWebDataset.similar_net(net_csvfile, tag2id)

        tfidf_dict = ProgramWebDataset.get_tfidf_dict(document)

        return ProgramWebDataset(data, co_occur_mat, tag2id, id2tag, tfidf_dict)

    @classmethod
    def load(cls, f):
        data = []
        tag2id = {}
        id2tag = {}

        document = []
        tag_occurance = {}
        buf = []

        with open(f, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
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

                if len(tag) == 0:
                    continue
                buf.append((id, title_ids, dscp_ids, tag))

                for t in tag:
                    if t not in tag_occurance:
                        tag_occurance[t] = 1
                    tag_occurance[t] += 1

        ignored_tags = set()
        for tag in tag_occurance:
            if tag_occurance[tag] < 100:
                ignored_tags.add(tag)

        for row in buf:

            id, title_ids, dscp_ids, tag = row

            if ignored_tags is not None:
                tag = [t for t in tag if t not in ignored_tags]

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
                'title_ids': title_ids,
                'title_tokens': title_tokens,
                'dscp_ids': dscp_ids,
                'dscp_tokens': dscp_tokens,
                'tag_ids': tag_ids,
                'dscp': dscp
            })

        print("The number of tags for training: {}".format(len(tag2id) - len(ignored_tags)))
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

        tfidf_model = TfidfVectorizer(sublinear_tf=True,
                                        strip_accents='unicode',
                                        analyzer='char',
                                        stop_words='english',
                                        ngram_range=(2, 6),
                                        max_features=50000).fit(document)
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

    def to_dict(self):
        data_dict = {
            'data': self.data,
            'co_occur_mat': self.co_occur_mat,
            'tag2id': self.tag2id,
            'id2tag': self.id2tag
        }
        return data_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

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

    def collate_fn(self, batch):
        result = {}
        # construct input
        inputs = [e['title_ids'] + e['dscp_ids'] for e in batch]

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

        dscp = [e['dscp'] for e in batch]

        inputs_tokens = [e['dscp_tokens'] for e in batch]
        inputs_tfidf = torch.zeros(size=(len(batch), max_len+2))

        for i, token_list in enumerate(inputs_tokens):
            for j, item in enumerate(token_list):
                if item in self.tfidf_dict:
                    inputs_tfidf[i, j+1] = self.tfidf_dict[item]

        # inputs_tfidf[inputs_tfidf>0]=1
        # ids *= inputs_tfidf.long()
        # ids[ids==0]=100

        return (ids, token_type_ids, attention_mask, inputs_tfidf), tags, dscp


# def CrossValidationSplitter(dataset, seed):
#     data_len = len(dataset)  # 获取文件总数

#     npr = np.random.RandomState(seed)

#     data_index = npr.permutation(data_len)

#     remainder = data_len % 10
#     data_index = data_index[:-1 * remainder]
#     data_index = np.array(data_index)
#     data_block = data_index.reshape(10, -1)  # split the data into 10 groups
#     return data_block


def load_dataset(api_csvfile=None, net_csvfile=None):

    cache_file_head = api_csvfile.split("/")[-1]

    if os.path.isfile(os.path.join('cache', cache_file_head + '.train')) \
            and os.path.isfile(os.path.join('cache', cache_file_head + '.eval')) \
            and os.path.isfile(os.path.join('cache', cache_file_head + '.encoded_tag')) \
            and os.path.isfile(os.path.join('cache', cache_file_head + '.tag_mask')) and False:

        print("load dataset from cache")

        train_dataset, val_dataset = ProgramWebDataset.from_dict(
            torch.load(os.path.join('cache', cache_file_head + '.train'))), ProgramWebDataset.from_dict(
            torch.load(os.path.join('cache', cache_file_head + '.eval')))
        encoded_tag, tag_mask = torch.load(os.path.join('cache', cache_file_head + '.encoded_tag')), \
                                torch.load(os.path.join('cache', cache_file_head + '.tag_mask'))

    else:

        print("build dataset")

        if not os.path.exists('cache'):
            os.makedirs('cache')

        dataset = ProgramWebDataset.from_csv(api_csvfile, net_csvfile)

        encoded_tag, tag_mask = dataset.encode_tag()

        torch.save(encoded_tag, os.path.join('cache', cache_file_head + '.encoded_tag'))
        torch.save(tag_mask, os.path.join('cache', cache_file_head + '.tag_mask'))

        data = np.array(dataset.data)
        train_dataset = dataset
        val_dataset = copy.copy(dataset)
        ind = np.random.permutation(len(data))
        split = int(len(data) * 0.8)
        train_dataset.data = data[ind[:split]].tolist()
        val_dataset.data = data[ind[split:]].tolist()

        torch.save(train_dataset.to_dict(), os.path.join('cache', cache_file_head + '.train'))
        torch.save(val_dataset.to_dict(), os.path.join('cache', cache_file_head + '.eval'))

    print("train_data_size: {}".format(len(train_dataset.data)))
    print("val_data_size: {}".format(len(val_dataset.data)))

    return train_dataset, val_dataset, encoded_tag, tag_mask
