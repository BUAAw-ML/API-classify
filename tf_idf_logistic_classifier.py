import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import sklearn.metrics as sm
from scipy.sparse import hstack


# class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

total = pd.read_csv('data/news_group20.csv',  encoding='utf-8', error_bad_lines=False).fillna(' ')
# test = pd.read_csv('../data/test.csv').fillna(' ')
print(total)
exit()

# domain = pd.read_csv('data/ProgrammerWeb/domainnet.csv', encoding='utf-8')
#
# print(domain)



class_names = total['tag'].drop_duplicates().values

test = total.sample(frac=0.2, axis=0, random_state=0)
train = total[~total['id'].isin(test['id'].values)]


train_text = train['descr']
test_text = test['descr']
all_text = pd.concat([train_text, test_text])

labels = dict()
for cla in tqdm(class_names):
    clist = list()
    for idx, row in train.iterrows():
        if row['tags2'].find(cla) == -1:
            clist.append(0)
        else:
            clist.append(1)
    labels[cla] = clist


label2 = labels.copy()
class_names2 = list(class_names)

key = list(labels.keys())
for k in key:
    if sum(labels[k]) < 100:
        labels.pop(k)
#         print(k)
#         class_names2.remove(k)


class_names2 = list(labels.keys())
len(class_names2)


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])


scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in tqdm(class_names2):
    train_target = labels[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

gt_labels = dict()
for cla in tqdm(class_names2):
    clist = list()
    for idx, row in test.iterrows():
        if row['tags2'].find(cla) == -1:
            clist.append(0)
        else:
            clist.append(1)
    gt_labels[cla] = clist


mAP = list()
for cla in tqdm(class_names2):
    ap = sm.average_precision_score(gt_labels[cla],submission[cla])
    mAP.append(ap)
print(np.mean(mAP))



# MAP
# 0.501658837537489  -- 大于50 的 178个tag
# 0.5550490386566358 -- 大于100 的 97个tag
