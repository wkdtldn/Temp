## Load Dataset
import pandas as pd

train = pd.read_csv('~/workspace/DiaryProject/backend/data/ratings_train.txt', sep='\t')
test = pd.read_csv('~/workspace/DiaryProject/backend/data/ratings_test.txt', sep='\t')

## ID : unnecessary => remove
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

## If Null => True
print(f'trainset null 개수: \n{train.isnull().sum()}\n')
print(f'testset null 개수: \n{test.isnull().sum()}')

## Remove Null
train.dropna(inplace=True)
test.dropna(inplace=True)

## Remove same Data
print(f"중복 제거 전 train length: {len(train)}")
train.drop_duplicates(subset=['document'], inplace=True, ignore_index=True)
print(f"중복 제거후 train length: {len(train)}\n")
print(f"중복 제거 전 test length: {len(test)}")
test.drop_duplicates(subset=['document'], inplace=True)
print(f"중복 제거 후 test length: {len(test)}\n")

## Except Korea(with 초성(initial consonant)) remove All
import re
from tqdm import tqdm

def removing_non_korean(df):
    for idx, row in tqdm(df.iterrows(), desc='removing_non_korean', total=len(df)):
        new_doc = re.sub('[^가-힣]', '', row['document']).strip()
        df.loc[idx, 'document'] = new_doc
    return df

train = removing_non_korean(train)
test = removing_non_korean(test)

## Remove StopWords (with. Mecab | Faster than others)
from konlpy.tag import Mecab

m = Mecab()

tags = ['JK', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM']

def remove_josa_mecab(df, tags):
    for idx, row in tqdm(df.iterrows(), desc='removing josa', total=len(df)):
        josa_removed = [x[0] for x in m.pos(row['document']) if x[1] not in tags]
        df.loc[idx, 'document'] = ' '.join(josa_removed)
    return df

train_mecab = remove_josa_mecab(train, tags)
test_mecab = remove_josa_mecab(test, tags)

## Check
import matplotlib.pyplot as plt
plt.style.use('seaborn')

train_mecab_vlcnt = train_mecab['label'].value_counts().reset_index()
test_mecab_vlcnt = test_mecab['label'].value_counts().reset_index()

plt.subplot(1, 2, 1)
plt.title('train_mecab', fontsize=20)
plt.bar(train_mecab_vlcnt['index'], train_mecab_vlcnt['label'], color='skyblue')

plt.subplot(1, 2, 2)
plt.title('test_mecab', fontsize=20)
plt.bar(test_mecab_vlcnt['index'], test_mecab_vlcnt['label'], color='skyblue')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

train_mecab_doc_len = [len(x) for x in train_mecab['document']]
test_mecab_doc_len = [len(x) for x in test_mecab['document']]

plt.subplots(constrained_layout=True)

plt.subplot(2, 1, 1)
plt.title('train_mecab', fontsize=20)
plt.hist(train_mecab_doc_len, bins=30)

plt.subplot(2, 1, 2)
plt.title('test_mecab', fontsize=20)
plt.hist(test_mecab_doc_len, bins=30)

plt.show()

train_mecab.to_csv('data/train_mecab.csv', index=False)
test_mecab.to_csv('data/test_mecab.csv', index=False)