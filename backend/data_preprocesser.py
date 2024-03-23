import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import urllib.request
import pandas as pd
import re
from konlpy.tag import Okt
from tqdm import tqdm
from Tokenizer import Tokenizer

okt = Okt()

# 수행 경로 설정
os.chdir("/Users/jangsiu/workspace/DiaryProject/backend/data")

# test,train data 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

# train,test data 정의
train_data = pd.read_table("ratings_train.txt")
test_data = pd.read_table("ratings_test.txt")

# 중복 요소 삭제
train_data.drop_duplicates(subset=['document'], inplace=True)
test_data.drop_duplicates(subset=['document'],inplace=True)

# # Null 값 제거
train_data = train_data.dropna(how = 'any')
test_data = test_data.dropna(how = 'any')

# 한글, 공백 제외, 모두 
train_data['document'] = train_data['document'].str.replace(r"[^가-힣 ]", '',regex=True)
test_data['document'] = test_data['document'].str.replace(r"[^가-힣 ]", '', regex=True)

train_data['document'].replace('', np.nan, inplace=True)
test_data['document'].replace('', np.nan, inplace=True)

# 특수 기호 등 제거 후, Null 값 제거
train_data = train_data.dropna(how = 'any')
test_data = test_data.dropna(how = 'any')

# 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

X_train = []
for sentence in tqdm(train_data['document']):
    train_tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    train_stopwords_removed_sentence = [word for word in train_tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train.append(train_stopwords_removed_sentence)

X_test = []
for sentence in tqdm(test_data['document']):
    test_tokenized_sentence = okt.morphs(sentence, stem=True) # 토근화
    test_stopwords_removed_sentence = [word for word in test_tokenized_sentence if not word in stopwords] # 불용어 제거
    X_test.append(test_stopwords_removed_sentence)

tokenizer = Tokenizer()
X_train = tokenizer.fit_morpheme(X_train)
X_test = tokenizer.fit_morpheme(X_test)
print(X_train[:5])
# X_train = tokenizer.by_index(X_train)
# X_test = tokenizer.by_index(X_test)

# threshold = 3
# total_cnt = len(X_train)
# rare_cnt = 0
# total_freq = 0
# rare_freq = 0

# for key, value in X_train.items():
#     total_freq += value

#     if (value < threshold):
#         rare_cnt += 1
#         rare_freq += value

# print('단어 집합(vocabulary)의 크기 :',total_cnt)
# print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
# print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
# print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

