import os
import re
import numpy as np
import urllib.request
import pandas as pd
from konlpy.tag import Okt
from tqdm import tqdm
from Tokenizer import Tokenizer
# from tensorflow. 텐서 플로우의 tokenizer 다운로드

okt = Okt()

# 수행 경로 설정
os.chdir("/Users/jangsiu/workspace/DiaryProject/backend/data")

# test,train data 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

class ReviewDataset():
    def __init__(self):

        # 데이터셋 로드
        self.train_data = pd.read_table("ratings_train.txt")
        self.test_data = pd.read_table("ratings_test.txt")

        # 불용어 정의
        self.stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

        # tokenizer 정의
        self.tokenizer = Tokenizer()

        self.max_len = 30

    def remove_unnecessary(self):

        # 중복 요소 삭제
        self.train_data.drop_duplicates(subset=['document'], inplace=True)
        self.test_data.drop_duplicates(subset=['document'], inplace=True)

        # Null 값 제거
        self.train_data = self.train_data.dropna(how = "any")
        self.test_data = self.test_data.dropna(how = "any")

        # 한글(초성 제외), 공백을 제외 모두 제거
        self.train_data['document'] = self.train_data['document'].str.replace(r"[^가-힣 ]", '', regex=True)
        self.test_data['document'] = self.test_data['document'].str.replace(r"[^가-힣 ]", '', regex=True)

        # 빈 값을 모두 Null 값으로 변환
        self.train_data['document'].replace('', np.nan, inplace=True)
        self.test_data['document'].replace('', np.nan, inplace=True)

        # Null 값 제거
        self.train_data = self.train_data.dropna(how = "any")
        self.test_data = self.test_data.dropna(how = "any")
    
    def tokenize_without_stopwords(self):

        self.X_train = []
        self.X_test = []
        
        # 토큰화, 불용어 제거
        for sentence in tqdm(self.train_data['document']):
            train_tokenized_sentence = okt.morphs(sentence, stem=True)
            train_stopwords_removed_sentence = [word for word in train_tokenized_sentence if not word in self.stopwords]
            self.X_train.append(train_stopwords_removed_sentence)
        
        # ...
        for sentence in tqdm(self.test_data['document']):
            test_tokenized_sentence = okt.morphs(sentence, stem=True)
            test_stopwords_removed_sentence = [word for word in test_tokenized_sentence if not word in self.stopwords]
            self.X_test.append(test_stopwords_removed_sentence)

    def string_tokenized(self):

        # 정수 시퀀스로 변환
        self.X_train = self.tokenizer.fit_morpheme(self.X_train)
        self.X_test = self.tokenizer.fit_morpheme(self.X_test)

        # y 값에 긍정, 부정을 매기는 label 값으로 정의
        self.y_train = np.array(self.train_data['label'])
        self.y_test = np.array(self.test_data['label'])

        threshold = 3 # 제거할 단어의 빈도수의 기준
        for char,index in self.X_data.items():
            total_cnt = [v for k,v in self.X_data if k == char]
            if len(total_cnt) >= threshold:
                pass
            else:
                del self.X_data[char]
        self.tokenizer.by_index(self.X_data)

    def remove_empty_samples(self):
        
        # 길이가 0인 샘플들 가져오기
        self.drop_train = [index for index, sentence in enumerate(self.X_train) if len(sentence) < 1]
        self.drop_test = [index for index, sentence in enumerate(self.X_test) if len(sentence < 1)]

        # 빈 샘플들을 제거
        self.X_train = np.delete(self.X_train, self.drop_train, axis=0)
        self.y_train = np.delete(self.y_train, self.drop_train, axis=0)
        self.X_test = np.delete(self.X_test, self.drop_test, axis=0)
        self.X_train = np.delete(self.y_test, self.drop_test, axis=0)


    def remove_below_threshold_len(self, max_len, nested_list):
        count = 0
        for sentence in nested_list:
            if(len(sentence) <= max_len):
                count += 1
        if (max_len, (count / len(nested_list))*100) >= 90:
            self.X_train = self.pad_sequnces(self.X_train, maxlen = max_len)
        else:
            self.max_len += 1
            self.remove_below_threshold_len(self.max_len, nested_list)
        print(len(self.X_train), self.X_train)
    
    def pad_sequnces(nested_list, maxlen):
        for line in nested_list:
            if len(line) > maxlen:
                while len(line) == maxlen:
                    line.append(0)
            elif len(line) == maxlen:
                pass
            else:
                while len(line) == maxlen:
                    line.pop()
            

test = ReviewDataset()
test.remove_unnecessary()
test.tokenize_without_stopwords()
test.string_tokenized()
test.remove_empty_samples()
test.remove_below_threshold_len()



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

