import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import os
import numpy as np
import urllib.request
import pandas as pd

# 수행 경로 설정
os.chdir(os.getcwd() + "\data")

# test,train data 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

# train,test data 정의
train_data = pd.read_table("ratings_train.txt")
test_data = pd.read_table("ratings_test.txt")

# 중복 요소 삭제
train_data.drop_duplicates(subset=['document'], inplace=True)
test_data.drop_duplicates(subset=['document'],inplace=True)

# Null 값 제거
train_data = train_data.dropna(how = 'any')
test_data = test_data.dropna(how = 'any')

# 한글과 공백을 제외, 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

# 특수기호 제거
train_data['document'] = train_data['document'].str.replace('^ +', "")
test_data['document'].replace('', np.nan, inplace=True)
test_data['document'] = test_data['document'].str.replace('^ +', "")
test_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())

# 특수 기호 등 제거 후, Null 값 제거
train_data = train_data.dropna(how = 'any')
test_data = test_data.dropna(how = 'any')

print(len(train_data), len(test_data))
