# data load
import pandas as pd

# for Regular Expression(정규 표현식)
import re
from tqdm import tqdm

# remove Stop tags
from konlpy.tag import Mecab

m = Mecab()

# for Checking
import matplotlib.pyplot as plt


class ReviewDataset():
    def __init__(self, train_data, test_data):

        # load Data
        self.train_data = train_data
        self.test_data = test_data

        # stop Tags
        self.stop_tag = ['JK', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM']

    def remove_unnecessary(self):

        # remove ID
        self.train_data.drop(['id'], axis=1, inplace=True)
        self.test_data.drop(['id'], axis=1, inplace=True)

        # remove Null
        self.train_data.dropna(inplace=True)
        self.test_data.dropna(inplace=True)

        # remove Same Data
        self.train_data.drop_duplicates(subset=['document'], inplace=True, ignore_index=True)
        self.test_data.drop_duplicates(subset=['document'], inplace=True, ignore_index=True)

        self.train_data = self.remove_not_korea(self.train_data)
        self.test_data = self.remove_not_korea(self.test_data)

        self.train_data = self.remove_stop_tags(df=self.train_data, tags=self.stop_tag)
        self.test_data = self.remove_stop_tags(df=self.test_data, tags=self.stop_tag)
        
        self.check()

    # remove without korea
    def remove_not_korea(self,df):
        for idx, row in tqdm(df.iterrows(), desc='removing_not_korean', total=len(df)):
            new_doc = re.sub("[^가-힣]", '', row['document']).strip()
            df.loc[idx, 'document'] = new_doc
        return df
    
    # remove unnecessary morpheme
    def remove_stop_tags(self,df, tags):
        for idx, row in tqdm(df.iterrows(), desc='removing josa', total=len(df)):
            josa_removed = [x[0] for x in m.pos(row['document']) if x[1] not in tags]
            df.loc[idx, 'document'] = ' '.join(josa_removed)
        return df
    
    def check(self):
        train_vlcnt = self.train_data['label'].value_counts().reset_index()
        test_vlcnt = self.test_data['label'].value_counts().reset_index()

        plt.subplot(1,2,1)
        plt.title('train_data', fontsize=20)
        plt.bar(train_vlcnt['index'], train_vlcnt['label'], color='blue')

        plt.subplot(1,2,2)
        plt.title('test_data', fontsize=20)
        plt.bar(test_vlcnt['index'], test_vlcnt['label'], color='red')

        plt.show()


train_data = pd.read_csv("~/workspace/DiaryProject/backend/data/ratings_train.txt", sep='\t')
test_data = pd.read_csv("~/workspace/DiaryProject/backend/data/ratings_test.txt", sep='\t')

dataseter = ReviewDataset(train_data=train_data, test_data=test_data)

dataseter.remove_unnecessary()