from tqdm import tqdm

class Tokenizer():
    def __init__(self):
        super().__init__()
        self.word_list = dict()
        self.new_word_num = 1
        self.encoding_list = dict()
        self.index_list = list()
        self.temp = 0
    
    def fit_morpheme(self, morph_list):
        for text in tqdm(morph_list):
            for char in text:
                # 넘버링
                if self.new_word_num == 1:
                    self.word_list[char] = self.new_word_num
                    self.new_word_num += 1
                else:
                    # encoding_list 초기 구현
                    if char in self.word_list:
                        for word, index in self.wㄴord_list.copy().items():
                                if word == char:
                                    self.encoding_list[char] = index
                        self.encoding_list[f'Space{self.temp}'] = False
                        self.temp += 1
                        pass
                    else:
                        self.word_list[char] = self.new_word_num
                        self.new_word_num += 1

        # # encoding_list 초기 구현
        # for text in tqdm(morph_list):
        #     for char in text:
        #         for word, index in self.word_list.copy().items():
        #             if word == char:
        #                 self.encoding_list[char] = index
        #     self.encoding_list[f'Space{self.temp}'] = False
        #     self.temp += 1
        return self.encoding_list
    
    def by_index(self, incoding_text_list):
        temp_list = list()
        for word, index in incoding_text_list.items():
            if index == False:
                self.index_list.append(temp_list)
                temp_list = []
            else:
                temp_list.append(index)
        return self.index_list

    # def text_frequency(self):
    #     threshold = 3
    #     if 
            


# test_list = [['굳다'], ['뭐', '야', '평점', '나쁘다', '않다', '점', '짜다', '리', '더', '더욱', '아니다'], ['지루하다', '않다', '완전', '막장', '임', '돈', '주다', '보기', '에는']]
# tokenizer = Tokenizer()
# tokenized_text = tokenizer.fit_morpheme(test_list)
# tokenized_text = tokenizer.by_index(tokenized_text)
# print(tokenized_text)