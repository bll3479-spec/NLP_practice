#토크나이저만 변경했는데 성능이 달라질 수 있는지 비교

#토크나이저 가져오기
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import pandas as pd

#데이터셋 클래스 정의
class HugDataset(Dataset):
    def __init__(self, df, tokenizer, MAX_LEN):
        super().__init__()
        self.texts = df['text'].tolist()
        self.labels =  torch.tensor(df['label'].tolist(), dtype = torch.long)
        self.tokenizer = tokenizer
        self.max_len = MAX_LEN
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #encoding
        #max_length = 문장 최대 길이.(roberta; 512)
        #padding = 패딩의 기준을 어떻게?
        #truncation => True(512 넘으면 text 자르기)
        #return_tensors => pt: pytorch 스타일로 return
        enc = self.tokenizer(self.texts[index],
                             max_length = self.max_len,
                             padding = 'max_length',
                             truncation = True,
                             return_tensors = 'pt')
        #squeeze: 축을 하나 없애기
        return enc['input_ids'].squeeze(0), self.labels[index]

from huggingface_hub import login
import main

def make_splits(length_of_df):
    split_num = torch.randperm(length_of_df, generator = torch.Generator().manual_seed(42)).tolist()
    n_train = int(length_of_df * 0.7)
    n_valid = int(length_of_df * 0.15)
    return split_num[:n_train], split_num[n_train:n_train+n_valid, split_num[n_train+n_valid]]


if __name__ == '__main__':

    #login()

    #huggingface에서 tokenizer download.
    #AutoTokenizer.from_pretrained -> 사전 학습된 토크나이저 모델을 사용 가능
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    #0. 전체 데이터셋인 df 가져옴
    df = pd.read_csv('SMSSpamCollection', sep = '\t', header = None, names = ['labels','text'])
    df['label'] = (df['label'] == 'spam').astype(int)       #데이터프레임 열 중 label 열을 0(정상), 1(스팸)으로 분류

    train_idx, valid_idx, test_idx =make_splits(len(df)) 

    #1. 전통적인 tokenizer를 거친 데이터셋을 갖고 와 lstm으로 훈련

    #2. 허깅페이스 tokenizer를 거친 데이터셋을 갖고 와 lstm으로 훈련












    # print(f'토크나이저의 어휘 크기 : {tokenizer.vocab_size}')
    # print(f'토크나이저의 최대 입력 길이 : {tokenizer.model_max_length}')
    # print(f'특수 토큰 ID 확인 : {tokenizer.special_tokens_map}')

    # sentence = input('자를 영어 문장을 입력하시오')
    # tokens = tokenizer.tokenize(sentence)
    # print(f'토근화 결과: {tokens}')

    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(f'입력되는 숫자: {input_ids}')

    # #실제 처리
    # enc = tokenizer(sentence)

    # #print(enc.keys())
    # print(enc['input_ids'])
    # print(enc['attention_mask'])
    # #디코딩
    # decode = tokenizer.convert_ids_to_tokens(enc['input_ids'])
    # print(f'디코딩 결과: {decode}')