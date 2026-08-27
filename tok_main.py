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
    return split_num[:n_train], split_num[n_train:n_train+n_valid], split_num[n_train+n_valid:]

import torch.nn as nn
import torch.optim as optim
from models.rnn import SpamRNN, SpamLSTM, SpamGRU
from utils.visualize import plot_comparison
from main import train, evaluate

if __name__ == '__main__':

    #login()

    #huggingface에서 tokenizer download.
    #AutoTokenizer.from_pretrained -> 사전 학습된 토크나이저 모델을 사용 가능
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    #0. 전체 데이터셋인 df 가져옴
    df = pd.read_csv('SMSSpamCollection', sep = '\t', header = None, names = ['label','text'])
    df['label'] = (df['label'] == 'spam').astype(int)       #데이터프레임 열 중 label 열을 0(정상), 1(스팸)으로 분류

    print(df.head())

    train_idx, valid_idx, test_idx = make_splits(len(df)) 

    #1. 전통적인 tokenizer를 거친 데이터셋을 갖고 와 lstm으로 훈련
    #1-1. 전통적인 tokenizer를 거친 데이터셋 로드
    original_vocab = main.build_vocab(df)
    original_data = main.SpamDataset(df, original_vocab, 50)

    original_train = DataLoader(Subset(original_data, train_idx), batch_size=32, shuffle=True)
    original_valid = DataLoader(Subset(original_data, valid_idx), batch_size=32)
    original_test = DataLoader(Subset(original_data, test_idx), batch_size=32)

    td, tl = next(iter(original_test))
    print(td, tl)

    #1-2. 모델 생성
    original_LSTM = SpamLSTM(len(original_vocab))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(original_LSTM.parameters())
    num_epochs = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #훈련, 평가
    original_history = train(original_LSTM, original_train, original_valid, criterion, optimizer, num_epochs, device, model_name = 'original')
    o_labels, o_preds = evaluate(original_LSTM, original_test, device, model_name = 'original')


    #2. 허깅페이스 tokenizer를 거친 데이터셋을 갖고 와 lstm으로 훈련
    #2-1.허깅페이스를 거친 데이터셋 로드
    hug_data = HugDataset(df, tokenizer, 50)
    hug_train = DataLoader(Subset(hug_data, train_idx), batch_size=32, shuffle=True)
    hug_valid = DataLoader(Subset(hug_data, valid_idx), batch_size=32)
    hug_test = DataLoader(Subset(hug_data, test_idx), batch_size=32)

    #2-2. 모델 생성, 훈련 및 평가
    #허깅페이스에서 다운받은 토크나이저의 특성 속에 vocab_size가 미리 정의되어 있음
    hug_LSTM = SpamLSTM(tokenizer.vocab_size)
    h_optimizer = optim.Adam(hug_LSTM.parameters())

    hug_history = train(hug_LSTM, hug_train, hug_valid, criterion, h_optimizer, num_epochs, device, model_name='hugging_face')
    h_labels, h_preds = evaluate(hug_LSTM, hug_test, device, model_name = 'hugging_face')

    histories, eval_results, train_models = [], [], {}

    histories.append(original_history)
    histories.append(hug_history)
    eval_results.append((o_labels, o_preds))
    eval_results.append((h_labels, h_preds))
    train_models['original'] = original_LSTM
    train_models['hugging_face'] = hug_LSTM

    plot_comparison(histories, ['original', 'hugging_face'])



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