#RNN, LSTM, GRU
# REsNet- > (residual)잔차, 구조가 특징
# RNN -> Recurrent 구조

# RNN, LSTM, GRU 만들때도 -> 토큰화 + vocab 거쳐야함
#urllib.request: 인터넷 자료 다운로드 라이브러리
import os, re, urllib.request, zipfile
import pandas as pd

MAX_LEN = 50


def load_data(data_path='SMSSpamCollection', batch_size = 32):
    #os.path.exists(경로): '경로'가 존재하는지?
    if not os.path.exists(data_path):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
        #정해놓은 url로 가서 zip 다운로드
        urllib.request.urlretrieve(url, 'smsspam.zip')
        with zipfile.ZipFile('smsspam.zip') as z:
            z.extractall('.')
        print('완료')


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = text.split()
    return text

#문자열을 정수로, 어떤 문자열이 몇 번 정수로 바뀌었는지 기억함
from collections import Counter #wordcloud 만들 때 '단어 수' 함수
def build_vocab(df, min_freq=2):
    #나는 딥러닝을 공부하고 있어. 딥러닝은 정말 많은 작업을 할 수 있어.
    # 나 0 는 1 딥러닝2 을 3 공부하고 4 있어 5 딥러닝 2 은 6 정말 7 많은 8 작업 9 을 3... -> 단어가 몇 번 나오는지 세야하기에.
    counter= Counter(tok for text in df for tok in tokenize(text))       
    #패딩: 빈자리 메꿔줌. (앞 문장과 뒷 문장의 크기 차이. 더 크기가 큰 문장이 input의 기준이 됨. 그래서 크기가 더 작은 문장에 dummy를 넣어줌. pre/after padding 종류 2가지 있음.)
    #알려지지 않음: unknown. 추론 시 학습 풀에서 나오지 않은 단어가 나오는 경우 내가 학습한 규칙에 따라 단어를 전처리한 뒤 넣어줘야하기에. 
    vocab = {'<PAD>': 0, '<UNK>': 1}
    #counter.items(딥러닝, 2)
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab




def preprocessing(data_path = 'SMSSpamCollection'):
    df = pd.read_csv(data_path, sep='\t', header = None, names =['label', 'text'])
    #print(df. head())
    
    df['label'] = (df['label'] == 'spam').astype(int)       #label 열 숫자로 변환
    print(f'스팸이 아닌 것:{(df.label == 0).sum()}, 스팸인 것: {(df.label == 1).sum()}')

    #vocab으로 변환 => 어휘 사전
    vocab = build_vocab(df['text'])
    #print(vocab)

#훈련용 클래스 생성: 자연어 모델을 위한 커스텀 데이터셋 만들긴
from torch.utils.data import DataLoader, Dataset, random_split
import torch
class SpamDataset(Dataset):
    def __init__(self, df, vocab):
        #데이터 + 라벨 필요
        #text의 vocab기반 정수 전환
        self.texts = df['text'].tolist()
        #labels의 tensor 전환 필요. dtype=int64
        self.labels = torch.tensor(df['label'].tolist(), dtype = torch.long)
        self.vocab = vocab

        #MAX_LEN: 최대 문장 길이를 알아야 padding 가능
    def _text_to_tensor(self, text, vocab, max_len = MAX_LEN):
        sample = [vocab.get(t) for t in tokenize(text)]

        if len(sample) >= max_len:
            sample = sample[:max_len]
        else:
            sample +=[0] * (max_len)

    def __len__(self):

    def __getitem__(self, index):



if __name__ == '__main__':
    preprocessing()