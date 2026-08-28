# [seq2seq의 장점, 핵심 아이디어 정리]
#1. 한계
#1-1. 기존 모델은 출력이 1:1 고정이 되어 있었음(I/O)
#1-2. 어순과 구조가 맞지 않는 언어의 번역 문제 / QA -> input을 읽고 바로 output을 낼 수 있도록 하는 구조

#2. 아이디어 -> "인코더 + 디코더" (I/O 길이 분리)

#[모델 훈련 연습]
#1. 데이터 준비
#2. 데이터셋 클래스 만들기
#3. seq2seq 모델 만들기
#4. 훈련-검증

import re, unicodedata

#텍스트 정제
def normalize(text, lang = 'en'):
    if lang == 'fr':
        #프랑스어 악센트 기호 처리
        text = unicodedata.normalize('NFD', text)
        #아스키 코드 범위 넘으면 무시, 다시 아스키로 변환
        text = text.encode('ascii', 'ignore').decode('ascii')

    text = text.lower().strip()
    text = re.sub(r'([.!?])', r' \1', text)       #., ?, ! 앞을 한 칸 띄울 것
    text = re.sub(r'[^a-z.!? ]+', ' ', text)    #a-z.!?공백이 아닌 것을 ' '(공백)으로 전환
    return text.split()

import os, shutil, urllib.request, zipfile, random

#인터넷에서 데이터 다운로드
def load_data(max_pairs = 20000):
    data_file = './data/data/eng-fra.txt'
    if not os.path.exists(data_file):
        #다운로드가 안됐다면 다운로드 하도록.
        os.makedirs('data', exist_ok=True)
        urllib.request.urlretrieve(
            'https://download.pytorch.org/tutorial/data.zip', 'data/data.zip'
        )

        with zipfile.ZipFile('data/data.zip') as z:
            z.extractall('data')

    pairs = []
    with open('./data/data/eng-fra.txt', encoding='utf-8') as f:
        for line in f:
            parts = line.split('\t')
            if len(parts)< 2:
                continue
            en = normalize(parts[0], 'en')
            fr = normalize(parts[1], 'fr')

            pairs.append((en,fr))
    random.shuffle(pairs)
    pairs = pairs[:max_pairs]
    return pairs

from collections import Counter

#vocab 만들기
def bulid_vocab(pairs):
    en_cnt, fr_cnt = Counter(), Counter()
    for en, fr in pairs:
        en_cnt.update(en)
        fr_cnt.update(fr)

    #eng vocab
    #fra vocab
    #MAX_LEN = 30
    en_vocab, fr_vocab = Vocab(30), Vocab(30)

    #w-> 단어, f->빈도
    for w, f in en_cnt.items():
        if f >= 2:
            en_vocab.add(w)

    for w, f in fr_cnt.items():
        if f >= 2:
            fr_vocab.add(w)

    print(f'영어 어휘 : {len(en_vocab)}, 프랑스어 어휘 {len(fr_vocab)}')
    return en_vocab, fr_vocab

class Vocab:
    def __init__(self, max_len):
        #w : word, i : index -> 단어를 숫자로, 숫자를 단어로
        self.w2i = {'<PAD>': 0,'<SOS>': 1,'<EOS>': 2,'<UNK>': 3}
        self.i2w = {v:k for k, v in self.w2i.items()}
        self.MAX_LEN = max_len

    #vocab 추가
    def add(self, word):
        if word not in self.w2i:
            i = len(self.w2i)
            self.w2i[word] = len(self.w2i)
            self.i2w[i] = word

    #token 인코드(문자->숫자)
    def encode(self, tokens):
        #<SOS> [그,는, 말하다, 피곤하다고] <EOS>
        SOS, EOS, UNK = 1, 2, 3
        ids = [1] + [self.w2i.get(t,3) for t in tokens] + [2]
        #MAX_LEN(최장 단어의 길이) 보다 작은 경우 아래처럼 0번 더해줌(패딩)
        #MAX_LEN = 50 가정, 50 - len(ids) + 2(sos, eos) 형태
        ids += [0] * (self.MAX_LEN - len(ids) + 2)
        return  ids[:self.MAX_LEN + 2]

    #숫자->문자
    def decode(self, ids):
        out = [] #id가 변환되어 쌓일 문자열 리스트

        for i in ids:
            w = self.i2w.get(i, '<UNK>')
            # <sos> 나는 이렇게 말했다 <pad> <pad> <eos>
            if w in ('<PAD>', '<SOS>', '<EOS>'):
                continue #append하지 말고 넘어가라.
            out.append(w)
        return out

    def __len__(self): 
        return len(self.w2i)
    
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, num_layers = 5, dropout=0.3):
        super().__init__()
        #vocab_size, embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers = num_layers,
            batch_first=True,
            dropout= dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embed = self.embedding(x)
        dropout = self.dropout(embed)
        _,(hidden, cell) = self.lstm(dropout)
        return hidden, cell
    
        #LSTM 층 통과한 결과물? (분류기로 썼을때)
        #out, _ = self.lstm(embedding)
        #last = out[:, -1, :]
        #return  self.fc(self.dropout(last))
        #output: 전체T 은닉층(분류로써 필요한 것), (h_c, c_n): 마지막T 결과(인코더에게 필요한 것)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        #임베딩 레이어, lstm 레이어, 드롭아웃 동일
        self.embedding = nn.Embedding()
        self.lstm = nn.LSTM(

        )
        self.dropout = nn.Dropout(dropout)
        #1개 추가: 
        
    def forward(self, token, hidden, cell):
        

if __name__ == '__main__':
    pairs = load_data()
    print(f'페어 길이: {len(pairs)}, 페어[0]: {pairs[0]}')

    en_vocab, fr_vocab = bulid_vocab(pairs)
