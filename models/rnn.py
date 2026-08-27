import torch
import torch.nn as nn

PAD_IDX = 0

class SpamRNN(nn.Module):
    def __init__(self, vocab_size, embed_size = 128, hidden_size = 256, num_layers = 5, 
                 dropout = 0.3, num_classes = 2):
        super().__init__()
        #nn.Embedding (vocab_size, embed_size): input 개수, output 개수;
        # input 개수: vocab_size, n개의 단어들을 갖고 있다는 뜻
        # output 개수: embed_size(embed_dim), 몇차원 공간
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        #input_size: the number of expected features in the input x
        #hidden_size: the number of features in the hidden state h
        #num_layers: number of recurrent layers
        #batch_first: if True, then the input(batch, seq, feature) => (32, 몇번째, embed)


        self.rnn = nn.RNN(
            input_size = embed_size,
            hidden_size= hidden_size,
            num_layers= num_layers,                       #재귀반복횟수
            batch_first= True,
            dropout= dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embed= self.embedding(x)
        out, _ = self.rnn(embed)
        #batch, seq, hidden으로 구성
        last = out[:, -1, :]
        result = self.fc(self.dropout(last))
        return result 



#0826복습문제: LSTM 클래스 생성
#https://docs.pytorch.org/docs/2.13/generated/torch.nn.LSTM.html
class SpamLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, dropout=0.3, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=5, batch_first = True, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward (self, x):
        embedding = self.embedding(x)
        out, _ = self.lstm(embedding)
        #batch, seq(슬라이싱 -1: 시퀀스에서 마지막 하나의 값 = 다 돌고 최신의 시퀀스 결과만), hidden(임베딩 차원)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))

# LSTM
# <게이트 3개>
# Forget Gate (망각 게이트) "무엇을 잊을 것인가?" : 이전 t-1과 현재를 받아, 버릴 정보 결정
# Input Gate (입력 게이트): "무엇을 기억할 것인가?" : 현재를 받아, 새롭게 저장할 정보 결정
# Output Gate (출력 게이트): "무엇을 출력으로 내보낼 것인가?" : 업데이트된 상태 바탕으로 다음 t+1로 보낼 정보 결정


# GRU
# <게이트 2개>
# Reset Gate (리셋 게이트): "이전 기억을 얼마나 무시할 것인가?"
# Update Gate (업데이트 게이트): "이전 기억과 새 기억의 비율을 어떻게 가져갈 것인가?"



class SpamGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, dropout=0.3, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=5, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.gru(embed)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))