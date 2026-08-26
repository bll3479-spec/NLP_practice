import torch
import torch.nn as nn

PAD_IDX = 0

class SpamRNN(nn.Module):
    def __init__(self, vocab_size, embed_size = 64, hidden_size = 128, num_layers = 3, 
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
        last = out[:, -1, :]
        result = self.fc(self.dropout(last))
        return result 