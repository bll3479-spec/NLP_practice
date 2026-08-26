#RNN, LSTM, GRU
# REsNet- > (residual)잔차, 구조가 특징
# RNN -> Recurrent 구조

# RNN, LSTM, GRU 만들때도 -> 토큰화 + vocab 거쳐야함
#urllib.request: 인터넷 자료 다운로드 라이브러리
import os, re, urllib.request, zipfile, torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
MAX_LEN = 50
#훈련용 클래스 생성: 자연어 모델을 위한 커스텀 데이터셋 만들기
class SpamDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        #데이터 + 라벨 필요
        #text의 vocab기반 정수 전환
        self.texts = df['text'].tolist()
        #labels의 tensor 전환 필요. dtype=int64
        self.labels = torch.tensor(df['label'].tolist(), dtype = torch.long)
        self.vocab = vocab
        self.max_len = MAX_LEN

        #MAX_LEN: 최대 문장 길이를 알아야 padding 가능
    def _text_to_tensor(self, text, vocab, max_len = MAX_LEN):
        # text를 tokenize -> t, t를 vocab 딕셔너리에서 get(키) -> 값으로 부름. get(2, 1)의 경우 1은 unk로 보내주세요(디폴트 설정)
        sample = [vocab.get(t, 1) for t in tokenize(text)]

        #길이 세팅. 샘플의 길이가 max_len을 넘으면 걍 자르는 것.(max_len: 얻은 데이터의 평균 길이로 정하기)
        if len(sample) >= max_len:
            sample = sample[:max_len]
        #패딩: 최대 길이에서 지금 샘플 길이를 뺀 뒤, 최대 길이 맞추도록 0을 더하는 것
        else:
            sample +=[0] * (max_len - len(sample))

        return torch.tensor(sample, dtype = torch.long)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        text = self._text_to_tensor(self.texts[index], self.vocab, self.max_len)
        label = self.labels[index]
        return text, label

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

    dataset = SpamDataset(df, vocab=vocab, max_len=MAX_LEN)
    n_total = len(dataset)
    n_train = int(n_total * 0.7)
    n_valid = int(n_total * 0.15)
    n_test = n_total - n_train - n_valid

    train_set, valid_set, test_set = random_split(dataset, [n_train, n_valid, n_test])

    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, valid_loader, test_loader, vocab

def train(model, train_loader, valid_loader, criterion, optimizer,
          num_epochs, device, model_name='Model'):
    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'valid_acc': []}
    best_valid_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            output = model(X_batch)
            loss   = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(output, 1)
            correct += (pred == y_batch).sum().item()
            total   += y_batch.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc  = correct / total * 100

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, pred = torch.max(model(X_batch), 1)
                v_correct += (pred == y_batch).sum().item()
                v_total   += y_batch.size(0)
        valid_acc = v_correct / v_total * 100

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

        if (epoch + 1) % 2 == 0:
            print(f'[{model_name}] Epoch {epoch+1:2d}/{num_epochs} | '
                  f'loss: {train_loss:.4f} | train: {train_acc:.2f}% | valid: {valid_acc:.2f}%')

    print(f'[{model_name}] 최고 검증 정확도: {best_valid_acc:.2f}%\n')
    return history

def evaluate(model, test_loader, device, model_name='Model'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            _, pred = torch.max(model(X_batch.to(device)), 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    return all_labels, all_preds

def text_to_tensor(text, vocab, max_len=MAX_LEN):
    sample = [vocab.get(t, 1) for t in tokenize(text)]

    if len(sample) >= max_len:
        sample = sample[:max_len]
    else:
        sample += [0] * (max_len - len(sample))
    return torch.tensor(sample, dtype=torch.long)

def predict(model, text, vocab, device):
    model.eval()
    tensor = text_to_tensor(text, vocab, 50).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch. softmax(model(tensor), dim=1)[0]
    label = 'SPAM' if prob[1] > 0.5 else 'HAM'
    print(f'입력 텍스트 {text} \n 판정 결과 {label} \n 신뢰도 {prob[1] * 100:.2f}%')



#0826복습문제: LSTM 클래스 생성
class SpamLSTM(nn.Module):
    def __init__(self):
        self.embedding
        self.lstm
        self.dropout
        self.fc

    def forward (self, x):
        embedding = self.embedding(x)
        out, _ = self.lstm(embedding)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))

from models.rnn import SpamRNN
import torch.nn as nn
import torch.optim as optim
if __name__ == '__main__':
    train_loader, valid_loader, test_loader, vocab = preprocessing()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_size = len(vocab)

    #훈련 코드
    #1. RNN.py의 rnn 가져오기
    
    #2. train에 입력 -> criterion(crossentropy), Adam 사용
    model = SpamRNN(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    num_epochs = 30

    train(model, train_loader, valid_loader, criterion, optimizer,
          num_epochs, device, model_name='Model')
    #3. evaluate에 입력
    evaluate(model, test_loader, device, model_name='Model')
    #한 줄의 문장으로 추론
    text = input('검증할 문장을 넣어주세요 : \n')
    predict(model, text, vocab, device)

    
    
    # x_train, y_train = next(iter(train_loader))
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_train[0])
    # print(y_train[0])