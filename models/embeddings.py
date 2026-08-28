#파일,. 폴더 등 접근 및 위치 관림
import os, shutil, sys
#인터넷에서파일 자동 다운로드 요청
import urllib.request
#압축파일 풀기
import zipfile, gzip
#인터넷을 오고 가는 요청 신호의 자료형 고정 -> fastAPI, pydantic
from typing import Dict, List, Optional, Tuple


import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# 사전학습 파일 다운로드
# ══════════════════════════════════════════════════════════════════════════════
#진행률 표시 함수
def _progress_hook(msg: str):
    """urllib 다운로드 진행률 콜백."""
    downloaded = [0]
    def hook(count, block_size, total_size):
        downloaded[0] += block_size
        if total_size > 0:
            pct = min(downloaded[0] / total_size * 100, 100)
            mb  = downloaded[0] / 1024 / 1024
            print(f'\r  {msg}: {pct:5.1f}%  ({mb:.0f} MB)', end='', flush=True)
    return hook

#임베딩 -> 모델

def download_glove(dim: int = 100, save_dir: str = '.') -> str:
    """
    Stanford GloVe 6B 파일을 다운로드하고 압축을 해제한다.

    파라미터  dim      : 50 / 100 / 200 / 300 중 선택
              save_dir : 저장 폴더
    반환      .txt 파일 경로
    크기      glove.6B.zip 약 822MB → 선택한 차원 .txt 만 추출
    """
    assert dim in (50, 100, 200, 300), "dim 은 50/100/200/300 중 하나"
    fname = f'glove.6B.{dim}d.txt'
    fpath = os.path.join(save_dir, fname)

    if os.path.exists(fpath):
        print(f'[GloVe] 이미 존재: {fpath}')
        return fpath

    os.makedirs(save_dir, exist_ok=True)
    zip_path = os.path.join(save_dir, 'glove.6B.zip')

    if not os.path.exists(zip_path):
        url = 'https://nlp.stanford.edu/data/glove.6B.zip'
        print(f'[GloVe] 다운로드 시작 (약 822MB)\n  URL: {url}')
        urllib.request.urlretrieve(url, zip_path,
                                   reporthook=_progress_hook('GloVe'))
        print()  # 줄바꿈

    print(f'[GloVe] 압축 해제 중 ({fname})...')
    with zipfile.ZipFile(zip_path) as z:
        z.extract(fname, save_dir)
    print(f'[GloVe] 완료: {fpath}')
    return fpath


def download_fasttext(save_dir: str = '.', max_words: int = 200_000) -> str:
    """
    fasttext.cc 영어 크롤링 벡터를 다운로드하고 gz 압축을 해제한다.

    파라미터  save_dir  : 저장 폴더
              max_words : 이후 로드 시 상위 N개만 사용 (메모리 절약 안내용)
    반환      .vec 파일 경로
    크기      cc.en.300.vec.gz 약 1.2GB → 압축 해제 후 약 7GB

    주의
      파일이 매우 크므로 네트워크·디스크 여유 공간을 확인하고 실행하세요.
      작은 파일이 필요하면 위키피디아 기반(약 6.5GB)을 쓰세요:
        https://fasttext.cc/docs/en/pretrained-vectors.html  →  wiki.en.vec
    """
    fname    = 'cc.en.300.vec'
    gz_fname = fname + '.gz'
    fpath    = os.path.join(save_dir, fname)
    gz_path  = os.path.join(save_dir, gz_fname)

    if os.path.exists(fpath):
        print(f'[FastText] 이미 존재: {fpath}')
        return fpath

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(gz_path):
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz'
        print(f'[FastText] 다운로드 시작 (약 1.2GB)\n  URL: {url}')
        print('  ※ 파일이 크므로 시간이 걸립니다.')
        urllib.request.urlretrieve(url, gz_path,
                                   reporthook=_progress_hook('FastText'))
        print()

    print(f'[FastText] gz 압축 해제 중 → {fpath}')
    print('  ※ 압축 해제 후 약 7GB가 필요합니다.')
    with gzip.open(gz_path, 'rb') as f_in, open(fpath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f'[FastText] 완료: {fpath}')
    return fpath

# ── 샘플 단어 정의 ─────────────────────────────────────────────────────────────
# 의미적으로 가까운 단어끼리 같은 그룹 → 임베딩 공간에서 얼마나 모이는지 확인
WORD_GROUPS: Dict[str, List[str]] = {
    '왕족'   : ['king', 'queen', 'prince', 'princess'],
    '국가'   : ['paris', 'france', 'london', 'england'],
    '동물'   : ['dog', 'cat', 'bird', 'fish'],
    '감정'   : ['happy', 'sad', 'angry', 'joy'],
    'IT기술' : ['computer', 'internet', 'data', 'algorithm'],
}

# 각 그룹에 할당할 색상 (plotly CSS 색상명)
GROUP_COLORS = {
    '왕족'   : '#E05A5A',   # 빨강 계열
    '국가'   : '#028090',   # 청록
    '동물'   : '#02C39A',   # 민트
    '감정'   : '#F4A261',   # 주황
    'IT기술' : '#065A82',   # 진파랑
}

# 단어 → 그룹명 역매핑
WORD_TO_GROUP: Dict[str, str] = {
    w: g for g, words in WORD_GROUPS.items() for w in words
}

ALL_WORDS: List[str] = [w for words in WORD_GROUPS.values() for w in words]

# #임베딩 -> 단어를 특정한 차원 개수만큼 숫자로 표현.
# 임베딩을 추출한다 -> 단어별 벡터를 찾아낸다
# 단어별 벡터는 모델에 따라 다양한 차원을 갖고 있을 것. 3차원으로 차원 축소
# 차원축소된 값들을 3D 시각화하면 단어끼리 얼마나 가까운지 볼 수 있음

#WORD_GROUP:Dict[str, List[str]]
#함수 이름(매개변수) -> 리턴값 자료형 명시
#1. 자료형 명시 -> 코드 자체가 문서처럼 어떤 값을 필요로 하는지 정확하게 표현
#2. 정적 분석 -> Extension(확장프로그램)으로 정적 분석 프로그램 사용 시, 버그 찾기 간편
def embed_random(words:List[str], embed_dim:int=64) -> Optional[np.ndarray]:
    #['a', 'b', 'c', 'd'] => 0, a/ 1, b/ 2, c ...
    #[a:0, b:1, c:2]
    vocab = {w:i for i, w in enumerate(words)}

    #임베딩 레이어를 통해 words를 embed_dim에 뿌리는 과정
    layer = nn.Embedding(len(words), embed_dim)
    # a->torch.tensor(1) / vocab[a] -> 1
    ids = torch.tensor([vocab[w] for w in words])
    with torch.no_grad():
        vectors = layer(ids).numpy()
    return vectors

#glove 임베딩 모델에서 내가 사전 정의한 words에 해당하는 벡터 값 추출
def embed_glove(words:List[str], path:str) -> Optional[np.ndarray]:

    if not os.path.exists(path):
        print(f'{path}에 파일이 존재하지 않음')
    #glove =  전체 단어에 대한 100개 차원의 사전 딕셔너리.
    glove = {}
    with open(path, encoding ='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            glove[parts[0]] = np.array(parts[1:], dtype = np.float32)

    #예외처리
    missing = [w for w in words if w not in glove]
    if missing:
        print(f'{w} 가 glove에 없습니다')
    #100dimension -> 몇 열로 이루어져 있는지
    dim = next(iter(glove.values())).shape[0]
    #딕셔너리.get(키, 디폴트)
    vectors = np.array([glove.get(w, np.zeros(dim)) for w in words])
    return vectors
#파일 읽어오기 -> 줄 단위로 자르고 -> 딕셔너리 만들고 -> 단어를 딕셔너리에서 찾고 -> return
def embed_fasttext(words:List[str], path:str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        print(f'{path} 경로가 존재하지 않습니다.')

    fasttext = {}
    with open(path, encoding='utf-8') as f:
        n_words, dim = map(int, f.readline().split())
        for i, line in enumerate(f):
            if i >= 200000:
                break
            parts = line.rstrip().split(' ')
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if vec.shape[0] == dim:
                fasttext[word] = vec

    missing = [w for w in words if w not in fasttext]
    if missing:
        print(f'{missing}이 fasttext 안에 존재하지 않음')

    vectors = np.array([fasttext.get(w, np.zeros(dim)) for w in words])
    return vectors
    
def embed_bert(words:List[str]) -> Optional[np.ndarray]:

    from transformers import AutoTokenizer, BertModel

    #Glove,Fastext의 파일 읽어 가져오기와 동일
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    word_embed = bert.embeddings.word_embeddings

    #words에 맞는 벡터를 임베딩 뭉치에서 추출
    vectors = []
    for word in words:
        token_ids = tokenizer.encode(word, add_special_tockens =False)

        #버트 모델이 갖고 있지 않은 단어 뭉치인 경우
        if not token_ids:
            #np.zeros -> 내가 알지 못하는 단어의 벡터는 임베딩 모델 존재하지 않음
            # 빈 값(null) 만들지 않기 위해 임시 0으로 채워진 값을 return. (버트 임베딩 모델의 차원의 수)
            vectors.append(np.zeros(768))
            continue
        with torch.no_grad():
            ids_tensor = torch.tensor(token_ids)
            embed = word_embed(ids_tensor).mean(0).numpy()

        vectors.append(embed)
    return np.array(vectors)

def decomposition_3d(vector:np.ndarray) -> np.ndarray:
    #pca 이용, 3차원으로 차원축소
    from sklearn.decomposition import PCA
    #정규화: 각 벡터별로 데이터의 범위가 다를 수 있으니
    #norm = 각 벡터를 줄 별로(axis=1) 계산해서 각 단어 줄에 대한 L2 norm 구하기
    norm = np.linalg.norm(vector, axis=1, keepdims=True) + 1e-9     #가장 작은 수를 붙여서 결과값을 0으로 만들지 않기 위함
    vector = vector/norm

    pca = PCA(n_components=3)
    result = pca.fit_transform(vector)

    #3차원으로 잘 반환했는지?
    explain = pca. explained_variance_ * 100
    print(f'PCA가 이 모델을 {explain.sum():.2f}% 설명 가능함')
    
    return result

if __name__ == '__main__':
    g_path = download_glove(dim=100, save_dir='.')
    f_path = download_fasttext(save_dir='.')

    #각 방법의 embedding 추출
    word = ALL_WORDS
    #1. random
    random_emb = embed_random(word)
    #2. Glove
    glove_emb = embed_glove(word,r'C:\Users\user\Desktop\Git\NLP_practice\models\glove.6B.100d.txt' )
    #3. FastText
    ft_emb = embed_fasttext(word, r'C:\Users\user\Desktop\Git\NLP_practice\models\cc.en.300.vec')
    #4. BERT
    bert_emb = embed_bert(word)