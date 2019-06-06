#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 필요한 객체 import
import pandas as pd
from konlpy.tag import Okt


# In[2]:


#!/usr/bin/env python
#-*- coding:utf-8 -*-

# 한글 영화 리뷰 데이터를 불러옴

train = pd.read_csv('~/data/ratings_train.txt', 
                    header=0, delimiter='\t', quoting=3,encoding='CP949')
test = pd.read_csv('~/data/ratings_test.txt', 
                   header=0, delimiter='\t', quoting=3,encoding='CP949')

print(train.shape)

print(train['document'].size)

print(test.shape)

print(test['document'].size)

train.head(10)


# In[3]:


pos_tagger = Okt()

# 품사 태깅 및 토큰화를 위한 함수

def tokenize(doc):
    # norm, stem은 optional
    if type(doc) is not str:
        return []
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

tokenized_list = []

# 전체 training data set 15만 개

for i in range(0,150000) :
    if i%10000 == 0: # 10000 개의 document마다 진행상황 표시
        print(i,'\n')
    review = tokenize(train["document"][i])
    tokenized_list.append(review) # 토큰화 된 리뷰를 리스트에 담아줌
    
print(tokenized_list)

#train_docs = [(tokenize(["document"]), row[2]) for row in train]
#test_docs = [(tokenize(row[1]), row[2]) for row in test]
# 잘 들어갔는지 확인
# => [(['아/Exclamation',
#   '더빙/Noun',
#   '../Punctuation',
#   '진짜/Noun',
#   '짜증/Noun',
#   '나다/Verb',
#   '목소리/Noun'],
#  '0')]


# In[4]:


# 품사 태깅된 단어들이 잘 tokenize 되었는지 확인
print(tokenized_list[:10])


# In[5]:


import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


# In[6]:


#하이퍼 파라메터 값 지정
num_features = 200 # 문자 벡터 차원수
min_word_count = 40 # 최소 문자 수
num_workers = 4 # 병렬 쓰레드 수
context = 10 # window size
downsampling = 1e-3 # 문자 빈도 수 downsample

# 초기화 및 모델 학습
from gensim.models import word2vec

#모델 학습
model = word2vec.Word2Vec(tokenized_list,
                                                  workers = num_workers,
                                                  size = num_features,
                                                  min_count = min_word_count,
                                                  window = context,
                                                  sample = downsampling)


# In[7]:


model


# In[8]:


# 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
model.init_sims(replace=True)

model_name = 'Word2Vec_Kor'
# model_name = '300features_50minwords_20text'
model.save(model_name)
model.wv.save_word2vec_format('my.embedding', binary=False)


# In[15]:


from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('my.embedding', binary=False, encoding='utf-8')


# In[53]:


# 참고 https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import gensim 
import gensim.models as g

# 그래프에서 한글 깨지는 문제에 대한 대처
#font_name = font_manager.FontProperties(fname = "/Users/LEE/Library/Fonts/BMYEONSUNG.ttf").get_name()
rc('font', family = 'AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

model_name = 'Word2Vec_Kor'
model = g.Doc2Vec.load(model_name)

vocab = list(model.wv.vocab)
X = model[vocab]

print(len(X))
print(X[0][:10])
tsne = TSNE(n_components=2)

# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(X[:100,:])
# X_tsne = tsne.fit_transform(X)


# In[51]:


import matplotlib.font_manager as fm
font_list_mac = fm.OSXInstalledFonts()
print(len(font_list_mac))
print(font_list_mac)
[(f.name,f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]


# In[41]:


df = pd.DataFrame(X_tsne, index=vocab[:100], columns = ['x','y'])
df.shape


# In[42]:


# 한글 임베딩 된 단어들 확인
df.head(15)


# In[54]:


fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.show()


# In[14]:


print(model.vocabulary.raw_vocab)


# In[ ]:


model.save('Word2vecKor.model')


# In[ ]:


model.most_similar('팝콘/Noun')

