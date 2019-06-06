#!/usr/bin/env python
# coding: utf-8

# In[1]:


# CNN, LSTM 등 예측 모델에 집어 넣어 학습 시킬 때 시각화 시켜주는 코드

import gensim
import tensorflow as tf
import codecs
import os
import numpy as np


# In[4]:


## 모델 불러오기

model = gensim.models.word2vec.Word2Vec.load('Word2Vec_Kor')
model.wv.most_similar('공포/Noun', topn = 20)
max_size = len(model.wv.vocab)-1
w2v = np.zeros((max_size, model.trainables.layer1_size))


# In[5]:


with codecs.open("metadata.tsv",'w+',encoding='utf8') as file_metadata:
    for i,word in enumerate(model.wv.index2word[:max_size]):
        w2v[i] = model.wv[word]
        file_metadata.write(word + "\n")


# In[6]:


from tensorflow.contrib.tensorboard.plugins import projector


# In[10]:


sess = tf.InteractiveSession()
# 임베딩이 된 tensor 객체를 생성
with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v, trainable= False, name = 'embedding')
    
tf.global_variables_initializer().run()

path = 'word2vec'

saver = tf.train.Saver()
writer = tf.summary.FileWriter(path, sess.graph)

# 프로젝트에 추가
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = 'embedding'
embed.metadata_path = 'YourPath' # 이 곳에 tensor file이 저장될 실제 주소를 입력
projector.visualize_embeddings(writer, config)
saver.save(sess, path+'/model.ckpt', global_step = max_size)

