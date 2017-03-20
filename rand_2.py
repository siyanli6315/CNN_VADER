#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import re
import numpy as np
import itertools
from collections import Counter

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def pad_sentences(sentences, padding_word="<PAD/>"):
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

neg=list(open("/Users/Lisiyan/Desktop/rt-polaritydata/rt-polaritydata/rt-polarity.neg").readlines())
pos=list(open("/Users/Lisiyan/Desktop/rt-polaritydata/rt-polaritydata/rt-polarity.pos").readlines())
neg=[s.strip() for s in neg]
pos=[s.strip() for s in pos]
x_text=pos+neg
x_text=[clean_str(sent) for sent in x_text]
x_text=[s.split(" ") for s in x_text]
x_text=pad_sentences(x_text)
word_counts = Counter(itertools.chain(*x_text))
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
x = np.array([[vocabulary[word] for word in sentence] for sentence in x_text])
neg_labels=[[0,1] for _ in neg]
pos_labels=[[1,0] for _ in pos]
y=np.concatenate([pos_labels,neg_labels], 0)
graph_in = Input(shape=(56,20))
convs=[]
for i in [3,4]:
    conv = Convolution1D(nb_filter=150,
                        filter_length=i,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1)(graph_in)
    pool = MaxPooling1D(pool_length=2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)
out = Merge(mode='concat')(convs)
graph = Model(input=graph_in, output=out)

np.random.seed(2)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices].argmax(axis=1)
model = Sequential()
model.add(Embedding(input_dim=len(vocabulary),output_dim=20,input_length=56))
model.add(Dropout(0.25,input_shape=(56,20)))
model.add(graph)
model.add(Dense(150))
model.add(Dropout(0.5))
model.add(Activation("relu"))
model.add(Dense(4))
model.add(Dropout(0.25))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
model.fit(x_shuffled, y_shuffled, batch_size=32,nb_epoch=50,shuffle=True,validation_split=0.1,verbose=2)
