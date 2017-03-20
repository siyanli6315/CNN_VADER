#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import re
import os
os.chdir(“./vaderSentiment-master/vaderSentiment")
from vaderSentiment import sentiment as vaderSentiment
os.chdir("/Users/Lisiyan/Desktop")

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

f=open('neg_vader.txt','w')
for line in open("/Users/Lisiyan/Desktop/rt-polaritydata/rt-polaritydata/rt-polarity.neg"):
    line=line.strip()
    line=clean_str(line)
    tmp=vaderSentiment(line).values()
    f.write(str(tmp)+"\n")
f.close()

f=open('pos_vader.txt','w')
for line in open("/Users/Lisiyan/Desktop/rt-polaritydata/rt-polaritydata/rt-polarity.pos"):
    line=line.strip()
    line=clean_str(line)
    tmp=vaderSentiment(line).values()
    f.write(str(tmp)+"\n")
f.close()
