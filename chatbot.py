import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    data =json.load(file)

words=[]
labels=[]
xdocs=[]
ydocs=[]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds=nltk.word_tokeniza(pattern)
        words.extend(wrds)
        xdocs.append(pattern)
        ydocs.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
words=[stemmer.stem(w.lower()) for w in words]
words=sorted(list(words))
labels=sorted(labels)

training=[]
out=[]
outEmpty=[0 for _ in range(len(labels))]
for x,doc in enumerate(xdocs):
    bag=[]
    wrds=[stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    outRow=outEmpty[:]
    outRow[labels.index(ydocs[x])]=1
    training.append(bag)
    out.append(outRow)
training=numpy.array(out)

