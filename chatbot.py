import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data =json.load(file)
try:
    with open("data.pickle","rb")as f:
        words,label,training,out=pickle.load(f)
except:
    words=[]
    labels=[]
    xdocs=[]
    ydocs=[]

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds=nltk.word_tokenize(pattern)
            words.extend(wrds)
            xdocs.append(wrds)
            ydocs.append(intent["tag"])
            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words=[stemmer.stem(w.lower()) for w in words if w not in "?"]
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

    training=numpy.array(training)
    out=numpy.array(out)

    with open("data.pickle","wb")as f:
        pickle.dump((words,label,training,out),f)

#good shit
tensorflow.reset_default_graph()
net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(out[0]),activation="softmax")
net=tflearn.regression(net)

model=tflearn.DNN(net)
model.fit(training,out,n_epoch=1001,batch_size=8,show_metric=True)
model.save("model.tflearn")
