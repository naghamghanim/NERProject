import datetime
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite.metrics import sequence_accuracy_score
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.model_selection import cross_val_predict
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.metrics import  accuracy_score
from sklearn import metrics
import numpy as np
from itertools import chain
from sklearn.metrics import *
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer



class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [( t,id,f) for  t, id, f in
                              zip(s["token"].values.tolist(),s["word_id"].values.tolist(),s["flat_tag"].values.tolist())]
        self.grouped = self.data.groupby("global_sentence_id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None




pathTrain="/var/home/nhamad/NERproject/draft_data/train80.csv"
pathTest="/var/home/nhamad/NERproject/data/test20.csv"


TRdata = pd.read_csv(pathTrain, encoding='utf-8') 

TEdata = pd.read_csv(pathTest, encoding='utf-8')  

df = pd.DataFrame(TRdata, columns = ['token'])
count=[]
count=df['token'].value_counts().tolist() # return the number of occurences for each token

tokens=[]
tokens=df['token'].value_counts().index.tolist() # here return the tokens in the corpus


conc=np.vstack((tokens,count)) # concatenate the number of occorences with each token


words = list(set(TRdata["token"].values))
TR_words = len(words)
print("Number of Different Words :" + "n_words")
print(TR_words) # 48432


# Count the number of uniqe "Labels" of training data...

TAG = list(set(TRdata["flat_tag"].values))
TR_tag = len(TAG)
print("Number of Different NER tags :" + "n_tag")
print(TR_tag) # 7


TRgetter = SentenceGetter(TRdata)
TEgetter = SentenceGetter(TEdata)
sent = TRgetter.get_next()
#print("SENT: {}".format(sent))

TRsentences = TRgetter.sentences
TEsentences = TEgetter.sentences

# Features extraction...

def word2features(sent, i):
    word = sent[i][0]
    word_id = sent[i][1]
    #no_word= "لايوجد"
   

    features = {'bias': 1.0, 'word': word, 'word_id': word_id, } # first 3 features
    
    if i > 0: # the pre word
        word1 = sent[i - 1][0]
        word_id1 = sent[i - 1][1]
        features.update({'-1:word': word1, '-1:word_id': word_id1, })
    else:
        #features['BOS'] = True # here the first word
        features['BOS'] =True
       

    if i < len(sent) - 1: # the next word
        word1 = sent[i + 1][0]
        word_id1 = sent[i + 1][1]
        features.update({'+1:word': word1, '+1:word_id': word_id1,  })
    else:
        features['EOS'] = True
        
#    [idword, word,BOS=1, wordid+1, word+1 ]
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [flat_tag for word, word_id, flat_tag in sent]


def sent2tokens(sent):
    return [token for token, word_id, flat_tag in sent]


X_tr = [sent2features(s) for s in TRsentences]
y_tr = [sent2labels(s) for s in TRsentences]

X_te = [sent2features(s) for s in TEsentences]
y_te = [sent2labels(s) for s in TEsentences]

print("Train", len(y_tr)) # 18179
print("T_test", len(y_te))# 6050


# here we have to try several paramters
""" 'lbfgs' - Gradient descent using the L-BFGS method
'l2sgd' - Stochastic Gradient Descent with L2 regularization term
'ap' - Averaged Perceptron
'pa' - Passive Aggressive (PA)
'arow' - Adaptive Regularization Of Weight Vector (AROW) """


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_tr, y_tr)
labels = list(crf.classes_)
#labels.remove('O')
y_pred = crf.predict(X_te)

print(classification_report(y_te, y_pred))
print(f1_score(y_te, y_pred))


 



