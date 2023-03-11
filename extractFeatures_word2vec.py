# extract features using BOF and word to vec

#----------------------- Bag of words with tf-idf---------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import preprocessing
import pandas as pd 
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite.metrics import sequence_accuracy_score
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB

from nltk.corpus import stopwords
#from textblob import TextBlob

#from tashaphyne.stemming import ArabicLightStemmer
from nltk.stem.isri import ISRIStemmer
#import tashaphyne.arabic_const as arabconst 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


import gensim
import itertools
import arabicstopwords.arabicstopwords as stp
from tashaphyne import stopwords as tashstopwords
from nltk.corpus import stopwords as nltkstopwords
import arabicstopwords.arabicstopwords as stp
from nltk.stem import WordNetLemmatizer


def extractDigits(lst):
    return list(map(lambda el:[el], lst))

####################################### Create wordToVec.##############################
#stops = set(stopwords.words("arabic"))

""" df = pd.read_csv('/var/home/nhamad/NERproject/data/corpus.csv',encoding='utf8')
data = pd.DataFrame(df, columns = ['token', 'flat_tag'])

data['flat_tag'] = data['flat_tag'].replace(['O'], 0)
data['flat_tag'] = data['flat_tag'].replace(['B-PERS'], 1)
data['flat_tag'] = data['flat_tag'].replace(['I-PERS'], 2)
data['flat_tag'] = data['flat_tag'].replace(['B-ORG'], 3)
data['flat_tag'] = data['flat_tag'].replace(['I-ORG'], 4)
data['flat_tag'] = data['flat_tag'].replace(['B-LOC'], 5)
data['flat_tag'] = data['flat_tag'].replace(['I-LOC'], 6)

data=data.values.tolist()

# download word2vec using these 2 commands from website https://www.kaggle.com/code/baselsaleh/arabic-entity-named-recognition-using-word2vec
# wget --load-cookies /tmp/cookies.txt "https://bakrianoo.ewr1.vultrobjects.com/aravec/full_uni_cbow_300_twitter.zip" -O full_uni_cbow_300_twitter.zip && rm -rf /tmp/cookies.txt
# unzip full_uni_cbow_300_twitter.zip

model_w2v = gensim.models.Word2Vec.load('./full_uni_cbow_300_twitter.mdl')

count=0 # number of words not in wod2vec
data_vec=data
pca = PCA(n_components=6)
l2=[]
for sent in data:
    token=sent[0]
    if token in model_w2v.wv:
        xx=model_w2v.wv[token].size
        yy=model_w2v.wv[token]
        #vec = max(model_w2v.wv[token])
        vec=model_w2v.wv[token]
    else:
        vec=[0] * 300
        count+=1
    l1=[]
    l1.append(sent[1]) # store the label
    i=1 
    for jj in vec:
        l1.insert(i,jj)
        i+=1    
    l2.append(l1)
    

print(count)
data_word2vec=pd.DataFrame(l2)
data_word2vec.to_csv('/var/home/nhamad/NERproject/data/sample_word2vec.csv',sep=',', index=False,encoding='utf8') """

#***************************************** reduction the features******************************************
df = pd.read_csv('/var/home/nhamad/NERproject/data/sample_word2vec.csv',encoding='utf8')

corpus = pd.DataFrame(df)
corpus=corpus.drop(columns=['0'])

corpus=np.asarray(corpus)
corpus2=corpus.T # to rduce the dimesnion
pca = PCA(n_components=12)
X_pca = pca.fit(corpus2) 
print(X_pca.components_) # after reduction
cc=X_pca.components_

after_reduction=cc.T

data_word2vec=pd.DataFrame(after_reduction,columns=['f1', 'f2', 'f3','f4','f5','f6','f7','f8','f9','f10','f11','f12'])

df = pd.read_csv('/var/home/nhamad/NERproject/data/sample_word2vec.csv',encoding='utf8')

corpus = pd.DataFrame(df,columns =['0']) # take the label

data_word2vec['label']=corpus

data_word2vec.to_csv('/var/home/nhamad/NERproject/Features/word2vec_features.csv',sep=',', index=False,encoding='utf8')



