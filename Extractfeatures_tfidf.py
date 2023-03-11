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

# for remosing a stop words
def remove_stop_words(corpus,stop_words):
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
        
    return results


#stopWords = open("../input/aranizi-dailect-training-data/arabicstopwords.txt").read().splitlines()
""" print('Number of Arabic stopwords in arabicstopwords package: ',len(stp.STOPWORDS.keys()))
df_train = pd.read_csv('/var/home/nhamad/NERproject/data/train80.csv',encoding='utf8')
train_data = pd.DataFrame(df_train, columns = ['token', 'flat_tag'])
df_test = pd.read_csv('/var/home/nhamad/NERproject/data/test20.csv',encoding='utf8')
test_data = pd.DataFrame(df_test, columns = ['token', 'flat_tag'])
train_data=train_data.values.tolist()
test_data=test_data.values.tolist()
#stops = stp.STOPWORDS.keys()
#vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=.01, max_df=.3)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_data)
xx=X.todense()
cc=np.asarray(xx)
cols=cc.shape[1]
i=0
j=0
ll=[]
for u in cc:
    xx=cc[:,i]
    yy=xx.reshape(1,-1)
    ff=yy[0]
    ll.append(yy[0])
    i+=1
    if(i==cols):
        break

Y=vectorizer.get_feature_names_out()
print(X.shape) """
#*****************************************************************************************

#example 
""" corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

vectorizer = TfidfVectorizer()   #TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=.01, max_df=.3)
tf_idf_vector = vectorizer.fit_transform(corpus) 

tf_idf_array = tf_idf_vector.toarray()

pca = PCA(n_components=2) # make a 
X_pca = pca.fit(tf_idf_array) 
print(X_pca.components_) # after reduction
cc=X_pca.components_
words_set=vectorizer.get_feature_names_out()

cc2=cc.T
print(cc2) """

#********************************Extract features using tf-idf******************************************

# this setion is used to extract the features using tf-idf, then reduce the dim to 12 dim, then store each feature with each token

""" file1 = open('/var/home/nhamad/NERproject/sentences.txt', 'r')
corpus = file1.readlines()

vectorizer = TfidfVectorizer()
tf_idf_vector = vectorizer.fit_transform(corpus)
tf_idf_array = tf_idf_vector.toarray()
print("**************************************************")
cols=tf_idf_vector.shape[1]
print("**************************************************")

Y=vectorizer.get_feature_names_out()
print (Y)
pca = PCA(n_components=12)
X_pca = pca.fit(tf_idf_array) 
print(X_pca.components_) # after reduction
cc=X_pca.components_

yy=cc.T
print("**************************************************")
print (yy)
testsample = pd.DataFrame(columns=['f1', 'f2', 'f3','f4','f5','f6','f7','f8','f9','f10','f11','f12'])
testsample=pd.DataFrame(yy)
testsample['token']= Y # tokens in the corpus getting from the vectroizer 
testsample.to_csv('/var/home/nhamad/NERproject/data/sample_tfidf.csv',sep=',', index=False,encoding='utf8') """
#---------------------------------------------------------------------------------------
#------------------ Replace the token with its features-------------------------
#---------------------------------------------------------------------------------------

#
corpus = pd.read_csv('/var/home/nhamad/NERproject/data/corpus.csv',encoding='utf8') #
corpus = pd.DataFrame(corpus)
corpus=np.asarray(corpus)

sample_tfidf = pd.read_csv('/var/home/nhamad/NERproject/data/sample_tfidf.csv',encoding='utf8') #
sample_tfidf = pd.DataFrame(sample_tfidf)

sample_tfidf=np.asarray(sample_tfidf)


list2=[]
tf_idf_features = pd.DataFrame(columns=['f1', 'f2', 'f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','label'])
for uu in corpus:
    g=uu[5]
    result =np.where(uu[5] == sample_tfidf[:,12]) # the index of token that is in sample_tfidf to get the features
    row=sample_tfidf[result,:12]
    r1=row[0]
    
    if(r1.size==0):
        continue
    
    r2=r1[0]
    list1=[]
    list1.append(uu[6]) # store the token from the corpus
    i=1
    for jj in r2:
        list1.insert(i,jj)
        i+=1
    list2.append(list1)    
    """ ll.append(r2)
    ll.extend(uu[6])
    ll.append(r2)
    test = pd.DataFrame(ll) """

    #uu[5]=sample_tfidf[result,:6]
    
array2=np.asarray(list2)
tf_idf_features = pd.DataFrame(array2,columns=['label','f1', 'f2', 'f3','f4','f5','f6','f7','f8','f9','f10','f11','f12'])

tf_idf_features.to_csv('/var/home/nhamad/NERproject/Features/tf_idf_features.csv',sep=',', index=False,encoding='utf8')

print("**************************************************")
