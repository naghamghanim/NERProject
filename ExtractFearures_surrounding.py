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
        self.grouped = self.data.groupby("sentence_id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None




pathTrain="/var/home/nhamad/NERproject/wojood_v1_validation_70.csv"
pathTest="/var/home/nhamad/NERproject/wojood_v1_validation_20.csv"


TRdata = pd.read_csv(pathTrain, encoding='utf-8') 

TEdata = pd.read_csv(pathTest, encoding='utf-8')  

df1 = pd.DataFrame(TRdata, columns = ['token'])
df2 = pd.DataFrame(TEdata, columns = ['token'])

result = pd.concat([df1, df2], ignore_index=True, sort=False)

count=[]
count=result['token'].value_counts().tolist()

tokens=[]
tokens=result['token'].value_counts().index.tolist()


conc=np.vstack((tokens,count))


words = list(set(TRdata["token"].values))
TR_words = len(words)
print("Number of Different Words :" + "n_words")
print(TR_words) # 44562


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
    no_word= -1
    #no_word="لا يوجد"
   

    features = {'bias': 1.0, 'word': word, 'word_id': word_id, } # first 3 features
    
    if i > 0: # the pre word
        features['BOS/EOS']=0  # not BOS not EOS
        word1 = sent[i - 1][0]
        word_id1 = sent[i - 1][1]
        features.update({'preword': word1, 'preword_id': word_id1, })
    else:
        #features['BOS'] = True # here the first word
        features['BOS/EOS'] = 1 # BOS
        word1 = no_word
        word_id1 = -1
        features.update({'preword': word1, 'preword_id': word_id1, })

    if i < len(sent) - 1: # the next word
        word1 = sent[i + 1][0]
        word_id1 = sent[i + 1][1]
        features.update({'nextword': word1, 'nextword_id': word_id1,  })
    else:
        features['BOS/EOS'] = 2 #EOS
        word1 = no_word
        word_id1 = +1
        features.update({'nextword': word1, 'nextword_id': word_id1,  })

#    [idword, word,BOS=1, wordid+1, word+1 ]
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [flat_tag for word, word_id, flat_tag in sent]


def sent2tokens(sent):
    return [token for token, word_id, flat_tag in sent]

#******************************************************get the features of test and train data
X_tr = [sent2features(s) for s in TRsentences]
y_tr = [sent2labels(s) for s in TRsentences]

""" 
print("Train", len(y_tr))
print("T_test", len(y_te))
i=0
j=0
features=[]
for x in X_tr:
    xx=np.asarray(X_tr[i])
    for xxx in xx:
        yy=xx[j]
        features.append(yy)
        j=j+1
    j=0
    i=i+1   
    
i=0
j=0
labels_tr=[]
for x in y_tr:
    xx=np.asarray(y_tr[i])
    for xxx in xx:
        yy=xx[j]
        labels_tr.append(yy)
        j=j+1
    j=0
    i=i+1       
      
train={}
i=0
jj=0
y=0
for x in features:
    xx=features[i]
    #for j in xx:
    for k in xx:
        tmp = str(k)
            #tmp = str(i)
        if (y<8):
            train[tmp] = [xx[k]]
            y=y+1
        else:
            train[tmp].append(xx[k])
    i=i+1
        
train2=pd.DataFrame(train)
label_tr = pd.DataFrame (labels_tr, columns = ['label'])

train2['label']= label_tr
train2.to_csv('/var/home/nhamad/NERproject/train2.csv',sep=',', index=False,encoding='utf8') """
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

X_te = [sent2features(s) for s in TEsentences]
y_te = [sent2labels(s) for s in TEsentences]

i=0
j=0
features=[]
for x in X_te:
    xx=np.asarray(X_te[i])
    for xxx in xx:
        yy=xx[j]
        features.append(yy)
        j=j+1
    j=0
    i=i+1   
    
i=0
j=0
labels_te=[]
for x in y_te:
    xx=np.asarray(y_te[i])
    for xxx in xx:
        yy=xx[j]
        labels_te.append(yy)
        j=j+1
    j=0
    i=i+1       
      
test={}
i=0
jj=0
y=0
for x in features:
    xx=features[i]
    #for j in xx:
    for k in xx:
        tmp = str(k)
            #tmp = str(i)
        if (y<8):
            test[tmp] = [xx[k]]
            y=y+1
        else:
            test[tmp].append(xx[k])
    i=i+1
        
test2=pd.DataFrame(test)
label_te = pd.DataFrame (labels_te, columns = ['label'])

test2['label']= label_te
test2.to_csv('/var/home/nhamad/NERproject/test2.csv',sep=',', index=False,encoding='utf8')

#************************************************************
allLabels=["O", "B-PERS", "I-PERS", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
 # 0, 1, 2, 3, 4, 5, 6
 
#***********convert the labels of train and test data into dataframe*******************
i=0
j=0
labels_tr=[]
for x in y_tr:
    xx=np.asarray(y_tr[i])
    for xxx in xx:
        yy=xx[j]
        labels_tr.append(yy)
        j=j+1
    j=0
    i=i+1   

label_tr = pd.DataFrame (labels_tr, columns = ['label'])

#training data: X_tr
# trianing labels: labels_tr

#testing data: data: X_te
#



print(label_tr)


""" label_tr['label'] = label_tr['label'].replace(['O'], 0)
label_tr['label'] = label_tr['label'].replace(['B-PERS'], 1)
label_tr['label'] = label_tr['label'].replace(['I-PERS'], 2)
label_tr['label'] = label_tr['label'].replace(['B-ORG'], 3)
label_tr['label'] = label_tr['label'].replace(['I-ORG'], 4)
label_tr['label'] = label_tr['label'].replace(['B-LOC'], 5)
label_tr['label'] = label_tr['label'].replace(['I-LOC'], 6) """

#print(label_tr)

i=0
j=0
labels_te=[]
for x in y_te:
    xx=np.asarray(y_te[i])
    for xxx in xx:
        yy=xx[j]
        labels_te.append(yy)
        j=j+1
    j=0
    i=i+1       
""" label_te = pd.DataFrame (labels_te, columns = ['label'])
label_te['label'] = label_te['label'].replace(['O'], 0)
label_te['label'] = label_te['label'].replace(['B-PERS'], 1)
label_te['label'] = label_te['label'].replace(['I-PERS'], 2)
label_te['label'] = label_te['label'].replace(['B-ORG'], 3)
label_te['label'] = label_te['label'].replace(['I-ORG'], 4)
label_te['label'] = label_te['label'].replace(['B-LOC'], 5)
label_te['label'] = label_te['label'].replace(['I-LOC'], 6)
 """
 
""" for index, item in enumerate(labels_te):
    if (labels_te[index]=='O'):
        labels_te[index] = 0
    if (labels_te[index]=='B-PERS'):
        labels_te[index] = 1
    if (labels_te[index]=='I-PERS'):
        labels_te[index] = 2        
    if (labels_te[index]=='B-ORG'):
        labels_te[index] = 3    
    if (labels_te[index]=='I-ORG'):
        labels_te[index] = 4
    if (labels_te[index]=='B-LOC'):
        labels_te[index] = 5
    if (labels_te[index]=='I-LOC'):
        labels_te[index] = 6        """      
#l = list(map(lambda x: x.replace('O',0), labels_te))

label_te=np.asarray(labels_te,dtype=object)
#print(label_te)
#*******************************************replace the string tokens into numbers 
i=0
j=0
features=[]
for x in X_tr:
    xx=np.asarray(X_tr[i])
    for xxx in xx:
        word=xx[j]['word']
    # find the word in array
        result = np.where(conc == word) # get the index
        rr=result[1][0] #take the index
        coun=conc[1][rr]
        coun=int(coun)
        yy=xx[j]
        xx[j]['word']=coun
        if(j<len(xx)-1):
            if(xx[j+1]['preword']==word):
               xx[j+1]['preword']=coun
        if (j!=0):    
            if (xx[j-1]['nextword']==word):
                xx[j-1]['nextword']=coun
            
        features.append(yy)
        j=j+1
    j=0
    i=i+1   
      
    
i=0
j=0

features_te=[]
for x in X_te:
    xx=np.asarray(X_te[i])
    for xxx in xx:
        word=xx[j]['word']
                    # find the word in array
        result = np.where(conc == word) # get the index
        rr=result[1][0] #take the index
        coun=conc[1][rr]
        coun=int(coun)
        yy=xx[j]
        xx[j]['word']=coun
        if(j<len(xx)-1):
            if(xx[j+1]['preword']==word):
               xx[j+1]['preword']=coun
        if (j!=0):    
            if (xx[j-1]['nextword']==word):
                xx[j-1]['nextword']=coun
            
        features_te.append(yy)
        j=j+1
    j=0
    i=i+1
    
#*************************conver train into dataframe**********************
train={}
i=0
jj=0
y=0
for x in features:
    xx=features[i]
    #for j in xx:
    for k in xx:
        tmp = str(k)
            #tmp = str(i)
        if (y<8):
            train[tmp] = [xx[k]]
            y=y+1
        else:
            train[tmp].append(xx[k])
    i=i+1
        
train2=pd.DataFrame(train)
train2['label']= label_tr
#train2.to_csv('/var/home/nhamad/NERproject/train.csv',sep=',', index=False,encoding='utf8')

print(train2)  

test={}
i=0
jj=0
y=0
for x in features_te:
    xx=features_te[i]
    for k in xx:
        tmp = str(k)
        #tmp = str(i)
        if (y<8):
            test[tmp] = [xx[k]]
            y=y+1
        else:
            test[tmp].append(xx[k])
    i=i+1  

test2=pd.DataFrame(test)
test2['label']= label_te

#test2.to_csv('/var/home/nhamad/NERproject/test.csv',sep=',', index=False,encoding='utf8')

print(test2)

#********************* up to here store the features into dataframe*********************************



k = 24
print("k={}".format(k))
knn = KNeighborsClassifier(n_neighbors=k)
print("start train \n")
logger.info("\n***** start train KNN *****")

knn.fit(train2, label_tr)
print("start predict \n")
logger.info("\n***** start predict KNN *****")

predicted = knn.predict(test2)

acc = accuracy_score(list(label_te), predicted)

logger.info("\n***** Accuracy of KNN= {}".format(acc))

#print("Accuracy of KNN:", acc)

#print(classification_report(label_te,  predicted))
result = confusion_matrix(list(label_te), predicted)

print(result)
#train2: dataframe
#lavel_tr: dataframe
clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(train2, label_tr)
y_pred=clf.predict(test2)

for index, item in enumerate(y_pred):
    if (y_pred[index]=='O'):
        y_pred[index] = 0
    if (y_pred[index]=='B-PERS'):
        y_pred[index] = 1
    if (y_pred[index]=='I-PERS'):
        y_pred[index] = 2        
    if (y_pred[index]=='B-ORG'):
        y_pred[index] = 3    
    if (y_pred[index]=='I-ORG'):
        y_pred[index] = 4
    if (y_pred[index]=='B-LOC'):
        y_pred[index] = 5
    if (y_pred[index]=='I-LOC'):
        y_pred[index] = 6 

clf.score(list(label_te),y_pred)      
    

#features = np.array(features)

# replace with a words with a numbers in array
# features: 
# conc: list of tokens and their count
# replace features[word]
i=0

df = pd.DataFrame(features)

""" for xx in features:
    # find the word in the feature
    word=features[i]['word']
    
    # find the word in array
    result = np.where(conc == word) # get the index
    rr=result[1][0] #take the index
    coun=conc[1][rr] # take the number of occerneces
    #Result = np.where([features[x]['word']  for x, word2 in range(0,len(features)-1)] == word, coun, features)
    i=i+1
    
    if(coun==1):
        features[i]['word']=int(coun)
     #now, replace the word with it's count
    else:
        ii=0
        for yy in features:
            if(features[ii]['word']==word ):
                features[ii]['word']=coun
            ii=ii+1  
     """


""" crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
features = np.array(features)
labels_tr = np.array(labels_tr)
crf.fit(features, labels_tr)
labels = list(crf.classes_)
labels.remove('O')

y_pred = crf.predict(features_te)
#f1=metrics.f1_score(y_te, y_pred, average='weighted', labels=labels)
print(metrics.flat_f1_score(labels_te, y_pred,average='weighted', labels=labels))

print("\n")

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
) """
""" print(metrics.flat_classification_report(
    y_te, y_pred, labels=sorted_labels)) """

#X_tr=np.array(X_tr, dtype=)






""" clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_tr, y_tr)
y_pred=clf.predict(X_te)

print(metrics.flat_f1_score(y_te, y_pred,
                      average='weighted', labels=labels)) """


""" vectorizer = CountVectorizer()  
vectorizer.fit(X_tr) # convert the vocabulary into vector, each word into number
train_mat = vectorizer.transform(X_tr)

tfidf = TfidfTransformer()
tfidf.fit(train_mat)
train_tfmat = tfidf.transform(train_mat)

test_mat = vectorizer.transform(X_te)
test_tfmat = tfidf.transform(test_mat)



lsvm=LinearSVC()
lsvm.fit(X_tr,y_tr)
y_pred_lsvm=lsvm.predict(X_te)

print("accuracy:", metrics.accuracy_score(y_te, y_pred_lsvm)) """
 



