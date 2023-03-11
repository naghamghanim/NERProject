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
from hmmlearn import hmm
def extractDigits(lst):
    return list(map(lambda el:[el], lst))

names = [
    "Naive Bayes",
    "Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "HMM"
    #"Neural Net",
    #"AdaBoost",
    #"Linear SVM",
    #"RBF SVM",
    #"Gaussian Process",
    #"QDA",
]

classifiers = [
    BernoulliNB(),
    KNeighborsClassifier(n_neighbors=24),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=10, max_features=1),
    hmm.GaussianHMM(n_components=7,algorithm='viterbi', covariance_type="tied", n_iter=1000),
    #MLPClassifier(alpha=1, max_iter=1000),
    #AdaBoostClassifier(n_estimators=100,base_estimator=SVC(probability=True, kernel='linear'),learning_rate=1, random_state=0),
    #SVC(kernel="linear", C=1),
    #SVC(gamma=1, C=0.05),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    #QuadraticDiscriminantAnalysis(),
]



train = pd.read_csv('/var/home/nhamad/NERproject/finalData/our_features2.csv',encoding='utf8')
train = pd.DataFrame(train)
#  bias   word  word_id  BOS/EOS  preword  preword_id  nextword  nextword_id  label
#'preword',  'preword_id','nextword','nextword_id'
#except the following features

#selection feature 1
#train=train.drop(columns=['bias','BOS/EOS','preword','preword_id' ])

#selection feature 2
#train=train.drop(columns=['bias','BOS/EOS','nextword','nextword_id' ])

#selection feature 3
train=train.drop(columns=['bias','BOS/EOS','nextword','nextword_id','preword','preword_id' ])

#print(train)
min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(train.loc[:,:])
x_scaled = min_max_scaler.fit_transform(train.loc[:, :'word_id'])
train_normalized = pd.DataFrame(x_scaled)

x=train_normalized.values
y=train['label'].values
allLabels=["O", "B-PERS", "I-PERS", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)

for name, clf in zip(names, classifiers):
    
    print("---------------------------------------------")
    print(name)
    print("---------------------------------------------\n")

    clf = clf
    if name=="HMM" :
        clf = clf.fit(x_train)
        predictions=clf.predict(x_test)

        result=clf.score(x_test,y_test)
    else:    
        clf = clf.fit(x_train, y_train)
    
        clf.score(x_test,y_test)

        predictions=clf.predict(x_test)

    result=clf.score(x_test,y_test)
    print("accurecy : {}".format(result))
    print(accuracy_score(y_test, predictions))
    

    y_test2=y_test.tolist()
    predictions=predictions.tolist()
    y_test2= [str(x) for x in y_test2]
    predictions= [str(x) for x in predictions]

    y_test1 = [s.replace('0', 'O') for s in y_test2]
    y_test1 = [s.replace('1', 'B-PERS') for s in y_test1]
    y_test1 = [s.replace('2', 'I-PERS') for s in y_test1]
    y_test1 = [s.replace('3', 'B-ORG') for s in y_test1]
    y_test1 = [s.replace('4', 'I-ORG') for s in y_test1]
    y_test1 = [s.replace('5', 'B-LOC') for s in y_test1]
    y_test1 = [s.replace('6', 'I-LOC') for s in y_test1]

    predictions1 = [s.replace('0', 'O') for s in predictions]
    predictions1 = [s.replace('1', 'B-PERS') for s in predictions1]
    predictions1 = [s.replace('2', 'I-PERS') for s in predictions1]
    predictions1 = [s.replace('3', 'B-ORG') for s in predictions1]
    predictions1 = [s.replace('4', 'I-ORG') for s in predictions1]
    predictions1 = [s.replace('5', 'B-LOC') for s in predictions1]
    predictions1 = [s.replace('6', 'I-LOC') for s in predictions1]

    y_test1=extractDigits(y_test1)
    predictions1=extractDigits(predictions1)
    
    print(confusion_matrix(y_test2, predictions))
    print(classification_report(y_test1, predictions1))
    #print(f1_score(y_test1, predictions1, average='micro'))
    print(f1_score(y_test1, predictions1))

   


