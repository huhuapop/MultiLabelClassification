
from numpy import *
import numpy as np
from sklearn import svm
from sklearn.metrics import jaccard_similarity_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import tree
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier


def loadtrainingData(datafileName, delim='\t'):

    trainingdata = np.genfromtxt(datafileName, dtype=float, delimiter = ',')
    return mat(trainingdata)
    
def loadtrainingLabel(labelfileName, delim='\t'):
    traininglabel = np.genfromtxt(labelfileName, dtype=int, delimiter = ',')
    return mat(traininglabel)

def loadtestdata(testdatafileName, delim='\t'):
  
    testdata = np.genfromtxt(testdatafileName, dtype=float, delimiter = ',')
    return mat(testdata)

def label_classification(trainingdata, traininglabel, testdata):

    data_train = np.array(trainingdata)
    label_train = np.array(traininglabel)
    data_test = np.array(testdata)

    data_train_train,data_train_test,\
    label_train_train,label_train_test=train_test_split(data_train,label_train , test_size=0.3, random_state=42)
    #Random forest 44.9%
    #classifier = RandomForestClassifier()
    #DecisionTreeClassifier 41.3%
    #classifier = DecisionTreeClassifier()
    #KNeighborsClassifier 48.9%
    classifier = KNeighborsClassifier()
    # ExtraTreeClassifier 38.67%
    #classifier = ExtraTreeClassifier()

    clf = OneVsRestClassifier(classifier).fit(data_train_train, label_train_train)
    train_test_predicted = clf.predict(data_train_test)
    accuracy=jaccard_similarity_score(label_train_test, train_test_predicted)
    print accuracy
    test_predicted = clf.predict(data_test)
    return test_predicted
    
if __name__=='__main__':
     mata_trainingdata=loadtrainingData('./MultLabelTrainData.csv')
     mata_traininglabel=loadtrainingLabel('./MultLabelTrainLabel.csv')
     mata_testdata = loadtestdata('./MultLabelTestData.csv')
     result = label_classification(mata_trainingdata, mata_traininglabel, mata_testdata)
     outlabel = np.asarray(result)
     np.savetxt('./MultilabelClassification.txt', result, delimiter=" ", fmt='%i')

