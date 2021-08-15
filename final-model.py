#####import libraries and data

#import libraries
import pandas as pd

#import data
X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')


#import ML package
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

#hard votingclassifier final model

#stacking def
def get_hard_voting():
    # define the base models
    level0 = list()
    level0.append(('gnb', GaussianNB(var_smoothing= 0.19630406500402708)))
    level0.append(('lr', LogisticRegression(C= 0.9999999999999999, max_iter= 15000)))
    level0.append(('svc', SVC(probability = True, random_state = 1,C= 70, gamma = 0.001, kernel= 'rbf')))
    hard_voting_model = VotingClassifier(estimators=level0, voting='hard')
    return hard_voting_model


hv_model = get_hard_voting()
hv_model.fit(X_train,y_train)
tpred_hv=hv_model.predict(X_test)


#####pickle
import pickle

outfile = open('hard_voting_classification_model.pkl', 'wb')
pickle.dump(hv_model,outfile)
outfile.close()
