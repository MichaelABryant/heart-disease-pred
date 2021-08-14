#####import libraries and data

#import libraries
import pandas as pd

#import data
X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')


#import ML package
from sklearn.svm import SVC

#support vector regressor final model
svc_model = SVC(probability = True, random_state = 1,C= 70, gamma = 0.001, kernel= 'rbf')
svc_model.fit(X_train,y_train)
tpred_svc=svc_model.predict(X_test)


#####pickle
import pickle

outfile = open('support_vector_classification_model.pkl', 'wb')
pickle.dump(svc_model,outfile)
outfile.close()
