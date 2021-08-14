#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#suppress warnings
warnings.filterwarnings("ignore")

#import data

X = pd.read_csv('data/preprocessed/X.csv')
X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = pd.read_csv('data/preprocessed/y_train.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')

#import ml algorithms
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from numpy import mean, std
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

##baseline

#naive Bayes with five-fold cross validation
gnb = GaussianNB()
cv = cross_val_score(gnb,X_train,y_train,cv=5)
print(mean(cv), '+/-', std(cv))

#logistic regression with five-fold cross validation
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train,y_train,cv=5)
print(mean(cv), '+/-', std(cv))

#decession tree with five-fold cross validation
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train,y_train,cv=5)
print(mean(cv), '+/-', std(cv))

#k-nearest neighbors classifier with five-fold cross validation
knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train,y_train,cv=5)
print(mean(cv), '+/-', std(cv))

#random forest classifier with five-fold cross validation
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train,y_train,cv=5)
print(mean(cv), '+/-', std(cv))

#support vector classifier with five-fold cross validation
svc = SVC(probability = True)
cv = cross_val_score(svc,X_train,y_train,cv=5)
print(mean(cv), '+/-', std(cv))

#xgboost classifier with five-fold cross validation
xgb = XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1)
cv = cross_val_score(xgb,X_train,y_train,cv=5)
print(mean(cv), '+/-', std(cv))

##hyperparameter tuning

#ml algorithm tuner
from sklearn.model_selection import GridSearchCV 

#performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: {} +/- {}'.format(str(classifier.best_score_),str(classifier.cv_results_['std_test_score'][classifier.best_index_])))
    print('Best Parameters: ' + str(classifier.best_params_))
    
#naive Bayes performance tuner
gnb = GaussianNB()
param_grid = {
              'var_smoothing': np.logspace(0,-10, num=100)
             }
clf_lr = GridSearchCV(gnb, param_grid = param_grid, cv = 5, n_jobs = -1)
best_clf_gnb = clf_lr.fit(X_train,y_train)
clf_performance(best_clf_gnb,'Naive Bayes')

#logistic regression performance tuner
lr = LogisticRegression()
param_grid = {'max_iter' : [15000],
              'C' : np.arange(.5,1.5,.1)
             }
clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, n_jobs = -1)
best_clf_lr = clf_lr.fit(X_train,y_train)
clf_performance(best_clf_lr,'Logistic Regression')

#decision tree performance tuner
dt = tree.DecisionTreeClassifier(random_state = 1)
param_grid = {
             'criterion':['gini','entropy'],
             'max_depth': np.arange(1, 15)
             }
clf_dt = GridSearchCV(dt, param_grid = param_grid, cv = 5, n_jobs = -1)
best_clf_dt = clf_dt.fit(X_train,y_train)
clf_performance(best_clf_dt,'Decision Tree')

#k-nearest neighbors classifier performance tuner
knn = KNeighborsClassifier()
param_grid = {
              'n_neighbors' : np.arange(15,20,1),
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree','brute'],
              'p' : [2,3,4,5]
             }
clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, n_jobs = -1)
best_clf_knn = clf_knn.fit(X_train,y_train)
clf_performance(best_clf_knn,'K-Nearest Neighbors Classifier')

#random forest performance tuner
rf = RandomForestClassifier(random_state = 1)
param_grid =  {
                'n_estimators': [310], 
                'bootstrap': [True,False], #bagging (T) vs. pasting (F)
                'max_depth': [1],
                'max_features': ['auto','sqrt'],
                #'min_samples_leaf': [1],
                #'min_samples_split': [1]
              }
clf_rf_rnd = GridSearchCV(rf, param_grid = param_grid, cv = 5, n_jobs = -1)
best_clf_rf_rnd = clf_rf_rnd.fit(X_train,y_train)
clf_performance(best_clf_rf_rnd,'Random Forest')

#support vector classifier performance tuner
svc = SVC(probability = True, random_state = 1)
param_grid = {
              'kernel': ['rbf'],
              'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4],
              'C': np.arange(70,85,1)
             }
clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, n_jobs = -1)
best_clf_svc = clf_svc.fit(X_train,y_train)
clf_performance(best_clf_svc,'Support Vector Classifier')

#xgboost classifier performance tuner
xgb = XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1)
param_grid = {
              'max_depth': [9],
              'n_estimators': [37],
              'learning_rate': [1.2]
             }
clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X_train,y_train)
clf_performance(best_clf_xgb,'XGBoost Classifier')

##StackingClassifier

#baseline

#stacking def
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('gnb', GaussianNB()))
    level0.append(('dt', tree.DecisionTreeClassifier(random_state = 1)))
    level0.append(('lr', LogisticRegression()))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('rf', RandomForestClassifier(random_state = 1)))
    level0.append(('svc', SVC(probability = True, random_state = 1)))
    level0.append(('xgb', XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1)))
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    stacking_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return stacking_model

#models def
def get_models():
    models = dict()
    models['gnb'] = GaussianNB()
    models['dt'] = tree.DecisionTreeClassifier(random_state = 1)
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['rf'] = RandomForestClassifier(random_state = 1)
    models['svc'] = SVC(probability = True, random_state = 1)
    models['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1)
    models['stacking'] = get_stacking()
    return models

#cross validate models and print results
models = get_models()
results, names = list(),list()
print('Mean accuracy:')
for name, model in models.items():
    scores = cross_val_score(model,X_train,y_train, scoring='accuracy', cv=5, n_jobs=-1)
    results.append(scores)
    names.append(name)
    print('>%s %.3f +/- %.3f' % (name, mean(scores), std(scores)))
    
#hyperparameter tuning

#stacking def
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('gnb', GaussianNB(var_smoothing= 0.19630406500402708)))
    #level0.append(('dt', tree.DecisionTreeClassifier(random_state = 1, criterion= 'gini', max_depth= 1)))
    #level0.append(('lr', LogisticRegression(C= 0.9999999999999999, max_iter= 15000)))
    #level0.append(('knn', KNeighborsClassifier(algorithm= 'auto', n_neighbors= 19, p= 4, weights= 'uniform')))
    #level0.append(('rf', RandomForestClassifier(random_state = 1, bootstrap= False, max_depth= 1, max_features= 'auto', n_estimators= 310)))
    level0.append(('svc', SVC(probability = True, random_state = 1,C= 70, gamma = 0.001, kernel= 'rbf')))
    #level0.append(('xgb', XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1, learning_rate= 1.2, max_depth= 9, n_estimators= 37)))
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    stacking_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return stacking_model

#models def
def get_models():
    models = dict()
    models['gnb'] = GaussianNB(var_smoothing= 0.19630406500402708)
    #models['dt'] = tree.DecisionTreeClassifier(random_state = 1, criterion= 'gini', max_depth= 1)
    #models['lr'] = LogisticRegression(C= 0.9999999999999999, max_iter= 15000)
    #models['knn'] = KNeighborsClassifier(algorithm= 'auto', n_neighbors= 19, p= 4, weights= 'uniform')
    #models['rf'] = RandomForestClassifier(random_state = 1, bootstrap= False, max_depth= 1, max_features= 'auto', n_estimators= 310)
    models['svc'] = SVC(probability = True, random_state = 1,C= 70, gamma = 0.001, kernel= 'rbf')
    #models['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1, learning_rate= 1.2, max_depth= 9, n_estimators= 37)
    models['stacking'] = get_stacking()
    return models

#cross validate models and print results
models = get_models()
results, names = list(),list()
print('Mean accuracy:')
for name, model in models.items():
    scores = cross_val_score(model,X_train,y_train, scoring='accuracy', cv=5, n_jobs=-1)
    results.append(scores)
    names.append(name)
    print('>%s %.3f +/- %.3f' % (name, mean(scores), std(scores)))
  
##Hard VotingClassifier

#baseline  
    
#voting def
def get_hard_voting():
    # define the base models
    level0 = list()
    level0.append(('gnb', GaussianNB()))
    level0.append(('dt', tree.DecisionTreeClassifier(random_state = 1)))
    level0.append(('lr', LogisticRegression()))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('rf', RandomForestClassifier(random_state = 1)))
    level0.append(('svc', SVC(probability = True, random_state = 1)))
    level0.append(('xgb', XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1)))
    # define meta learner model
    hard_voting_model = VotingClassifier(estimators=level0, voting='hard')
    return hard_voting_model

#models def
def get_models():
    models = dict()
    models['gnb'] = GaussianNB()
    models['dt'] = tree.DecisionTreeClassifier(random_state = 1)
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['rf'] = RandomForestClassifier(random_state = 1)
    models['svc'] = SVC(probability = True, random_state = 1)
    models['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1)
    models['hv'] = get_hard_voting()
    return models

#cross validate models and print results
models = get_models()
results, names = list(),list()
print('Mean accuracy:')
for name, model in models.items():
    scores = cross_val_score(model,X_train,y_train, scoring='accuracy', cv=5, n_jobs=-1)
    results.append(scores)
    names.append(name)
    print('>%s %.3f +/- %.3f' % (name, mean(scores), std(scores)))
    
    
#hyperparameter tuning

#voting def
def get_hard_voting():
    # define the base models
    level0 = list()
    level0.append(('gnb', GaussianNB(var_smoothing= 0.19630406500402708)))
    #level0.append(('dt', tree.DecisionTreeClassifier(random_state = 1, criterion= 'gini', max_depth= 1)))
    level0.append(('lr', LogisticRegression(C= 0.9999999999999999, max_iter= 15000)))
    #level0.append(('knn', KNeighborsClassifier(algorithm= 'auto', n_neighbors= 19, p= 4, weights= 'uniform')))
    #level0.append(('rf', RandomForestClassifier(random_state = 1, bootstrap= False, max_depth= 1, max_features= 'auto', n_estimators= 310)))
    level0.append(('svc', SVC(probability = True, random_state = 1,C= 70, gamma = 0.001, kernel= 'rbf')))
    #level0.append(('xgb', XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1, learning_rate= 1.2, max_depth= 9, n_estimators= 37)))
    hard_voting_model = VotingClassifier(estimators=level0, voting='hard')
    return hard_voting_model

#models def
def get_models():
    models = dict()
    models['gnb'] = GaussianNB(var_smoothing= 0.19630406500402708)
    #models['dt'] = tree.DecisionTreeClassifier(random_state = 1, criterion= 'gini', max_depth= 1)
    models['lr'] = LogisticRegression(C= 0.9999999999999999, max_iter= 15000)
    #models['knn'] = KNeighborsClassifier(algorithm= 'auto', n_neighbors= 19, p= 4, weights= 'uniform')
    #models['rf'] = RandomForestClassifier(random_state = 1, bootstrap= False, max_depth= 1, max_features= 'auto', n_estimators= 310)
    models['svc'] = SVC(probability = True, random_state = 1,C= 70, gamma = 0.001, kernel= 'rbf')
    #models['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1, learning_rate= 1.2, max_depth= 9, n_estimators= 37)
    models['hv'] = get_hard_voting()
    return models

#cross validate models and print results
models = get_models()
results, names = list(),list()
print('Mean accuracy:')
for name, model in models.items():
    scores = cross_val_score(model,X_train,y_train, scoring='accuracy', cv=5, n_jobs=-1)
    results.append(scores)
    names.append(name)
    print('>%s %.3f +/- %.3f' % (name, mean(scores), std(scores)))
    

##Soft VotingClassifier

#baseline

#voting def
def get_soft_voting():
    # define the base models
    level0 = list()
    level0.append(('gnb', GaussianNB()))
    level0.append(('dt', tree.DecisionTreeClassifier(random_state = 1)))
    level0.append(('lr', LogisticRegression()))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('rf', RandomForestClassifier(random_state = 1)))
    level0.append(('svc', SVC(probability = True, random_state = 1)))
    level0.append(('xgb', XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1)))
    # define meta learner model
    soft_voting_model = VotingClassifier(estimators=level0, voting='soft')
    return soft_voting_model

#models def
def get_models():
    models = dict()
    models['gnb'] = GaussianNB()
    models['dt'] = tree.DecisionTreeClassifier(random_state = 1)
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['rf'] = RandomForestClassifier(random_state = 1)
    models['svc'] = SVC(probability = True, random_state = 1)
    models['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1)
    models['sv'] = get_soft_voting()
    return models

#cross validate models and print results
models = get_models()
results, names = list(),list()
print('Mean accuracy:')
for name, model in models.items():
    scores = cross_val_score(model,X_train,y_train, scoring='accuracy', cv=5, n_jobs=-1)
    results.append(scores)
    names.append(name)
    print('>%s %.3f +/- %.3f' % (name, mean(scores), std(scores)))
    
#hyperparameter tuning

#voting def
def get_soft_voting():
    # define the base models
    level0 = list()
    level0.append(('gnb', GaussianNB(var_smoothing= 0.19630406500402708)))
    #level0.append(('dt', tree.DecisionTreeClassifier(random_state = 1, criterion= 'gini', max_depth= 1)))
    #level0.append(('lr', LogisticRegression(C= 0.9999999999999999, max_iter= 15000)))
    #level0.append(('knn', KNeighborsClassifier(algorithm= 'auto', n_neighbors= 19, p= 4, weights= 'uniform')))
    #level0.append(('rf', RandomForestClassifier(random_state = 1, bootstrap= False, max_depth= 1, max_features= 'auto', n_estimators= 310)))
    level0.append(('svc', SVC(probability = True, random_state = 1,C= 70, gamma = 0.001, kernel= 'rbf')))
    #level0.append(('xgb', XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1, learning_rate= 1.2, max_depth= 9, n_estimators= 37)))
    # define meta learner model
    soft_voting_model = VotingClassifier(estimators=level0, voting='soft')
    return soft_voting_model

#models def
def get_models():
    models = dict()
    models['gnb'] = GaussianNB(var_smoothing= 0.19630406500402708)
    #models['dt'] = tree.DecisionTreeClassifier(random_state = 1, criterion= 'gini', max_depth= 1)
    #models['lr'] = LogisticRegression(C= 0.9999999999999999, max_iter= 15000)
    #models['knn'] = KNeighborsClassifier(algorithm= 'auto', n_neighbors= 19, p= 4, weights= 'uniform')
    #models['rf'] = RandomForestClassifier(random_state = 1, bootstrap= False, max_depth= 1, max_features= 'auto', n_estimators= 310)
    models['svc'] = SVC(probability = True, random_state = 1,C= 70, gamma = 0.001, kernel= 'rbf')
    #models['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='error', random_state =1, learning_rate= 1.2, max_depth= 9, n_estimators= 37)
    models['sv'] = get_soft_voting()
    return models

#cross validate models and print results
models = get_models()
results, names = list(),list()
print('Mean accuracy:')
for name, model in models.items():
    scores = cross_val_score(model,X_train,y_train, scoring='accuracy', cv=5, n_jobs=-1)
    results.append(scores)
    names.append(name)
    print('>%s %.3f +/- %.3f' % (name, mean(scores), std(scores)))
    
##BaggingClassifier

#baggingclassifier baseline
bagging_model = BaggingClassifier(base_estimator=RandomForestClassifier(),
                                     bootstrap=True,
                                     random_state=1,
                                     n_jobs=-1
                                     )

bagging_model.fit(X_train , y_train)

cv = cross_val_score(bagging_model, X_train, y_train, cv=5)
print(mean(cv), '+/-', std(cv))

#baggingclassifier tuning
bagging_model = BaggingClassifier(base_estimator=RandomForestClassifier(),
                                     bootstrap=True,
                                     random_state=1,
                                     n_estimators=20,
                                     max_samples=50,
                                     n_jobs=-1,
                                     )

bagging_model.fit(X_train , y_train)

cv = cross_val_score(bagging_model, X_train, y_train, cv=5)
print(mean(cv), '+/-', std(cv))

##BaggingClassifier (pasting)

#baggingclassifier (pasting) baseline
pasting_model = BaggingClassifier(base_estimator=RandomForestClassifier(),
                                     bootstrap=False,
                                     random_state=1,
                                     n_jobs=-1
                                     )

pasting_model.fit(X_train , y_train)

cv = cross_val_score(pasting_model, X_train, y_train, cv=5)
print(mean(cv), '+/-', std(cv))

#baggingclassifier (pasting) tuner
pasting_model = BaggingClassifier(base_estimator=RandomForestClassifier(random_state = 1, bootstrap=True,max_depth=7, max_features='auto', n_estimators=340),
                                     bootstrap=False,
                                     random_state=1,
                                     n_estimators=20,
                                     max_samples=50,
                                     n_jobs=-1,
                                     )

pasting_model.fit(X_train , y_train)

cv = cross_val_score(pasting_model, X_train, y_train, cv=5)
print(mean(cv), '+/-', std(cv))

##AdaBoostClassifier

#addboostclassifier baseline
adaboost_model = AdaBoostClassifier(base_estimator=RandomForestClassifier(),
                                       random_state=1)

adaboost_model.fit(X_train , y_train)

cv = cross_val_score(adaboost_model, X_train, y_train, cv=5)
print(mean(cv), '+/-', std(cv))

#adaboostclassifier tuning
adaboost_model = AdaBoostClassifier(base_estimator=RandomForestClassifier(random_state = 1, bootstrap=True,max_depth=7, max_features='auto', n_estimators=340),
#                                        learning_rate=1,
                                       random_state=1)

adaboost_model.fit(X_train , y_train)

cv = cross_val_score(adaboost_model, X_train, y_train, cv=5)
print(mean(cv), '+/-', std(cv))

##Evaluating the best models

#import evaluation tools
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
import scikitplot as skplt

#SVC

#create support vector classifier model with tuned parameters
svc = SVC(probability = True, random_state = 1,C= 70, gamma = 0.001, kernel= 'rbf')
svc.fit(X_train,y_train)
y_pred1 = svc.predict(X_test)

#assess accuracy
print('SVC test accuracy: {}'.format(accuracy_score(y_test, y_pred1)))

#support vector classifier confusion matrix
#create and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred1)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

#plot as heatmap
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=sns.color_palette('Reds'), linewidths=0.2, vmin=0, vmax=1)

#plot settings
class_names = ['Heart disease', 'No heart disease']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Support Vector Classifier')
plt.show()

#support vector classifier sensitivity and specificity calculations
total=sum(sum(matrix))

print('SVC')
sensitivity = matrix[0,0]/(matrix[0,0]+matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = matrix[1,1]/(matrix[1,1]+matrix[0,1])
print('Specificity : ', specificity)

#view the support vector classification report
print('SVC')
print(classification_report(y_test, y_pred1))

#lift curve for support vector classifier
target_prob = svc.predict_proba(X_test)
skplt.metrics.plot_lift_curve(y_test, target_prob)
plt.show()

#Matthews correlation coefficient for SVC
print('SVC MCC: {}'.format(matthews_corrcoef(y_test, y_pred1)))

#StackingClassifier

#create stacking classifier model
stacking_model = get_stacking()
stacking_model.fit(X_train,y_train)
y_pred2 = stacking_model.predict(X_test)

#assess accuracy
print('StackingClassifier test accuracy: {}'.format(accuracy_score(y_test, y_pred2)))

#stacking classifier confusion matrix
#create and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred2)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

#plot as heatmap
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=sns.color_palette('Reds'), linewidths=0.2, vmin=0, vmax=1)

#plot settings
class_names = ['Heart disease', 'No heart disease']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Stacking Classifier')
plt.show()

#stacking classifier sensitivity and specificity calculations
total=sum(sum(matrix))

print('StackingClassifier')
sensitivity = matrix[0,0]/(matrix[0,0]+matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = matrix[1,1]/(matrix[1,1]+matrix[0,1])
print('Specificity : ', specificity)

#view the stacking classification report
print('StackingClassifier')
print(classification_report(y_test, y_pred2))

#lift curve for the stacking model
target_prob = stacking_model.predict_proba(X_test)
skplt.metrics.plot_lift_curve(y_test, target_prob)
plt.show()

#Matthews correlation coefficient for StackingClassifier
print('StackingClassifier MCC: {}'.format(matthews_corrcoef(y_test, y_pred2)))

#Hard VotingClassifier

#create hard voting classifier model
hv_model = get_hard_voting()
hv_model.fit(X_train,y_train)
y_pred3 = hv_model.predict(X_test)

#assess accuracy
print('Hard VotingClassifier test accuracy: {}'.format(accuracy_score(y_test, y_pred3)))

#hard voting classifier confusion matrix
#create and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred3)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

#plot as heatmap
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=sns.color_palette('Reds'), linewidths=0.2, vmin=0, vmax=1)

#plot settings
class_names = ['Heart disease', 'No heart disease']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Hard VotingClassifier')
plt.show()

#hard voting classifier sensitivity and specificity calculations
total=sum(sum(matrix))

print('Hard VotingClassifier')
sensitivity = matrix[0,0]/(matrix[0,0]+matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = matrix[1,1]/(matrix[1,1]+matrix[0,1])
print('Specificity : ', specificity)

#view the hard voting classification report
print('Hard VotingClassifier')
print(classification_report(y_test, y_pred3))

#Matthews correlation coefficient for Hard VotingClassifier
print('Hard VotingClassifier MCC: {}'.format(matthews_corrcoef(y_test, y_pred3)))

#Soft VotingClassifier

#create soft voting classifier model
sv_model = get_soft_voting()
sv_model.fit(X_train,y_train)
y_pred4 = sv_model.predict(X_test)

#assess accuracy
print('Soft VotingClassifier test accuracy: {}'.format(accuracy_score(y_test, y_pred4)))

#soft voting classifier confusion matrix
#create and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred4)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

#plot as heatmap
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=sns.color_palette('Reds'), linewidths=0.2, vmin=0, vmax=1)

#plot settings
class_names = ['Heart disease', 'No heart disease']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Soft VotingClassifier')
plt.show()

#soft voting classifier sensitivity and specificity calculations
total=sum(sum(matrix))

print('Soft VotingClassifier')
sensitivity = matrix[0,0]/(matrix[0,0]+matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = matrix[1,1]/(matrix[1,1]+matrix[0,1])
print('Specificity : ', specificity)

#view the soft voting classification report
print('Soft VotingClassifier')
print(classification_report(y_test, y_pred4))

#lift curve for the soft voting model
target_prob = sv_model.predict_proba(X_test)
skplt.metrics.plot_lift_curve(y_test, target_prob)
plt.show()

#Matthews correlation coefficient for Soft VotingClassifier
print('Soft VotingClassifier MCC: {}'.format(matthews_corrcoef(y_test, y_pred4)))

#Naive Bayes

#create naive bayes model
gnb = GaussianNB(var_smoothing= 0.19630406500402708)
gnb.fit(X_train,y_train)
y_pred5 = gnb.predict(X_test)

#assess accuracy
print('GaussianNB test accuracy: {}'.format(accuracy_score(y_test, y_pred5)))

#naive bayes confusion matrix
#create and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred5)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

#plot as heatmap
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=sns.color_palette('Reds'), linewidths=0.2, vmin=0, vmax=1)

#plot settings
class_names = ['Heart disease', 'No heart disease']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for GaussianNB')
plt.show()

#naive bayes sensitivity and specificity calculations
total=sum(sum(matrix))

print('GaussianNB')
sensitivity = matrix[0,0]/(matrix[0,0]+matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = matrix[1,1]/(matrix[1,1]+matrix[0,1])
print('Specificity : ', specificity)

#view the naive bayes report
print('GaussianNB')
print(classification_report(y_test, y_pred5))

#lift curve for the pasting model
target_prob = gnb.predict_proba(X_test)
skplt.metrics.plot_lift_curve(y_test, target_prob)
plt.show()

print('GaussianNB MCC: {}'.format(matthews_corrcoef(y_test, y_pred5)))

##ROC/AUC

#plot ROC curve for best models
from sklearn import metrics

pred_prob1 = svc.predict_proba(X_test)
pred_prob2 = stacking_model.predict_proba(X_test)
pred_prob4 = sv_model.predict_proba(X_test)
pred_prob5 = pasting_model.predict_proba(X_test)
fpr1, tpr1, thresholds1 = metrics.roc_curve(y_test, pred_prob1[:,1],pos_label=1)
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_test, pred_prob2[:,1],pos_label=1)
fpr4, tpr4, thresholds2 = metrics.roc_curve(y_test, pred_prob4[:,1],pos_label=1)
fpr5, tpr5, thresholds3 = metrics.roc_curve(y_test, pred_prob5[:,1],pos_label=1)

fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(fpr1, tpr1, label='SVC')
ax.plot(fpr2, tpr2, label='Stacking')
ax.plot(fpr4, tpr4, label='Soft Voting')
ax.plot(fpr5, tpr5, label='GaussianNB')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.legend()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for heart disease classifiers')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

#calculate AUC for classifiers
print('SVC AUC: {}'.format(metrics.auc(fpr1, tpr1)))
print('Stacking AUC: {}'.format(metrics.auc(fpr2, tpr2)))
print('Soft Voting AUC: {}'.format(metrics.auc(fpr4, tpr4)))
print('GaussianNB AUC: {}'.format(metrics.auc(fpr5, tpr5)))

##Feature importance

#import libraries
from matplotlib import pyplot as plt
# import eli5
# from eli5.sklearn import PermutationImportance
from sklearn import tree
import graphviz
# import shap

#SVC

# #build SVC model
# svc_model = SVC(probability = True, random_state = 1, C= 100, gamma= 0.01, kernel = 'rbf').fit(X_train,y_train)

# #create object that can calculate shap values
# explainer = shap.KernelExplainer(svc_model.predict_proba, X_train)

# pred_data = pd.DataFrame(X_test)

feature_names = X.columns

# pred_data.columns = feature_names

# data_for_prediction = pred_data

# #calculate Shap values
# shap_values = explainer.shap_values(data_for_prediction)

# shap.initjs()
# shap.summary_plot(shap_values[1], data_for_prediction)

# #StackingClassifier

# #fit the model
# stacking_model.fit(X_train,y_train)

# #make prediction
# y_pred = stacking_model.predict(X_test)

# #determine feature weights
# perm = PermutationImportance(stacking_model, random_state=1).fit(X_test, y_test)
# eli5.show_weights(perm, feature_names = list(feature_names), top=len(feature_names))

#Decision Tree

#create decision tree model with tuned parameters
dt = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth= 3)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

#dt accuracy print
print('dt test accuracy: {}'.format(accuracy_score(y_test, y_pred)))

#look at decision tree
#value tells how many records from each category entered the box (i.e., [# of records = 0, # of records = 1])
tree_graph = tree.export_graphviz(dt, out_file=None, feature_names=feature_names)
graphviz.Source(tree_graph)

# #create object that can calculate shap values
# explainer = shap.TreeExplainer(dt)

# pred_data = pd.DataFrame(X_test)

# pred_data.columns = feature_names

# data_for_prediction = pred_data

# #calculate Shap values
# shap_values = explainer.shap_values(data_for_prediction)

# #create summary plot
# shap.initjs()
# shap.summary_plot(shap_values[1], data_for_prediction)


# Best model
# SVC
# Accuracy: 0.8421
# Sensitivity: 0.8677
# Specificity: 0.8145
# Precision: 0.8372
# AUC: 0.9310

# Most important features
# thal
# ca
# sex
# cp