#import libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#suppress warnings
warnings.filterwarnings("ignore")

#import data
heart_data = pd.read_csv('data/heart.csv')

#look at formatting of entries
heart_data.head()

#display null values and data types
heart_data.info()

#null values for ca
heart_data[heart_data.ca==4]

#null percentage for ca
print('Percentage of ca null: {}%'.format((len(heart_data[heart_data.ca==4])/len(heart_data.ca))*100))

#null values for thal
heart_data[heart_data.thal==0]

#null percentage for thal
print('Percentage of thal null: {}%'.format((len(heart_data[heart_data.thal==0])/len(heart_data.thal))*100))

#numerical features
numerical = [
    'age',
    'trestbps',
    'chol',
    'thalach',
    'oldpeak',
]

#categorical features
categorical = [
    'sex',
    'cp',
    'fbs',
    'restecg',
    'exang',
    'slope',
    'ca',
    'thal',
    'target'
]

#look at distribution of data
heart_data.describe()

#look at number of outliers greater than or equal to 3 std from mean
heart_data[np.abs(stats.zscore(heart_data)) >= 3]

#look at number of outliers greater than or equal to 4 std from mean
heart_data[np.abs(stats.zscore(heart_data)) >= 4]

#an outlier who is a 67 year old female with a cholesterol greater than six std from mean
heart_data[np.abs(stats.zscore(heart_data)) >= 6]

#oldpeak outlier visualized
sns.boxplot(x=heart_data['oldpeak'], palette='Set1')
plt.xlabel('oldpeak')

#cholesterol outlier visualized
sns.boxplot(x=heart_data['chol'], palette='Set1')
plt.xlabel('chol')

#look at numerical data distribution
for i in heart_data[numerical].columns:
    plt.hist(heart_data[numerical][i], color='steelblue', edgecolor='black')
    plt.xticks()
    plt.xlabel(i)
    plt.ylabel('number of people')
    plt.show()
    
#look at categorical data distribution
for i in heart_data[categorical].columns:
    sns.barplot(edgecolor='black',x=heart_data[categorical][i].value_counts().index,y=heart_data[categorical][i].value_counts(),palette='Set1')
    plt.xlabel(i)
    plt.ylabel('number of people')
    plt.show()
    
#heat map to see numerical correlations, pearson measures monotonic relationship (numerical or ordinal categorical)
plt.figure(figsize=(16, 6))
sns.heatmap(heart_data[numerical].corr(method='pearson'), vmin=-1, vmax=1, annot=True,cmap='coolwarm')
plt.title('Pearson Correlation Heatmap for numerical Variables', fontdict={'fontsize':12}, pad=12);

#look at how target is distributed among variables
sns.pairplot(heart_data,hue='target',palette='Set1')
plt.legend()
plt.show()

#thalach vs age
sns.lmplot(x='age', y='thalach', data=heart_data, palette='Set1')

#settings to display all markers
xticks, xticklabels = plt.xticks()
xmin = (3*xticks[0] - xticks[1])/2.
xmax = (3*xticks[-1] - xticks[-2])/2.
plt.xlim(xmin, xmax)
plt.xticks(xticks)

plt.show()

#thalach vs age with target hue
sns.lmplot(x='age', y='thalach', hue='target', data=heart_data,palette='Set1')

#settings to display all markers
xticks, xticklabels = plt.xticks()
xmin = (8*xticks[0] - xticks[1])/2.
xmax = (3*xticks[-1] - xticks[-2])/2.
plt.xlim(xmin, xmax)
plt.xticks(xticks)

plt.show()

#age vs target
sns.violinplot(x='target', y='age', data=heart_data, palette='Set1')

#cp distribution with exang hue
sns.histplot(discrete=True,x="cp", hue="exang", data=heart_data, stat="count", multiple="stack",palette='Set1')

plt.ylabel('number of people')
plt.show()

#cp distribution with target hue
sns.histplot(discrete=True, x="cp", hue="target", data=heart_data, stat="count", multiple="stack",palette='Set1')

plt.ylabel('number of people')
plt.xticks(ticks=[0,1,2,3])
plt.show()

#thalach vs exang
sns.violinplot(x='exang', y='thalach', data=heart_data, palette='Set1')
plt.show()

#thalach vs exang with target hue
sns.violinplot(x='exang', y='thalach', data=heart_data, palette='Set1', hue='target')
plt.show()

#thalach vs exang with target hue
sns.swarmplot(y=heart_data['thalach'],
              x=heart_data['exang'], hue=heart_data['target'],palette='Set1')

plt.show()

#thalach vs target
sns.violinplot(x='target', y='thalach', data=heart_data, palette='Set1')
plt.show()

#exang distribution with target hue
sns.histplot(discrete=True, x="exang", hue="target", data=heart_data, stat="count", multiple="stack",palette='Set1')
plt.ylabel('number of people')
plt.xticks(ticks=[0,1])
plt.show()

#oldpeak vs slope
sns.violinplot(x='slope', y='oldpeak', data=heart_data, palette='Set1')
plt.show

#oldpeak vs target
sns.violinplot(x='target', y='oldpeak', data=heart_data, palette='Set1')
plt.show()

#distribution of slope with target hue
sns.histplot(discrete=True, x="slope", hue="target", data=heart_data, stat="count", multiple="stack",palette='Set1')

plt.ylabel('number of people')
plt.xticks(ticks=[0,1,2])
plt.show()

#distribution of ca with target hue
sns.histplot(discrete=True, x="ca", hue="target", data=heart_data, stat="count", multiple="stack",palette='Set1')

plt.ylabel('number of people')
plt.xticks(ticks=[0,1,2,3,4])
plt.show()

#distribution of thal with target hue
sns.histplot(discrete=True, x="thal", hue="target", data=heart_data, stat="count", multiple="stack",palette='Set1')

plt.ylabel('number of people')
plt.xticks(ticks=[0,1,2,3])
plt.show()

#creating arrays that meet critera for risk factors
age_sex_risk = heart_data.loc[(heart_data.sex == 0) & (heart_data.age >= 50) |
                                   (heart_data.sex == 1) & (heart_data.age >= 45) ]

high_blood_pressure_risk = heart_data.loc[heart_data.trestbps >= 130]

high_cholesterol_risk = heart_data.loc[heart_data.chol >= 240]

diabetes_risk = heart_data.loc[heart_data.fbs == 1]

#creating a new column called 'risk factors' which counts the number of risk factors each patient has
risk_factors_indices = np.concatenate((age_sex_risk.index,
                                       high_blood_pressure_risk.index,
                                       high_cholesterol_risk.index,
                                       diabetes_risk.index))

risk_factor_counts = np.bincount(risk_factors_indices)

risk_factors = pd.DataFrame(risk_factor_counts)

risk_factors['risk factors']=risk_factors

risk_factors['target'] = heart_data['target'].copy()

#distribution of risk factors with target hue
sns.histplot(discrete=True, x="risk factors", hue="target", data=risk_factors, stat="count", multiple="stack",palette='Set1')

plt.ylabel('number of people')
plt.xticks(ticks=[0,1,2,3,4])
plt.show()

#remove target variable from categorical array
categorical.remove('target')

#change dtype of categorical features to object
heart_data[categorical]=heart_data[categorical].astype('object')

#copy of variables and target
X = heart_data.copy()
y = X.pop('target')

#import libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#get feature names
X = pd.concat([X[numerical],pd.get_dummies(X[categorical], drop_first=True)],axis=1)
feature_names = X.columns

# train/test split with stratify making sure classes are evenlly represented across splits
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.75, random_state=1)

#define scaler
scaler=MinMaxScaler()

#apply preprocessing to split data with scaler
X_train[numerical] = scaler.fit_transform(X_train[numerical])
X_test[numerical] = scaler.transform(X_test[numerical])

#####pickle
import pickle

outfile = open('scaler.pkl', 'wb')
pickle.dump(scaler,outfile)
outfile.close()

####save data to csv

import os

path = r'C:\Users\malex\Desktop\heart-disease\data\preprocessed\\'

export_X = X.to_csv(os.path.join(path,r'X.csv'),index=False)
export_y = y.to_csv(os.path.join(path,r'y.csv'),index=False)

export_X_train = X_train.to_csv(os.path.join(path,r'X_train.csv'),index=False)
export_X_test = X_test.to_csv(os.path.join(path,r'X_test.csv'),index=False)
export_y_train = y_train.to_csv(os.path.join(path,r'y_train.csv'),index=False)
export_y_test = y_test.to_csv(os.path.join(path,r'y_test.csv'),index=False)
