# Project 3: Heart Disease Prediction (Python/HTML)

This repository is for the analysis and modeling done with the Kaggle: Heart Disease ICU dataset. Below you will find an overview of the data, code, and results. The goal was to create an end-to-end project where I perform an exploratory data analysis (EDA), feature engineer, apply machine learning algorithms to predict heart disease, and create a [deployed application with a front end](https://predict-heart-diseases.herokuapp.com/) to productionize the best performing model. The repo for the app can be found [here](https://github.com/MichaelBryantDS/heart-disease-pred-app).

### Code Used 

**Python Version:** 3.7.10 <br />
**Packages:** pandas, numpy, scipy, sklearn, matplotlib, seaborn, flask, statsmodels, shap, eli5, pickle<br />
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  

## Heart Disease Dataset

The dataset was gathered from [Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci). The dataset contains 14 variables and 303 patient records.

### Variables

`age`, `sex`,`cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

## Files

### eda-engineering.py

This file contains the EDA and feature engineering. The EDA is performed using descriptive statistics, histograms to determine distributions, and a correlation heatmap using the Pearson correlation coefficient. A feature is engineered by creating a predictor based on risk factors. Other feature engineering includes the creation of dummy variables for categorical variables with pandas get_dummies and numerical features are scaled using MinMaxScaler. The scaler is pickled after fitting for use with productionization.

### modeling.py

This file contains the modeling where I hyperparameter tune: GaussianNB, LogisticRegression, DecisionTreeClassifier, kNeighborsClassifier, RandomForestClassifier, SVC (support vector classifier), XGBClassifier, StackingClassifier, (hard) VotingClassifier, (soft) VotingClassifier, BaggingClassifier, (pasting with) BaggingClassifier, and AdaBoostClassifier. Since the computational needs are low from having 303 records with 12 features, I used 13 ML algorithms and ensemble methods. The models are hyperparameter tuned with GridSearchCV based on accuracy and the best models are judged based on accuracy, sensitivity, specificity, precision, and AUC metrics. This file also contains code to derive the feature importance from the best models using shap and eli5 package.

### final-model.py

This file contains the best model (SVC) and it is pickled for use with productionization.

## Results

### EDA

I looked at the distributions of the data and the correlations between variables. Below are some of the highlights:

<div align="center">
  
<figure>
<img src="images/cor-heatmap.jpg"><br/>
  <figcaption>Figure 1: Correlation heatmap for numerical variables using Pearson correlation coefficient</figcaption>
</figure>
<br/><br/>
  
</div>

<div align="center">
  
<figure>
<img src="images/thalach-age-target.jpg"><br/>
  <figcaption>Figure 2: Scatter plot with linear regression lines showing maximum heart rate decreases at a greater rate with age for those with heart disease.</figcaption>
</figure>
<br/><br/>
  
</div>

<div align="center">
  
<figure>
<img src="images/oldpeak-target.jpg"><br/>
  <figcaption>Figure 3: Violin plot showing lower oldpeak values for those with heart disease.</figcaption>
</figure>
<br/><br/>
  
</div>

<div align="center">
  
<figure>
<img src="images/risk-factors-target.jpg"><br/>
  <figcaption>Figure 4: Bar plot showing minimal relationship between number of risk factors and heart disease.</figcaption>
</figure>
<br/><br/>
  
</div>

### Feature Engineering

I feature engineered using the dataset for future modeling. I made the following changes:
* Created dummy variables for `sex`,`cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`

### Model Building

First, I split the data into train and tests sets with a test set size of 25%.

I then hyperparameter tuned 13 different models with five-fold cross-validation and evaluated them using accuracy.

The models I used were GaussianNB, LogisticRegression, DecisionTreeClassifier, kNeighborsClassifier, RandomForestClassifier, SVC (support vector classifier), XGBClassifier, StackingClassifier, (hard) VotingClassifier, (soft) VotingClassifier, BaggingClassifier, (pasting with) BaggingClassifier, and AdaBoostClassifier.

### Model Performance

For this application it's important to minimize false negatives (i.e., people who have heart disease but were predicted not to). I looked at accuracy, precision, specificity, sensitivity, AUC, and MCC to choose the best model. For these reasons, the best model was:
<br/><br/>
**Hard VotingClassifier**
* Accuracy: 0.8553
* Sensitivity: 0.8717
* Specificity: 0.8367
* Precision: 0.8571
* MCC: 0.7084

The hard voting classifier included GaussianNB, LogisticRegression, and SVC.

### Feature Importance

According to the permutation importance for the hard voting classifier, the most important features, in order, were `slope`, `thalach`, `oldpeak`, `ca`, and `cp`.

<div align="center">
  
<figure>
<img src="images/perm-import-hv.JPG"><br/>
  <figcaption>Figure 9: Permutation importance for the hard voting classifier.</figcaption>
</figure>
<br/><br/>
  
</div>

<div align="center">
  
<figure>
<img src="images/shap-svc.jpg"><br/>
  <figcaption>Figure 8: SHAP summary plot of the feature importance for the SVC model.</figcaption>
</figure>
<br/><br/>
  
</div>

<div align="center">
  
<figure>
<img src="images/dt.jpg"><br/>
  <figcaption>Figure 10: Graphical representation of a decision tree.</figcaption>
</figure>
<br/><br/>
  
</div>

## Productionization

I built a [Heroku web app](https://predict-heart-diseases.herokuapp.com/) with a front end which takes patient medical information input from the user and outputs a heart disease prediction using the SVC model. More information on this can be found at it's [repo](https://github.com/MichaelBryantDS/heart-disease-pred-app).

<div align="center">
  
<figure>
<img src="images/front-end.JPG"><br/>
  <figcaption>Figure 12: Front end of application using the SVC model.</figcaption>
</figure>
<br/><br/>
  
</div>

## Resources

1. [Kaggle: Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)
2. [Kaggle: Ken Jee - Titanic Project Example](https://www.kaggle.com/kenjee/titanic-project-example)
3. [Machine Learning Mastery: Stacking Ensemble Machine Learning with Python](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)
4. [Machine Learning Mastery: How to Report Classifier Performance with Confidence Intervals](https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/)
5. [Medium: Evaluating a Random Forest Model](https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56)
6. [Analytics Vidhya: AUC-ROC Curve in Machine Learning Clearly Explained](https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/)
