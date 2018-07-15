# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 17:51:30 2018

@author: HP
"""

import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import sklearn.metrics import confusion_matrix
import sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

#plotting learning curve
'''
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 10)):
    #https://docs.scipy.org/doc/numpy-1.8.1/reference/generated/numpy.linspace.html#numpy.linspace
    plt.figure() #creates a new figure
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis =1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color="r")
    #Fill the area between 2 horizontal curves. The curves are defined by the points (x, y1) and (x,y2). This creates one or multiple polygons describing the filled area
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.1, color = "g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color = "r", label = "Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color = "r", label = "Cross-validation score")
    plt.legend(loc = "best")
    return plt
'''

df = pd.read_excel('H:\Desktop\IdeaBuilder.xlsx', sheetname = 'DetailReport')
listSuExO = df['submitterorg']
newListSuExO = [i.split('\n', 1)[0] for i in listSuExO]
new_df = pd.DataFrame(newListSuExO, columns = ['submitterorg'])
df.update(new_df)
listBenefitCategory = df['Benefit Category']
listImpactArea = df['Impace Area']
data_X = pd.DataFrame({'submitterorg' : listSuExO, 'Benefit Category' : listBenefitCategory, 'Impact Area' : listImpactArea}, columns = ['submitterorg', 'Benefit Category', 'Impact Area'])
data_X["submitterorg"] = data_X["submitterorg"].astype('category')
data_X["Benefit Category"] = data_X["Benefit Category"].astype('category')
data_X["Impact Area"] = data_X["Impact Area"].astype('category')
dict_submitterorg = dict(enumerate(data_X["submitterorg"].cat.categories))
dict_benefit_category = dict(enumerate(data_X["Benefit Category"].cat.categories))
data_impact_area = dict(enumerate(data_X["Impact Area"].cat.categories))
data_X["submitterorg"] = data_X["submitterorg"].cat.codes
data_X["Benefit Category"] = data_X["Benefit Category"].cat.codes
data_X["Impact Area"] = data_X["Impact Area"].cat.codes
listShortDesc = df["Short Description"]
#create the transform
vectorizer = CountVectorizer()
#tokenize and build vocab
vectorizer.fit(listShortDesc)
#summarize
#print(vectorizer.vocabulary_) 
#encode document
vector = vectorizer.transform(listShortDesc)
#summarize encoded vector
print(vector.shape)
print(type(vector))
array_this = vector.toarray()
data = pd.DataFrame(array_this)
data_X = data_X.join(data)
data_X['Impact_Area_sq'] = data_X['Impact Area']**2
#Check if all the values are correct labelled
pd.set_option('display.max_rows', 20)

#Onehot encoding
data_X = pd.get_dummies(data_X, columns = ["submitterorg", "Benefit Category", "Impact Area_sq"])

'''
result_sub = data_sub.idxmax(axis = 1)
result_ben = data_ben.idxmax(axis = 1)
result_imp = data_imp.idxmax(axis = 1)
data_X_out = pd.concat([result_sub, result_ben, results_impl, axis = 1])
'''

''' 
def outputX():
    i = 0
    while i <= len(data_X.shape[0]):
        print(result_ben.iloc[7])
        return;
outputX()
'''

#finding the number of columns in the dataframe
#print(len(data.columns))

#find the max and min labels for all columns
#print(data.max())

#Setting the y
#finding number of rows and columns of dataset in pandas
#print(data_X.shape[1])
listUpdatedIdeaStatus = df['UPDATED Idea Status']
listPriority = df['PRIORITY']
listEcoSys = df['ECOSYSTEM']

data_Y = pd.DataFrame({'UPDATED Idea Status':listUpdatedIdeaStatus, 'PRIORITY': listPriority,'ECOSYSTEM':listEcoSys},columns = ['UPDATED Idea Status', 'PRIORITY', 'ECOSYSTEM'])

data_Y["UPDATED Idea Status"] = data_Y["UPDATED Idea Status"].astype('category')
data_Y["PRIORITY"] = data_Y["PRIORITY"].astype('category')
data_Y["ECOSYSTEM"] = data_Y["ECOSYSTEM"].astype('category')
data_Y["UPDATED Idea Status"] = data_Y["UPDATED Idea Status"].cat.codes
data_Y["PRIORITY"] = data_Y["PRIORITY"].cat.codes
data_Y["ECOSYSTEM"] = data_Y["ECOSYSTEM"].cat.codes

'''
title = "Learning curves(SVC)"
cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
estimator = SVC()
plot_learning_curve(estimator, title, data_X, data_Y["UPDATED Idea Status"], (0.7, 1.01), cv = cv, n_jobs = 4)
plt.show()
'''

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y["UPDATED Idea Status"], test_size = 0.2)

'''
classifier = MLPCLassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (5,2), random_state = 1)
clf = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score*100)
y_pred = classifier.predict(X_test)
confusion_matrix  = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print('Accuracy of classifier on test set : {:.2f}'.format(classifier.score(X_test, y_test)))
'''
'''
#Evaluate different models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#Evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits = 10, random_state = 7)
	cv_results = cross_val_score(model, X_train, y_train, cv = kfold, scoring = 'accuracy')
	resluts.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
'''

classifier = LogisticRegression()
clf = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score * 100)

'''
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
'''

#GridSearchCV
'''
parameter_candidates = [{'C' : [0.01, 0.1, 1, 10, 100, 1000], 'kernel' : ['poly']},{'C' : [0.01, 0.1, 1, 10, 100, 1000], 'gamma' : [0.001, 0.0001], 'kernel' : ['rbf']},]
clf = GridSearchCV(estimator = SVC(), param_grid = parameter_candidates, n_jobs = -1)
clf.fit(X_train, y_train)
print('Best score: ', clf.best_score_)
print('Best C:', clf.best_estimator_.kernel)
print('Best gamma:', clf.best_estimator_.gamma)
'''