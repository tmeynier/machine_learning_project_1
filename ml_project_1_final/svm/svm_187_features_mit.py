import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from imblearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy import stats

# Load dataset
df_train = pd.read_csv("../../input/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("../../input/mitbih_test.csv", header=None)

Y_train = np.array(df_train[187].values).astype(np.int8)
X_train = np.array(df_train[list(range(187))].values)

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)

def basic_pipeline(X, y, X_test):
    line = Pipeline([
        ('var', VarianceThreshold()),
        ('scale', StandardScaler()),
        ('clas', svm.SVC(decision_function_shape='ovr',class_weight='balanced'))
    ])
         
    
    parameters = {"clas__C": [2],
                  "clas__gamma": [0.01]
                 }
    grid = GridSearchCV(line, parameters, scoring='balanced_accuracy', cv=5, n_jobs=-1, verbose=1)
    grid.fit(X, y)
    return grid, grid.predict(X_test)

grid_small, y_hat = basic_pipeline(X_train, Y_train, X_test)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, y_hat)
accuracy_str = 'Accuracy: %f' % accuracy 
print(accuracy_str)
# precision tp / (tp + fp)
precision = precision_score(Y_test, y_hat, labels=range(5), average='weighted')
precision_str = 'Precision: %f' % precision 
print(precision_str)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, y_hat, labels=range(5), average='weighted')
recall_str = 'Recall: %f' % recall
print(recall_str)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, y_hat, labels=range(5), average='weighted')
f1_str = 'F1 score: %f' % f1 
print(f1_str)

# Write output results in file
out = open("output_svm_187_features_mit.txt","w") 
out.write("Result metrics:" + "\n")
out.write(accuracy_str + "\n")
out.write(precision_str + "\n")
out.write(recall_str + "\n")
out.write(f1_str + "\n")
out.close() 
