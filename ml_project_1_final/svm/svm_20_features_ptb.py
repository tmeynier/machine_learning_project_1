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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy import stats

# Load dataset
df_1 = pd.read_csv("../../input/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("../../input/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y_train = np.array(df_train[187].values).astype(np.int8)
X_train = np.array(df_train[list(range(187))].values)

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)

# Feature extraction
def find_features(X):

    ############################# FEATURE DANIEL #####################################

    # Analyse the peaks in the signal
    peaks = []
    peaks_index = []
    peaks_heights = []
    max_height = np.zeros((len(X),1))
    mean_height = np.zeros((len(X),1))
    var_height =np.zeros((len(X),1))
    for ecg in range(len(X)): 
        peaks.append(find_peaks(X[ecg,:],height = 0.7, distance = 10))#find peaks
        peaks_index.append(peaks[ecg][0])
        peaks_heights.append(peaks[ecg][1]['peak_heights'])
        if peaks_heights[ecg].size == 0:
            peaks_heights[ecg] = [0]
        else:
            peaks_heights[ecg] = peaks_heights[ecg]
        max_height[ecg,0] = np.max(peaks_heights[ecg]) #FEATURE: Max height of peak
        mean_height[ecg,0] = np.mean(peaks_heights[ecg])#FEATURE: Mean height of peak
        var_height[ecg,0] = np.var(peaks_heights[ecg]) #FEATURE: Variance in peak heigh

    peaks_int_max = np.zeros((len(peaks_index),1))
    peaks_int_min = np.zeros((len(peaks_index),1))
    peaks_int_med = np.zeros((len(peaks_index),1))
    peaks_int_mean = np.zeros((len(peaks_index),1))
    peaks_int = []
    for ecg in range(len(peaks_index)):
        A =[]
        for peak in range(len(peaks_index[ecg])-1):
            A.append(peaks_index[ecg][peak+1]-peaks_index[ecg][peak])
        peaks_int.append(np.array(A))
        if peaks_int[ecg].size == 0:
            peaks_int[ecg] = [0]
        else:
            peaks_int[ecg] = peaks_int[ecg]
        peaks_int_max[ecg,0] = np.max(peaks_int[ecg])# FEATURE: maximum time difference between peaks
        peaks_int_min[ecg,0] = np.min(peaks_int[ecg])# FEATURE: minimum time difference between peaks
        peaks_int_med[ecg,0] = np.median(peaks_int[ecg])# FEATURE: median time difference between peaks
        peaks_int_mean[ecg,0] = np.mean(peaks_int[ecg])#FEATURE: mean time difference between peaks
        

    # Find the signals with 'thick' peaks - peaks that last long
    peaks = []
    peaks_index_thick = []
    peaks_heights_thick = []
    peaks_thick = np.zeros((len(peaks_index),1))
    for ecg in range(len(X)):
        peaks.append(find_peaks(X[ecg,:],height = 0.3, width=20))#find thick peaks
        peaks_index_thick.append(peaks[ecg][0])
        peaks_heights_thick.append(peaks[ecg][1]['peak_heights'])
        if peaks_heights_thick[ecg].size == 0:
            peaks_heights_thick[ecg] = [0]
            peaks_index_thick[ecg] = [0]
        else:
            peaks_heights_thick[ecg] = peaks_heights_thick[ecg]
            peaks_index_thick[ecg] = peaks_index_thick[ecg]
        if peaks_heights_thick[ecg][0] == 0:
            peaks_thick[ecg]=0
        else:
            peaks_thick[ecg]=[len(peaks_index_thick[ecg])] #FEATURE: number of peaks thicker than 20 and above 0.3 in height

    #FEATURE: Total length of heartbeat
    X_time =np.zeros((len(X),1))

    for column in range(len(X)):
        row = 20
        while row < 185:
            if X[column,row]== X[column,row+1] and X[column,row+1]== X[column,row+2] and X[column,row+2]==0:
                X_time[column]=row
                row = 200
            else:
                row += 1
        if X_time[column]== 0:
            X_time[column]=188

    ############################# FEATURE TRISTAN #####################################

    X_deriv = np.gradient(X, axis=1)

    # Parameters of the feature extraction
    signal_over_threshold = 0.7
    signal_under_threshold = 0.2
    last_samples = 125
    first_samples = 25
    local_maximum_threshold = 0.95
    local_maximum_distance = 20
    last_values_deriv = 75

    # 1. Overall value of signal over "signal_over_threshold"
    overall_value_signal_over = np.asarray(np.transpose(np.sum(np.where(X> signal_over_threshold,X,0), axis = 1)))
    overall_value_signal_over = np.expand_dims(overall_value_signal_over, axis=1)
    
    # 2. Overall value of signal under "signal_under_threshold"
    overall_value_signal_under = np.transpose(np.sum(np.where(X< signal_under_threshold,X,0), axis = 1))
    overall_value_signal_under = np.expand_dims(overall_value_signal_under, axis=1)
    

    # 3. Overall value after 125 samples
    overall_value_after = np.transpose(np.sum(X[:, last_samples:], axis = 1))
    overall_value_after = np.expand_dims(overall_value_after, axis=1)
    
    # Local maximum statistic
    peaks = []
    peaks_index = []
    peaks_index_average = []
    peaks_heights = []
    peaks_number = []
    for ecg in range(len(X)): 
        peaks.append(find_peaks(X[ecg,first_samples:],height = local_maximum_threshold, distance = local_maximum_distance))#find peaks
        peaks_index.append(peaks[ecg][0])
        peaks_index_average.append(np.mean(peaks[ecg][0]))
        peaks_heights.append(peaks[ecg][1]['peak_heights'])
        peaks_number.append(len(peaks[ecg][0]))

    # 4. The number of local maximums
    # Peaks number

    # 5. The average index of the local maximums  
    # Peaks index average

    # 6. The height of the first local minimum 
    height_first_local_min = np.transpose(np.amin(X[:, :first_samples], axis = 1))
    height_first_local_min = np.expand_dims(height_first_local_min, axis=1)
    
    # 7. The minimum value of the derivative
    min_value_deriv = np.transpose(np.amin(X_deriv, axis = 1))
    min_value_deriv = np.expand_dims(min_value_deriv, axis=1)
    
    # 8. The variance of the last values of the derivative
    variance_last_values = np.transpose(np.var(X_deriv[:,last_values_deriv:], axis = 1))
    variance_last_values = np.expand_dims(variance_last_values, axis=1)
    
    dataset1 = np.hstack((max_height, mean_height,var_height, peaks_int_max, peaks_int_min, peaks_int_med, 
    peaks_int_mean,peaks_thick,X_time))
    
    dataset2 = np.hstack((overall_value_signal_over, overall_value_signal_under, overall_value_after,
    height_first_local_min, min_value_deriv, variance_last_values))

    return np.hstack((dataset1, dataset2))

def basic_pipeline(X, y, X_test):
    line = Pipeline([
        ('var', VarianceThreshold()),
        ('scale', StandardScaler()),
        ('clas', svm.SVC(decision_function_shape='ovr',class_weight='balanced'))
    ])
         
    parameters = {"clas__C":[0.5,1,2,4],
                  "clas__gamma": [0.01, 0.1]
                 }
    grid = GridSearchCV(line, parameters, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
    grid.fit(X, y)
    return grid, grid.predict(X_test)

grid_small, y_hat = basic_pipeline(find_features(X_train), Y_train, find_features(X_test))

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, y_hat)
accuracy_str = 'Accuracy: %f' % accuracy 
print(accuracy_str)
# precision: tp / (tp + fp)
precision = precision_score(Y_test, y_hat)
precision_str = 'Precision: %f' % precision
print(precision_str)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, y_hat)
recall_str = 'Recall: %f' % recall
print(recall_str)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, y_hat)
f1_str = 'F1 score: %f' % f1
print('F1 score: %f' % f1)
# auroc
roc = roc_auc_score(Y_test, y_hat)
roc_str = "Test auroc score : %s "% roc
print(roc_str)
# auprc
auprc = average_precision_score(Y_test, y_hat)
auprc_str = "Test auprc score : %s "% auprc
print(auprc_str)

# Write output results in file
out = open("output_svm_20_features_ptb.txt","w") 
out.write("Result metrics:" + "\n")
out.write(accuracy_str + "\n")
out.write(precision_str + "\n")
out.write(recall_str + "\n")
out.write(f1_str + "\n")
out.write(roc_str + "\n")
out.write(auprc_str + "\n")
out.close() 
