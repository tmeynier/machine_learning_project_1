import sys
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# README: Use the command line argument to put the paths of the h5 files
# resulting from the models you want to average

# load pre-trained ensemble members
models = list()
models_name = list()
for i in range(len(sys.argv)-1):
	# load model
	filename = sys.argv[i+1]
	models_name.append(filename)
	model = load_model(filename)
	# store in memory
	models.append(model)

# Load test dataset
df_1 = pd.read_csv("../../input/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("../../input/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

Y_test = np.array(df[187].values).astype(np.int8)
X_test = np.array(df[list(range(187))].values)[..., np.newaxis]

# make predictions
yhats = np.asarray([model.predict(X_test) for model in models])

# sum across ensembles
summed = np.sum(yhats, axis=0)


# argmax across classes
outcomes = (summed>0.5*len(models)).astype(np.int8)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, outcomes)
accuracy_str = 'Accuracy: %f' % accuracy 
print(accuracy_str)
# precision: tp / (tp + fp)
precision = precision_score(Y_test, outcomes)
precision_str = 'Precision: %f' % precision
print(precision_str)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, outcomes)
recall_str = 'Recall: %f' % recall
print(recall_str)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, outcomes)
f1_str = 'F1 score: %f' % f1
print('F1 score: %f' % f1)
# auroc
roc = roc_auc_score(Y_test, outcomes)
roc_str = "Test auroc score : %s "% roc
print(roc_str)
# auprc
auprc = average_precision_score(Y_test, outcomes)
auprc_str = "Test auprc score : %s "% auprc
print(auprc_str)

# Write output results in file
out = open("output_nn_ensemble_method_softmax_ptbdb.txt","w") 
out.write("Ensemble method using the following models: " + str(models_name) + "\n")
out.write("Ensemble method used: softmax" + "\n")
out.write("Result metrics:" + "\n")
out.write(accuracy_str + "\n")
out.write(precision_str + "\n")
out.write(recall_str + "\n")
out.write(f1_str + "\n")
out.write(roc_str + "\n")
out.write(auprc_str + "\n")
out.close() 

