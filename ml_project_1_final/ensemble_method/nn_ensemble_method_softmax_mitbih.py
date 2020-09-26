import sys
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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
df_test = pd.read_csv("../../input/mitbih_test.csv", header=None)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
Y_test = np.array(df_test[187].values).astype(np.int8)

# make predictions
yhats = np.asarray([model.predict(X_test) for model in models])
# sum across ensembles
summed = np.sum(yhats, axis=0)
# argmax across classes
outcomes = np.argmax(summed, axis=1)


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, outcomes)
accuracy_str = 'Accuracy: %f' % accuracy 
print(accuracy_str)
# precision tp / (tp + fp)
precision = precision_score(Y_test, outcomes, labels=range(5), average='weighted')
precision_str = 'Precision: %f' % precision 
print(precision_str)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, outcomes, labels=range(5), average='weighted')
recall_str = 'Recall: %f' % recall
print(recall_str)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, outcomes, labels=range(5), average='weighted')
f1_str = 'F1 score: %f' % f1 
print(f1_str)

# Write output results in file
out = open("output_nn_ensemble_method_softmax_mitbih.txt","w") 
out.write("Ensemble method using the following models: " + str(models_name) + "\n")
out.write("Ensemble method used: softmax")
out.write("Result metrics:" + "\n")
out.write(accuracy_str + "\n")
out.write(precision_str + "\n")
out.write(recall_str + "\n")
out.write(f1_str + "\n")
out.close() 