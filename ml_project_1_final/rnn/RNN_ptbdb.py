#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, SimpleRNN, GRU, Input, Dropout, concatenate, LSTM, RNN, Embedding
from keras.models import Sequential
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


# In[4]:


df_1 = pd.read_csv("../../input/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("../../input/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


# In[11]:


def get_model():
    nclass = 1
    model = Sequential()
    model.add(SimpleRNN(128, return_sequences=True, input_shape=(187, 1)))
    model.add(SimpleRNN(64))
    model.add(Dense(1, activation="sigmoid"))
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


# In[ ]:


model = get_model()
file_path = "rnn_ptbdb.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit(X, Y, epochs=1000, verbose=2, batch_size=512, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)


# In[7]:


pred_test = model.predict(X_test)
pred_test = (pred_test>0.5).astype(np.int8)

f1 = f1_score(Y_test, pred_test)

print("Test f1 score : %s "% f1)


# In[8]:


acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)


# In[9]:


roc = roc_auc_score(Y_test, pred_test)

print("Test auroc score : %s "% roc)


# In[10]:


auprc = average_precision_score(Y_test, pred_test)

print("Test auprc score : %s "% auprc)


# In[ ]:




