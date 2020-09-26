#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, SimpleRNN, Dropout, concatenate, LSTM, RNN, CuDNNLSTM
from keras.models import Sequential
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


# In[2]:


df_train = pd.read_csv("../../input/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("../../input/mitbih_test.csv", header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


# In[ ]:


def get_model():
    nclass = 5
    model = Sequential()
    #model.add(LSTM(32, input_shape=(187, 1)))
    #model.add(Dropout(rate=0.1))
    model.add(SimpleRNN(256, return_sequences=True, input_shape=(187, 1)))
    model.add(SimpleRNN(124))
    model.add(Dense(5, activation="softmax"))
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model


# In[ ]:


model = get_model()
file_path = "rnn_mitbih.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit(X, Y, epochs=1000, batch_size=512, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)


# In[ ]:


pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

f1 = f1_score(Y_test, pred_test, average="macro")

print("Test f1 score : %s "% f1)


# In[ ]:


acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)

