import pandas as pd
import numpy as np

from keras.models import Model
from keras.models import load_model
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

filename = '../rnn/rnn_mitbih.h5'
#Model trained with the MIT data set
model = load_model(filename)



nclass = 5
new_model = Model(model.input, model.layers[-2].output) #take the 2nd layer from the end

#outputs are the vector representation of the PTB Diagnostic ECG Database samples.
df_1 = pd.read_csv("../../input/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("../../input/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

ptb_representation = new_model.predict(X)

#train a new small feedforward neural network equivalent to the layers you removed from the first model.
inp = Input((ptb_representation.shape[1],))
dropout = Dropout(0.2)(inp)
#dense_0 = Dense(64)(inp)
dense_1 = Dense(1, activation=activations.sigmoid, name="dense_2")(dropout)

model2= models.Model(inputs=inp, outputs=dense_1)
opt = optimizers.Adam(0.0001)
model2.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])

model2.summary()

file_path = "transfert_learning1.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=15, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model2.fit(ptb_representation , Y, batch_size=64, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split = 0.1)  

X_test_transformed = new_model.predict(X_test)
pred_test = model2.predict(X_test_transformed)
pred_test = (pred_test>0.5).astype(np.int8)


f1 = f1_score(Y_test, pred_test, average="macro")

print("Test f1 score : %s "% f1)


acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)


