from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

filename = '../baseline/baseline_cnn_ptbdb.h5'
model = load_model(filename)

df_1 = pd.read_csv("../../input/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("../../input/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[... , np.newaxis]

n = X.shape[0]

target_dict = {0:'Normal' , 1:'Abnormal'}
target_ind = target_dict.keys()
target_val = target_dict.values()

#index = np.random.choice(n,2000)
#X = X[index]
#Y=Y[index]

get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[18].output])
t =get_layer_output([X])[0]

tsne = TSNE(n_components=2)
components = tsne.fit_transform(t)

df = pd.DataFrame(data = components,
                          columns = ['component 1' ,
                                    'component 2'])

labels = pd.DataFrame(Y)

finalDf = pd.concat([df, labels] , axis=1)

fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Component 1', fontsize = 15)
ax.set_ylabel('Component 2', fontsize = 15)
ax.set_title('2 component TSNE for ptb', fontsize = 20)

colors = ['r', 'g'] #k = black #m = magenta
for target, color in zip(target_ind , colors):
    #print(target)
    #indicesToKeep = finalDf[187] == target
    indicesToKeep = labels == target
    indicesToKeep = indicesToKeep.values[:,0]
    #print(indicesToKeep)
    ax.scatter(finalDf.loc[indicesToKeep, 'component 1']
               , finalDf.loc[indicesToKeep, 'component 2']
               , c = color
               , s = 50)
ax.legend(target_val)
ax.grid()

plt.savefig('plot_tnse_ptb.png')
plt.show()