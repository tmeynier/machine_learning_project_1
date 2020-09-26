from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.manifold import TSNE


filename = '../baseline/baseline_cnn_mitbih.h5'
model = load_model(filename)

df_train = pd.read_csv("../../input/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("../../input/mitbih_test.csv", header=None)


Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[...,np.newaxis]

n = X.shape[0]

target_dict = {0:'N' , 1:'S' , 2:'V', 3:'F' , 4:'Q'}
target_ind = target_dict.keys()
target_val = target_dict.values()


index_tot = np.array([])


for i in target_ind :
    n_int = np.argwhere(Y==i)[:,0]
    index_int = np.random.permutation(n_int)[:600]
    #index_int = np.random.permutation(n_int)[:10]
    index_tot = np.concatenate( (index_tot, index_int) )

index_tot = index_tot.astype(int)

X = X[index_tot]
Y = Y[index_tot]

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
ax.set_title('2 component TSNE for mit', fontsize = 20)
#targets = ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
#targets = target_ind #range(4)

colors = ['r', 'g', 'b' , 'y' , 'm'] #k = black #m = magenta
for target, color in zip(target_ind , colors):
    indicesToKeep = labels == target
    indicesToKeep = indicesToKeep.values[:,0]
    ax.scatter(finalDf.loc[indicesToKeep, 'component 1']
               , finalDf.loc[indicesToKeep, 'component 2']
               , c = color
               , s = 50)
ax.legend(target_val)
ax.grid()

plt.savefig('plot_tnse_mit.png')
plt.show()