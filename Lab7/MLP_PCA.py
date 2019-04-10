import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def onehot(X,n_classes):
    Y = np.zeros([X.shape[0],n_classes],dtype = int)
    
    for i in range(X.shape[0]):
       
        Y[i,X[i,0]] = 1;

    #print(Y[:5])
    return Y


data = pd.read_csv('PaviaU_features.csv')

data = np.array(data)

X = data
print(data.shape)

Y = pd.read_csv('PaviaU-label.csv')
Y = Y - 1
n_classes = len(np.unique(Y))

Y = np.array(Y)

Y = onehot(Y,n_classes)
data = np.concatenate([X,Y],axis = 1)
#data = shuffle(data)

X = data[:,:103]
Y = data[:,103:]
Train_X,Test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.5,random_state = 0)

n_features = [103,83,63,43,23]
test_Y = test_Y.argmax(axis=1)
#print(Y.shape)
a = []
for x in n_features:
    if x!=103:
        pca = decomposition.PCA(n_components = x)

        
        train_X = pca.fit_transform(Train_X)
        test_X = pca.transform(Test_X)
    else:
        train_X = Train_X
        test_X = Test_X
    print(test_X.shape)
    print(train_X.shape)
    ##print(n_classes)



    mlp = MLP(hidden_layer_sizes = (128,64),batch_size = 128,learning_rate_init=0.001,epsilon = 1e-08,max_iter=100)
    mlp.fit(train_X,train_Y)

    pred = mlp.predict(test_X)
    pred = pred.argmax(axis=1)
    
    c_matrix = confusion_matrix(test_Y,pred)
    print(c_matrix)
    accuracy = accuracy_score(test_Y,pred)
    print('Accuracy : ',accuracy)
    a.append(accuracy)
    #precision = true positive / total predicted positive(True positive + False positive)
    #recall = true positive / total actual positive(True positive + False Negative)
    print(classification_report(test_Y,pred))

print(a)
indx = np.arange(len(n_features))
plt.bar(indx,a)
plt.xlabel('Number of Features')
plt.xticks(indx,n_features)
plt.ylabel('Accuracy')
plt.show()





