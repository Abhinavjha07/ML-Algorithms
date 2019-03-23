import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import math
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

data = pd.read_csv('iris.csv')
#print(data)
data = np.array(data)


def distance(c,x,k):
    d = np.zeros((k,1))
    for i in range(k):
        d[i,0] = np.linalg.norm(x-c[i,:])

    return d

def update(X,assign,k):
    n_c = np.zeros((k,X.shape[1]))
    for i in range(k):
        cnt = 0
        p = np.zeros((1,X.shape[1]))
        for j in range(X.shape[0]):
            if assign[j,0] == i:
                cnt += 1
                p[0,:] = p[0,:] + X[j,:]
        #print(p,cnt)
        n_c[i,:] = p[0,:] / cnt
    #print(n_c)
    return n_c

def k_means(X,Y,k):
    x= random.sample(range(0,X.shape[0]),3)
    
    c = X[x,:]
    
    #print(c)
    itr = 100
    for i in range(itr):
        assign = np.zeros((X.shape[0],1),dtype = int)
        for j in range(X.shape[0]):
            dist = distance(c,X[j,:],k)
            
            assign[j,0] = np.argmin(np.array(dist))
        
        new_c = update(X,assign,k)
        x = 0
##        print()
##        print(c)
##        print(new_c)
##        print()
        for j in range(k):
            for l in range(X.shape[1]):
                if new_c[j,l] == c[j,l]:
                    x += 1

        if x == k*X.shape[1]:
            print('Last Step : ',i)
            break

        c = new_c
        
    return c

def transform(X,centers,variance):
    data = np.zeros((X.shape[0],centers.shape[0]))
    for i in range(X.shape[0]):
        for j in range(centers.shape[0]):
            data[i,j] = math.exp(-(((np.linalg.norm(X[i,:]-centers[j,:])**2/(2*variance[j,0])))))

    return data

def onehot(X,n_classes):
    Y = np.zeros([X.shape[0],n_classes],dtype = int)
    
    for i in range(X.shape[0]):
       
        Y[i,X[i,0]] = 1;

    #print(Y[:5])
    return Y



X = np.array(data[:,0:4])
X = normalize(X)
#print(X)
Y = data[:,4:]
label = []
for x in Y:
    if x == 'Iris-setosa':
        label.append(0)
    elif x == 'Iris-virginica':
        label.append(1)
    else:
        label.append(2)
n_classes = 3
Y = np.array(label)
Y = np.reshape(Y,(Y.shape[0],1))
Y = onehot(Y,n_classes)
data = np.concatenate([X,Y],axis = 1)
data = shuffle(data)
X = data[:,:4]
Y = data[:,4:]
#print(X.shape,Y.shape)
n_clusters = 3

centers = k_means(X,Y,n_clusters)
#print(centers.shape)

variance = np.zeros((n_clusters,1))
for i in range(n_clusters):
    t = 0
    for j in range(centers.shape[0]):
        t+= (np.linalg.norm(centers[j,:]-centers[i,:]))**2

    t /= n_clusters
    variance[i,0] = t

#print(variance) 

X = transform(X,centers,variance)
print(X)

clf = Perceptron(max_iter = 1000,alpha = 0.001)

clf.fit(X,Y.argmax(axis=1))
y = clf.predict(X)

c_matrix = confusion_matrix(Y.argmax(axis=1),y)
print(c_matrix)

accuracy = accuracy_score(Y.argmax(axis=1),y)
print(accuracy)

