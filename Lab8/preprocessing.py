import cv2
import numpy as np
from tqdm import tqdm
import os
from keras.utils import to_categorical

dir_name = '101_ObjectCategories'
classes = dict()
x = 0
X = []
Y = []
for d in tqdm(os.listdir(dir_name)):
    classes.update({d:x})
    x += 1

    for img in tqdm(os.listdir(dir_name+'/'+d)):
        
        im = cv2.imread(dir_name+'/'+d+'/'+img,1)
        
        #print(img)
        im = cv2.resize(im,(224,224),interpolation = cv2.INTER_LINEAR)
        #print(im)
        X.append(np.array(im))
        Y.append(classes[d])

X = np.array(X)
Y = np.array(Y)
Y = np.reshape(Y,(-1,1))
Y = to_categorical(Y,dtype=int)
np.save('X.npy',X)
np.save('Y.npy',Y)
print(X.shape,Y.shape)
#print(classes)
print(X[:2])
print(Y[:2])
        
