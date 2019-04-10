import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout

from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt


batch_size = 64
n_classes = 10
epochs = 10

(train_X,train_Y),(test_X,test_Y) = mnist.load_data()
train_Y = to_categorical(train_Y) 
test_Y = to_categorical(test_Y)

train_X = train_X/255
test_X = test_X/255
train_Y = np.reshape(train_Y,(-1,10))
test_Y = np.reshape(test_Y,(-1,10))

#print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)

input_shape = (28,28,1)
train_X = np.reshape(train_X,(-1,28,28,1))
test_X = np.reshape(test_X,(-1,28,28,1))
z = [1,2,3,4,5]
accuracies = []
for i in z:
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation = 'relu',padding = 'same',input_shape = input_shape))
    model.add(MaxPooling2D(pool_size=(3,3)))
    for j in range(i-1):
        model.add(Conv2D(16,kernel_size=(3,3),activation = 'relu',padding='same'))
        
        
    
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes,activation = 'softmax'))
    print(model.summary())
    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    print('Training : ')
    model.fit(train_X,train_Y,epochs = epochs,batch_size = batch_size)
    scores = model.evaluate(test_X,test_Y,verbose = 0)
    print("Accuracy: %.4f%%" % (scores[1]*100))
    accuracies.append(scores[1])
    model.save('ConVNet_MNIST_'+str(i))

indx = np.arange(len(z))
plt.bar(indx,accuracies)
plt.xlabel('Number of Conv Layers')
plt.xticks(indx,z)
plt.ylabel('Accuracy')
plt.show()


