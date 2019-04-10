import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
X = np.load('X.npy')
Y = np.load('Y.npy')

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33, random_state=42)
model = Sequential()
model.add(Conv2D(filters=96,input_shape=(150,150,3),kernel_size = (11,11),activation='relu',strides = (4,4),padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))

model.add(Conv2D(filters=256,kernel_size=(11,11),strides=(1,1),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu',padding=’valid’))

model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding=’valid’))

model.add(Conv2D(filters=256,activation='relu', kernel_size=(3,3), strides=(1,1), padding=’valid’))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’))

model.add(Flatten())

model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(4096,activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(1000,activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(102,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.fit(train_X,train_Y,epochs = epochs,batch_size = batch_size)
scores = model.evaluate(test_X,test_Y,verbose = 0)
print("Accuracy: %.4f%%" % (scores[1]*100))
model.save('AlexNet_Mine')
