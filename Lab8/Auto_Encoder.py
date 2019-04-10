import numpy as np
from keras.layers import Dense,Input
from keras.datasets import mnist
from keras.models import Model,Sequential,load_model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt

(train_X,train_Y),(test_X,test_Y) = mnist.load_data()

train_X = train_X/255
test_X = test_X/255

train_X = np.reshape(train_X,(train_X.shape[0],train_X.shape[1]*train_X.shape[2]))
test_X = np.reshape(test_X,(test_X.shape[0],test_X.shape[1]*test_X.shape[2]))

train_Y = to_categorical(train_Y)
test_Y = to_categorical(test_Y)
hidden_size = [256,128,64,32]


input_img = Input(shape=(784,))
encode = Dense(256,activation='relu')(input_img)
output_img = Dense(784,activation='sigmoid')(encode)
encoder_1 = Model(input_img,encode)
auto_1 = Model(input_img,output_img)
auto_1.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
auto_1.fit(train_X, train_X, epochs=5,batch_size=256,validation_data=(test_X,test_X))


encoder_1.save('Encoder_1')
in_1 = encoder_1.predict(train_X)

input_img = Input(shape=(in_1.shape[1],))
encode = Dense(128,activation='relu')(input_img)
output_img = Dense(in_1.shape[1],activation='sigmoid')(encode)
encoder_2 = Model(input_img,encode)
auto_2 = Model(input_img,output_img)
auto_2.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])  
auto_2.fit(in_1, in_1, epochs=5,batch_size=256)

encoder_2.save('Encoder_2')
in_2 = encoder_2.predict(in_1)

input_img = Input(shape=(in_2.shape[1],))
encode = Dense(64,activation='relu')(input_img)
output_img = Dense(in_2.shape[1],activation='sigmoid')(encode)
encoder_3 = Model(input_img,encode)
auto_3 = Model(input_img,output_img)
auto_3.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])  
auto_3.fit(in_2, in_2, epochs=5,batch_size=256)

encoder_3.save('Encoder_3')


model = Sequential()
x = load_model('Encoder_1')
model.add(x)
x = load_model('Encoder_2')
model.add(x)
x = load_model('Encoder_3')
model.add(x)

model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])  
model.fit(train_X, train_Y, epochs=5,batch_size=256,validation_data = (test_X,test_Y))





