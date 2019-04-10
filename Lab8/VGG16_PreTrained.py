import numpy as np
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from sklearn.model_selection import train_test_split

vgg = VGG16()
vgg.summary()
for layer in vgg.layers[:-1]:
    layer.trainable = False

epochs = 10
batch_size = 128
X = np.load('/content/drive/My Drive/X.npy')
Y = np.load('/content/drive/My Drive/Y.npy')
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0)

model = Sequential()

model.add(vgg)

model.add(Dense(102,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.fit(train_X,train_Y,epochs = epochs,batch_size = batch_size,validation_data = (test_X,test_Y))
scores = model.evaluate(test_X,test_Y,verbose = 0)
print("Accuracy: %.4f%%" % (scores[1]*100))

model.save('VGG_pretrained')



