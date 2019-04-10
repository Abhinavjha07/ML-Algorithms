import numpy as np
from keras.layers import Dense,Input
from keras.datasets import mnist
from keras.models import Model
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

##input_img = Input(shape=(784,))
##   
##code = Dense(256,activation='relu')(input_img)
##output_img = Dense(784,activation = 'sigmoid')(code)
#Accuracy : 0.8154 , Loss = 0.0680

input_img = Input(shape=(784,))
encode = Dense(256,activation='relu')(input_img)
encode = Dense(128,activation='relu')(code)
encode = Dense(64,activation='relu')(code)
encode = Dense(32,activation = 'relu')(code)
decode = Dense(64,activation='relu')(encode)
decode = Dense(128,activation='relu')(decode)
decode = Dense(256,activation='relu')(decode)
output_img = Dense(784,activation = 'sigmoid')(decode)
##output = Dense(10,activation='softmax')(code)
encoder = Model(input_img,encode)
autoencoder = Model(input_img,output_img)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])  #mean_squared_error 
autoencoder.fit(train_X, train_X, epochs=50,batch_size=256,validation_data=(test_X,test_X))
encoder.save('Encoder_MNIST')

encoded_imgs = encoder.predict(X_test)
predicted = autoencoder.predict(X_test)

plt.figure(figsize=(40, 4))
for i in range(10):
    # display original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display encoded images
    ax = plt.subplot(3, 20, i + 1 + 20)
    plt.imshow(encoded_imgs[i].reshape(8,4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# display reconstructed images
    ax = plt.subplot(3, 20, 2*20 +i+ 1)
    plt.imshow(predicted[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  
    
plt.show()


##score = autoencoder.evaluate(test_X,test_X,verbose=0)
##print("Accuracy: %.4f%%" % (score[1]*100))
##pred = autoencoder.predict(test_X)
##c_matrix = confusion_matrix(test_Y.argmax(axis=1),pred.argmax(axis=1))
##print(c_matrix)
##accuracy = accuracy_score(test_Y.argmax(axis=1),pred.argmax(axis=1))
##print('Accuracy : ',accuracy)
###precision = true positive / total predicted positive(True positive + False positive)
###recall = true positive / total actual positive(True positive + False Negative)
##print(classification_report(test_Y.argmax(axis=1),pred.argmax(axis=1)))
##
##autoencoder.save('AutoEncoder_Model_MNIST')


