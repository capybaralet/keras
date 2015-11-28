from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianDropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.constraints import maxnorm

'''
TODO:
    early stopping on validation set
'''

batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(GaussianDropout(.8, input_shape=(784,)))
model.add(Dense(8192, W_constraint=maxnorm(1.936)))
model.add(Activation('relu'))
model.add(GaussianDropout(.5))
model.add(Dense(8192, W_constraint=maxnorm(1.936)))
model.add(Activation('relu'))
model.add(GaussianDropout(.5))
model.add(Dense(10, W_constraint=maxnorm(1.936)))
model.add(Activation('softmax'))

if 0:
    model.compile(loss='categorical_crossentropy', optimizer='adam')
if 0:
    rms = rmsprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)
if 1:
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.95, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

#assert False

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
