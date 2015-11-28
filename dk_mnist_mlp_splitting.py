from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(0)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, RepeatVector, Flatten, Layer
from keras.optimizers import *
from keras.utils import np_utils

def add_splitter(model):
    model.add(RepeatVector(2))
    model.add(Activation('split_relu'))
    model.add(Flatten())

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--bs", type=int, dest="bs", default=128)
parser.add_argument("--nhids", type=int, dest="nhids", default=128)
parser.add_argument("--nlayers", type=int, dest="nlayers", default=2)
parser.add_argument("--opt", type=str, dest="opt", default='rms')
parser.add_argument("--pdrop", type=float, dest="pdrop", default=.2)
parser.add_argument("--split", type=int, dest="split", default=0)
args = parser.parse_args()
locals().update(vars(args))

batch_size = bs
nb_classes = 10
nb_epoch = 60

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype("float32") / 255
X_test = X_test.reshape(10000, 784).astype("float32") / 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(nhids, input_shape=(784,)))
for n in range(nlayers - 1):
    if split:
        add_splitter(model)
    else:
        model.add(Activation('relu'))
    # TODO: splitter and dropout interaction??
    model.add(Dropout(pdrop))
    model.add(Dense(nhids))
model.add(Activation('relu'))
model.add(Dropout(pdrop))
model.add(Dense(10))
model.add(Activation('softmax'))

if opt == 'adad':
    optimizer = adadelta()
if opt == 'adag':
    optimizer = adagrad()
if opt == 'adam':
    optimizer = Adam()
if opt == 'mom':
    optimizer = SGD(momentum=.9)
if opt == 'nesterov':
    optimizer = SGD(momentum=.9, nesterov=1)
if opt == 'rms':
    optimizer = RMSprop()
if opt == 'sgd':
    optimizer = SGD()

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
