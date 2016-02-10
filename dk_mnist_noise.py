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
parser.add_argument("--bs", type=int, dest="bs", default=100)
parser.add_argument("--epochs", type=int, dest="epochs", default=40)
parser.add_argument("--nhid", type=int, dest="nhid", default=1000)
parser.add_argument("--nlayers", type=int, dest="nlayers", default=2)
parser.add_argument("--opt", type=str, dest="opt", default='adam')
parser.add_argument("--pdrop", type=float, dest="pdrop", default=.5)
split = 0
# noise: random incorrrect label
#parser.add_argument("--noised", type=float, dest="noised", default=0)
parser.add_argument("--tr_noised", type=int, dest="tr_noised", default=0)
parser.add_argument("--te_noised", type=int, dest="te_noised", default=0)
# for now this is 1/x - 1 (where x in the average declared noisines) (TODO)
parser.add_argument("--noise_penalty", type=float, dest="noise_penalty", default=1)
args = parser.parse_args()
locals().update(vars(args))
#tr_noised = int(60000 * noised)
#te_noised = int(10000 * noised)

batch_size = bs
nb_classes = 10
nb_epoch = epochs

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype("float32") / 255
X_test = X_test.reshape(10000, 784).astype("float32") / 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# noise the labels
y_train[:tr_noised] = (y_train[:tr_noised] + np.random.randint(1, 10, tr_noised)) % 10
y_test[:te_noised] = (y_test[:te_noised] + np.random.randint(1, 10, te_noised)) % 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)




model = Sequential()
model.add(Dense(nhid, input_shape=(784,)))
for n in range(nlayers - 1):
    if split:
        add_splitter(model)
    else:
        model.add(Activation('relu'))
    # TODO: splitter and dropout interaction??
    model.add(Dropout(pdrop))
    model.add(Dense(nhid))
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
