from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility


import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--nlabeled",              type=int,   dest="nlabeled",            default=10000,           help="")
parser.add_argument("--margin",             type=float, dest="margin",              default=1.,              help="")
parser.add_argument("--ul_weight",             type=float, dest="ul_weight",           default=.5,              help="")
args = parser.parse_args()
locals().update(vars(args))


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.objectives import get_ssl_objective

'''
    Train a simple convnet on the MNIST dataset.

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

    Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
'''

batch_size = 128
nb_classes = 10
nb_epoch = 1000

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

Y_train[:-nlabeled] = 0

# TODO: add noise (.3, .45, .6) (additive gaussian noise everywhere...)
model = Sequential()
#
model.add(Convolution2D(32, 5, 5, border_mode='full', input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#
model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(10, 1, 1))
model.add(Activation('relu'))
# TODO: global mean pool
#
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
ssl_cost_fn = get_ssl_objective(margin=margin, ul_weight=ul_weight)
# WHAT OPTIMIZER??
model.compile(loss=ssl_cost_fn, optimizer='adadelta')
#model.compile(loss="ssl_objective", optimizer='adadelta')


model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
