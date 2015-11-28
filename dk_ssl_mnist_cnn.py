from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility


import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--nlabeled",              type=int,   dest="nlabeled",            default=50000,           help="")
parser.add_argument("--margin",                type=float, dest="margin",              default=1.,              help="")
parser.add_argument("--ul_weight",             type=float, dest="ul_weight",           default=.5,              help="")
args = parser.parse_args()
locals().update(vars(args))


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.objectives import get_ssl_objective


# FIXME: cost on unlabeled examples is not 0 for unlabeled weight = 0???

# NTS: think about how to use SL cost and/or entropy to set UL cost weight...
# NTS: wow, keras model sucks... doesn't give you access to any of the symbolic things you want...
'''
TODO: 

    early stopping

    logging
        save best
        learning curves
        ul/sl cost terms
        entropy
    compare hparams
        optimizer
        margin
        cost weight (schedule)
            vs. sampling ratio (schedule)

    make training with few labels efficient
'''

batch_size = 128
nb_classes = 10
nb_epoch = 1000

### DATA ###
# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# make validation set
X_valid = X_train[-10000:]
X_train = X_train[:-10000]
Y_valid = Y_train[-10000:]
Y_train = Y_train[:-10000]
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')

# Remove labels
Y_train[nlabeled:] = 0

### MODEL ###
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='full',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

### TRAIN ###
ssl_cost = get_ssl_objective(margin=margin, ul_weight=ul_weight)
#xsym = T.matrix('xsym') 
#ysym = T.matrix('ysym')
#cost_fn = theano.function([xsym, ysym], ssl_cost(ysym, model.y_test))
model.compile(loss=ssl_cost, optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))

if 0:
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
