from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from six.moves import range

if theano.config.floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7


# returns max(cost - margin, 0)
def get_margin_cost(cost_fn, margin, batchwise=False):
    if batchwise: # then we only need the mean margin to be large enough
        def cost(y_true, y_pred):
            cost = cost_fn(y_true, y_pred).mean(axis=0)
            return (cost - margin) * (cost > margin)
    else:
        def cost(y_true, y_pred):
            cost = cost_fn(y_true, y_pred)
            return (cost - margin) * (cost > margin)
    return cost

def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean(axis=-1)


def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean(axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    return T.abs_((y_true - y_pred) / T.clip(T.abs_(y_true), epsilon, np.inf)).mean(axis=-1) * 100.


def mean_squared_logarithmic_error(y_true, y_pred):
    return T.sqr(T.log(T.clip(y_pred, epsilon, np.inf) + 1.) - T.log(T.clip(y_true, epsilon, np.inf) + 1.)).mean(axis=-1)


def squared_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean(axis=-1)


def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean(axis=-1)


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    return cce

cce = categorical_crossentropy

def binary_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(y_pred, y_true).mean(axis=-1)
    return bce


# what is this??
def poisson_loss(y_true, y_pred):
    return T.mean(y_pred - y_true * T.log(y_pred + epsilon), axis=-1)


# applies cce for labeled examples, and [entropy - k]_+ for unlabeled
def get_ssl_objective(margin=1., ul_weight=.5):
    def cost_fn(y_true, y_pred):
        islabeled = T.sum(y_true, axis=-1)
        sl_cost = islabeled * cce(y_true, y_pred)
        ul_cost = cce(y_pred, y_pred)
        ul_cost = T.maximum(ul_cost - margin, 0.)
        ul_cost *= (1 - islabeled)
        cost = ul_weight * ul_cost + (1 - ul_weight) * sl_cost
        return cost
    return cost_fn

def ssl_objective(y_true, y_pred, margin=1., ul_weight=.5):
    islabeled = T.sum(y_true, axis=-1)
    sl_cost = islabeled * cce(y_true, y_pred)
    ul_cost = cce(y_pred, y_pred)
    ul_cost = T.maximum(ul_cost - margin, 0.)
    ul_cost *= (1 - islabeled)
    cost = ul_weight * ul_cost + (1. - ul_weight) * sl_cost
    return cost



# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
