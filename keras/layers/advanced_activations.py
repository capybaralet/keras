from ..layers.core import Layer
from ..utils.theano_utils import shared_zeros

#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy.random


class LeakyReLU(Layer):
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.params = []

    def output(self, train):
        X = self.get_input(train)
        return ((X + abs(X)) / 2.0) + self.alpha * ((X - abs(X)) / 2.0)


class PReLU(Layer):
    '''
        Reference: 
            Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
                http://arxiv.org/pdf/1502.01852v1.pdf
    '''
    def __init__(self, input_shape):
        self.alphas = shared_zeros(input_shape)
        self.params = [self.alphas]

    def output(self, train):
        X = self.get_input(train)
        pos = ((X + abs(X)) / 2.0)
        neg = self.alphas * ((X - abs(X)) / 2.0)
        return pos + neg


class Permutation(Layer):
    ''' Permute input activations (assumed flat).  Keeps a list of permutations and applies the last element.'''
    def __init__(self, input_len):#, random_seed=123):
        self.input_len = input_len
        #self.rng = RandomStreams(random_seed)
        self.permutations = [numpy.random.permutation(self.input_len)]
        self.params = []

    def reshuffle(self):
        self.permutations.append(numpy.random.permutation(self.input_len))

    def output(self, train):
        X = self.get_input(train)
        return X[:,self.permutations[-1]]



