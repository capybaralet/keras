from __future__ import absolute_import
import theano.tensor as T


class Regularizer(object):
    def set_param(self, p):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__}


class WeightRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        loss += T.sum(abs(self.p)) * self.l1
        loss += T.sum(self.p ** 2) * self.l2
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}


class ActivityRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0., sparse_filtering=0.):
        self.l1 = l1
        self.l2 = l2
        self.sparse_filtering = sparse_filtering

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        loss += self.l1 * T.sum(T.mean(abs(self.layer.get_output(True)), axis=0))
        loss += self.l2 * T.sum(T.mean(self.layer.get_output(True) ** 2, axis=0))
        if self.sparse_filtering: # FIXME (wtf? no test values!?!?!)
            tvar = self.layer.get_output(True)
            f_tilde = tvar / tvar.norm(L=2, axis=0).dimshuffle('x', 0, 1, 2) 
            import ipdb; ipdb.set_trace()
            f_hat = f_tilde / f_tilde.norm(L=2, axis=1).dimshuffle(0, 'x', 1, 2) 
            loss += self.sparse_filtering * T.mean(f_hat.norm(L=1, axis=1)) 
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2,
                "sparse_filtering": self.sparse_filtering}


# OBSOLETE
class ConvSparseFilteringRegularizer(Regularizer):
    def __init__(self, l=0.):
        self.l = l

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        tvar = self.layer.get_output(True)
        f_tilde = tvar / tvar.norm(L=2, axis=0).dimshuffle('x', 0, 1, 2) 
        f_hat = f_tilde / f_tilde.norm(L=2, axis=1).dimshuffle(0, 'x', 1, 2) 
        loss += self.l * T.mean(f_hat.norm(L=1, axis=1)) 
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}


def l1(l=0.01):
    return WeightRegularizer(l1=l)


def l2(l=0.01):
    return WeightRegularizer(l2=l)


def l1l2(l1=0.01, l2=0.01):
    return WeightRegularizer(l1=l1, l2=l2)


def activity_l1(l=0.01):
    return ActivityRegularizer(l1=l)


def activity_l2(l=0.01):
    return ActivityRegularizer(l2=l)


def activity_l1l2(l1=0.01, l2=0.01):
    return ActivityRegularizer(l1=l1, l2=l2)

identity = Regularizer

from .utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer', instantiate=True, kwargs=kwargs)
