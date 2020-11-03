from abc import abstractmethod
import torch
import numpy as np

class optimizer(object):
    @abstractmethod
    def compute_update(self, gradients):
        raise NotImplementedError('compute_update function is not implemented.')

class SGDOptimizer(optimizer):
    def __init__(self, args):
        self.learning_rate = args.learning_rate

    def update(self, model):
        for p in model.parameters():
            p.grad *= self.learning_rate

class MomentumSGDOptimizer(optimizer):
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.rho = args.rho
        self.m = None

    def update(self, model):
        if self.m is None:
            self.m = [torch.zeros(p.size()) for p in model.parameters()]

        for i, p in enumerate(model.parameters()):
            self.m[i] = self.rho * self.m[i] + p.grad
            p.grad = self.learning_rate * self.m[i]

class AdagradOptimizer(optimizer):
    def __init__(self, args):
        self.delta = args.delta
        self.learning_rate = args.learning_rate
        self.r = None

    def update(self, model):
        if self.r is None:
            self.r = [torch.zeros(p.size()) for p in model.parameters()]

        for i, p in enumerate(model.parameters()):
            ## TODO
            self.r[i]=self.r[i]+p.grad*p.grad
            p.grad=p.grad*self.learning_rate/(self.delta+self.r[i]**0.5)
            
            #raise NotImplementedError('You should write your code HERE')


class RMSPropOptimizer(optimizer):
    def __init__(self, args):
        self.tau = args.tau
        self.learning_rate = args.learning_rate
        self.r = None
        self.delta = args.delta

    def update(self, model):
        if self.r is None:
            self.r = [torch.zeros(p.size()) for p in model.parameters()]

        for i, p in enumerate(model.parameters()):
            ## TODO
            self.r[i]=self.r[i]*self.tau+(1-self.tau)*p.grad*p.grad
            p.grad = p.grad * self.learning_rate / (self.delta + self.r[i] ** 0.5)

            #raise NotImplementedError('You should write your code HERE')


class AdamOptimizer(optimizer):
    def __init__(self, args):
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.learning_rate = args.learning_rate
        self.delta = args.delta
        self.iteration = None
        self.m1 = None
        self.m2 = None

    def update(self, model):
        if self.m1 is None:
            self.m1 = [torch.zeros(p.grad.size()) for p in model.parameters()]
        if self.m2 is None:
            self.m2 = [torch.zeros(p.grad.size()) for p in model.parameters()]
        if self.iteration is None:
            self.iteration = 1

        for i, p in enumerate(model.parameters()):
            ## TODO
            self.m1[i]=self.m1[i]*self.beta1+(1-self.beta1)*p.grad
            self.m2[i]=self.m2[i]*self.beta2+(1-self.beta2)*p.grad*p.grad
            m1_bias=self.m1[i]/(1-self.beta1**self.iteration)
            m2_bias=self.m2[i]/(1-self.beta2**self.iteration)
            p.grad=(self.learning_rate * m1_bias) / (self.delta + np.sqrt(m2_bias))
            #raise NotImplementedError('You should write your code HERE')

        self.iteration = self.iteration+1



def createOptimizer(args):
    if args.optimizer == "sgd":
        return SGDOptimizer(args)
    elif args.optimizer == "momentumsgd":
        return MomentumSGDOptimizer(args)
    elif args.optimizer == "adagrad":
        return AdagradOptimizer(args)
    elif args.optimizer == "rmsprop":
        return RMSPropOptimizer(args)
    elif args.optimizer == "adam":
        return AdamOptimizer(args)
