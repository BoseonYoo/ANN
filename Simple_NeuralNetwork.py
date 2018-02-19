import numpy as np
from collections import OrderedDict


def softmax(x):
    temp = x.T
    rst = np.exp(temp) / np.sum(np.exp(temp), axis = 0)
    return rst.T

def cross_entropy_error(y, t):
    batch_size = y.shape[0]
    if y.size == t.size :
        t = np.argmax(t, axis=1)
        
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
    

class AffineLayer:
    def __init__(self, num_input, hidden_nodes, gen_coef, lr):
        self.w = gen_coef * np.random.randn(num_input, hidden_nodes)
        self.b = np.zeros(hidden_nodes)
        self.x = None
        self.dw = None
        self.db = None
        self.lr = lr
        
    def forward(self, x):
        self.x = x
        y = np.dot(self.x, self.w) + self.b
        
        return y
    
    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        self.w = self.w - self.lr * self.dw
        self.b = self.b - self.lr * self.db
        return dx

class ReLU:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        x[self.mask] = 0
        
        return x
    
    def backward(self, x):
        dx = x
        dx[self.mask] = 0
        
        return dx
  
class Softmax:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self):
        batch_size = self.y.shape[0]
        return (self.y - self.t) / batch_size
    

class NeuralNet:
    def __init__(self, num_input, hidden_nodes, num_class, gen_coef, lr):
        self.layers = OrderedDict()
        self.layers['input_layer'] = AffineLayer(num_input, hidden_nodes, gen_coef, lr)
        self.layers['r1'] = ReLU()
        self.layers['output_layer'] = AffineLayer(hidden_nodes, num_class, gen_coef, lr)
        self.last_layer = Softmax()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        
        return self.last_layer.forward(y, t)
    
    def learn(self, x, t):
        error = self.loss(x, t)
        
        layers = list(self.layers.values())
        layers.reverse()
        
        dout = self.last_layer.backward()
        
        for layer in layers:
            dout = layer.backward(dout)
            
        return error
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy