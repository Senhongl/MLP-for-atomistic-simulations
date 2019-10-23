import autograd.numpy as np


class Activation(object):

    """
    Interface for activation functions (non-linearities).
    """


    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented
        
class Tanh(Activation):

    """
    Sigmoid non-linearity
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state 

    def derivative(self):
        return 1 - self.state**2
    

class neuralnetwork:
    def __init__(self, sizes, act = 'sigmoid', seeds = None):
        self.act = act
        self.sizes = sizes
        self.weights = self.initialization(self.sizes, seeds)
        self.activations = []

    def initialization(self, sizes, seed = None):
        """
        Initialize the weights and biases parameters with random seed.
        Parameters:
        -----------
        sizes: list or np.ndarray
            The shape will be the layer size of the neural network and the value will be
            the hidden units of the corresponding layer.

            >>> layer_size = array([1, 2, 3, 1])

            whihc means that there are 2 hidden layers and the hidden units will be 2 and
            3, respectively. The input and output will always be 1, which is the fingerprint 
            of an atom and atomic energy, respectively.
        Returns:
        ------------
        A weights matrix and a biases matrix
        """
        rs = np.random.RandomState(seed=seed)
        weights = []
        for i in range(len(sizes) - 1):
            weights += list(rs.randn((sizes[i] + 1) * sizes[i + 1]))
        return np.stack(weights)
    

    
    def feedforward(self, weights, inputs, mask):
        w_index = 0
        x = np.array(inputs)
        for i in range(len(self.sizes) - 1):
            if len(self.activations) <= i:
                activation = Tanh()
                self.activations.append(activation)
            else:
                activation = self.activations[i]
                
            weight = weights[w_index : w_index + self.sizes[i] * self.sizes[i + 1]]
            bias = weights[w_index + self.sizes[i] * self.sizes[i + 1] : w_index + (self.sizes[i] + 1) * self.sizes[i + 1]][None, :]
            w_index += (self.sizes[i] + 1) * self.sizes[i + 1]
            weight = weight.reshape(self.sizes[i], self.sizes[i + 1])
            if i < len(self.sizes) - 2:
                x = activation((x @ weight) + bias) 
            else:
                x = x @ weight + bias
        x *= mask
        return x

    def dEdfp(self, weights, mask):
        w_index = len(weights) - (self.sizes[-2] + 1) * self.sizes[-1]
        derivative = weights[w_index : -self.sizes[-1]].reshape(self.sizes[-2], self.sizes[-1]).T
        derivative = mask @ derivative

        for i in range(len(self.sizes) - 3, -1, -1):
            activation = self.activations[i]
            w_index -= ((self.sizes[i] + 1) * self.sizes[i + 1])
            w = weights[w_index : w_index + self.sizes[i] * self.sizes[i + 1]].reshape(self.sizes[i], self.sizes[i + 1]).T
            derivative = derivative * activation.derivative() @ w
            
        return derivative