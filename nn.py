import autograd.numpy as np

"""
Construct the basic neuralnetwork for BPNN and SPNN, anyone of them would conduct this file as a basic block.
"""


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
    Tanh non-linearity
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state 

    def derivative(self):
        return 1 - self.state**2
    

class neuralnetwork:
    def __init__(self, sizes, act = 'tanh', seeds = None):
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

            >>> layer_size = array([3, 2, 3, 1])

            whihc means that there are 2 hidden layers and the hidden units will be 2 and
            3, respectively. The inputs size should be different when the number of 
            fingerprints and the type of fingeprints are different. In the BPNN, the outputs 
            should always be 1, which means the atomic energy of a element. In the SPNN, 
            the outputs would be different when the number of elements in the configuration
            is different.
            
        Returns:
        ------------
        A weights matrix, which contains biases.
        """
        rs = np.random.RandomState(seed = seed)
        weights = []
        for i in range(len(sizes) - 1):
            weights += list(rs.randn((sizes[i] + 1) * sizes[i + 1]))
        return np.stack(weights)
    

    
    def feedforward(self, weights, inputs, mask):
        """
        Do the feedforward for the network.
        Parameters:
        -----------
        weights: np.ndarray
            The weights matrix should be an attribute of the neuralnetwork. However, in order 
            to do the autograd, it has to be an argument of the function.
        inputs: np.ndarray
            The inputs of the neuralnetwork would be the value of fingerprints. The shape of 
            it depends of the number of fingerprints and the type of fingerprints. 
        mask: np.ndarray
            The mask is used for the final outputs. For single element system, the mask will 
            just be one.
        
        Returns:
        ------------
        atomic energy
        """
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
        """
        Do the backpropgation for the network.
        Parameters:
        -----------
        weights: np.ndarray
            The weights matrix should be an attribute of the neuralnetwork. 
        mask: np.ndarray
            The mask is used for the final outputs. For single element system, the mask will 
            just be one.
        
        Returns:
        ------------
        the derivative of atomic energy w.r.t. fingerprints
        """
        
        w_index = len(weights) - (self.sizes[-2] + 1) * self.sizes[-1]
        derivative = weights[w_index : -self.sizes[-1]].reshape(self.sizes[-2], self.sizes[-1]).T
        derivative = mask @ derivative

        for i in range(len(self.sizes) - 3, -1, -1):
            activation = self.activations[i]
            w_index -= ((self.sizes[i] + 1) * self.sizes[i + 1])
            w = weights[w_index : w_index + self.sizes[i] * self.sizes[i + 1]].reshape(self.sizes[i], self.sizes[i + 1]).T
            derivative = derivative * activation.derivative() @ w
            
        return derivative
