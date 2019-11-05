"""
This file conduct the Behler–Parinello Neural Network (BPNN). The network is conducted according to the paper 
"Constructing High-Dimensional Neural Network Potentials: A Tutorial Review" 
(https://onlinelibrary.wiley.com/doi/full/10.1002/qua.24890).
"""


import autograd.numpy as np
from autograd import jacobian 
from nn import *
from Gaussian import *

class BPNN:
    def __init__(self, db, params, n_radial = 2, n_angular = 1):
        self.db = db
        self.E, self.F, self.atoms_set, self.numbers, self.num_atoms = self.data()
        self.elements = np.array(params['elements'])
        
        # generate NNs for every element
        self.NN = []
        self.sizes = params['sizes']
        
        # insert the input layer, since for radial symmetry function, the number of pair terms would
        # be equal to the number of elements in the configuration. For example, for Au-Pd system, the pair
        # terms would be Au-Au and Au-Pd when the center atom is Au. 
        # For angular symmetry function, the number of triplet terms would be a permulation problem.
        self.sizes.insert(0, n_radial * len(self.elements) + n_angular * np.sum(np.arange(len(self.elements) + 1)))
        # insert the output layer, it will always be one for single neural network.
        self.sizes.append(1)
        
        # initialize neuralnetwork for different elements
        for i in range(len(self.elements)):
            self.NN.append(neuralnetwork(self.sizes))
        
        # generate fingerprints and the derivative of fingerprints w.r.t. atomic position
        self.fp_params = params['fp_params']
        self.n_radial = n_radial 
        self.n_angular = n_angular
        self.fps, self.dfpdX = self.generate_fps()

        self.E_coeff = params['E_coeff']
        self.F_coeff = params['F_coeff']

    def data(self):
        """
        Extract data from database
        -----------
        Returns:
        ------------
        E, F: np.ndarray
            The value of energy and forces for the configuration. 
        atoms_set: list
            A list of configurations in the database.
        numbers: np.ndarray
            A matrix that contains the atomic number in all of the configurations.
        num_atoms: int
            A scalar value that is the number of atoms in the configurations.
        """
        
        E_labels = []
        F_labels = []
        atoms_set = []
        numbers = []
        
        for d in self.db.select():
            E_labels.append(d.energy)
            F_labels.append(d.forces)
            atoms_set.append(d.toatoms())
            numbers.append(d.toatoms().numbers)
            
        num_atoms = len(atoms_set[0].positions)
        return np.array(E_labels), np.array(F_labels), atoms_set, np.array(numbers), num_atoms

    def generate_fps(self):
        """
        Generate not only the fingerprints of all the configuration, but also the derivative of 
        fingerprints w.r.t. atoms position. The process of derivation is conducted by the autograd.
        """
        
        fps = []
        dfpdX = []
        for atoms in self.atoms_set:
            atoms_fps = []
            single_dfpdX = None
            for idx in range(self.n_radial):
                g2 = G2(atoms.positions, atoms.cell, atoms.numbers, self.elements, self.fp_params['radial'][idx])
                atoms_fps.extend(g2.T)

                try:
                    single_dfpdX = np.concatenate((single_dfpdX, self.dG2dR(atoms.positions, atoms.cell, atoms.numbers, self.elements, self.fp_params['radial'][idx])), axis = 1)
                except:
                    single_dfpdX = self.dG2dR(atoms.positions, atoms.cell, atoms.numbers, self.elements, self.fp_params['radial'][idx])

            for idx in range(self.n_angular):
                g4 = G4(atoms.positions, atoms.cell, atoms.numbers, self.elements, self.fp_params['angular'][idx])
                atoms_fps.extend(g4.T)
                
                try:
                    single_dfpdX = np.concatenate((single_dfpdX, self.dG4dR(atoms.positions, atoms.cell, atoms.numbers, self.elements, self.fp_params['angular'][idx])), axis = 1)
                except:
                    single_dfpdX = self.dG4dR(atoms.positions, atoms.cell, atoms.numbers, self.elements, self.fp_params['angular'][idx])


            fps.append(np.array(atoms_fps).T)
            dfpdX.append(np.array(single_dfpdX))

        return np.array(fps), np.array(dfpdX)


    def dG2dR(self, positions, cell, numbers, elements, fp_params):
        dGdR = jacobian(G2, argnum = 0)
        return dGdR(positions, cell, numbers, elements, fp_params)

    def dG4dR(self, positions, cell, fp_params):
        dGdR = jacobian(G4, argnum = 0)
        return dGdR(positions, cell, fp_params)

    
    
    def predict_value(self, weights):
        """
        Computing the energy and forces of the configurations.
        Parameters:
        -----------
        weights: np.ndarray
            It would a huge array contains all of the weights in the neuralnetwork.
            The weights matrix should be an attribute of the neuralnetwork. In the following part,
            the netwrok would be trained by Scipy.optimize.minimize to optimize the weights.
            
        Returns:
        ------------
        Predict value of energy and forces.
        """
        
        E_predict = []
        dEdfp = []
        F_predict = np.zeros(np.array(self.F).shape)
        
        # # create a mask for different elements
        # mask = []
        # for element in self.elements:
        #     mask.append((list(map(lambda x: x == element, self.numbers)) * np.ones(self.numbers.shape)).T)
        # mask = np.array(mask).T
        
        # loop over every single element to get the total energy and force
        for idx, element in enumerate(self.elements):
            mask = list(map(lambda x: x == element, self.numbers)) * np.ones(self.numbers.shape)
            nn = self.NN[idx]
            weight = weights[idx * len(nn.weights): (idx + 1) * len(nn.weights)] 
            nn.weights = weight
            for j in range(len(self.atoms_set)):
                fps = self.fps[j]
                E_predict.append(np.sum(nn.feedforward(weight, fps, mask[j][:, None])))
                dEdfp.append(nn.dEdfp(weight, mask[j][:, None])) 
        
        E_predict = np.array(E_predict)
        d0 = len(E_predict)
        E_predict = E_predict.reshape(len(self.elements), d0 // len(self.elements))
        E_predict = np.sum(E_predict, axis = 0)

        dEdfp = np.array(dEdfp)
        d0, d1, d2 = dEdfp.shape
        dEdfp = dEdfp.reshape(len(self.elements), d0 // len(self.elements), d1, d2)
        dEdfp = np.sum(dEdfp, axis = 0)
        
        for i in range(len(F_predict)):
            dfpidX = self.dfpdX[i]
            shape = dfpidX.shape
            d0, d1, d2, d3 = dfpidX.shape
            F_predict[i] = -np.dot(dEdfp[i].reshape(1, -1), dfpidX.reshape((d0 * d1, d2 * d3))).reshape((d2, d3))


        return E_predict, F_predict



    def error(self, weights):
        E_predict, F_predict = self.predict_value(weights)
        
        E_MSE = np.mean(np.square(self.E - E_predict))
        F_MSE = np.mean(np.square(self.F - F_predict))

        return E_MSE, F_MSE
        
    def objective(self, weights):

        E_MSE, F_MSE = self.error(weights)

        return self.E_coeff * E_MSE + self.F_coeff * F_MSE
    
