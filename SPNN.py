import autograd.numpy as np
from autograd import jacobian
from nn import *
from Gaussian import *

"""
This file conduct the shared-parameters neural network (SPNN), which means that no matter how many elements in the 
configuration, all of them would share the same parameters to compute the atomic energy. The neural network is 
currently only support the G2 and G4 function as fingerprints.
"""


class SPNN:
    def __init__(self, db, params, n_radial = 2, n_angular = 1):
        self.db = db
        self.E, self.F, self.atoms_set, self.numbers, self.num_atoms = self.data()
        self.elements = np.array(params['elements'])

        self.sizes = params['sizes']
        
        # insert the input layer, since for radial symmetry function, the number of pair terms would
        # be equal to the number of elements in the configuration. For example, for Au-Pd system, the pair
        # terms would be Au-Au and Au-Pd when the center atom is Au. 
        # For angular symmetry function, the number of triplet terms would be a permulation problem.
        self.sizes.insert(0, n_radial * len(self.elements) + n_angular * np.sum(np.arange(len(self.elements) + 1)))
        
        
        # insert the output layer, for SPNN, the number of output units would be equal to the number of elements
        self.sizes.append(len(self.elements))
        
        
        # initialize a neuralnetwork
        self.NN = neuralnetwork(self.sizes)
        
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
            E.append(d.energy)
            F.append(d.forces)
            atoms.append(d.toatoms())
            numbers.append(d.toatoms().numbers)
            
        num_atoms = len(atoms[0].positions)
        return np.array(E), np.array(F), atoms_set, np.array(numbers), num_atoms

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

    def dG4dR(self, positions, cell, numbers, elements, fp_params):
        dGdR = jacobian(G4, argnum = 0)
        return dGdR(positions, cell, numbers, elements, fp_params)


    def predict_value(self, weights):
        """
        Computing the energy and forces of the configurations.
        Parameters:
        -----------
        weights: np.ndarray
            The weights matrix should be an attribute of the neuralnetwork. In the following part,
            the netwrok would be trained by Scipy.optimize.minimize to optimize the weights.
            
        Returns:
        ------------
        Predict value of energy and forces.
        """
        
        E_predict = []
        F_predict = np.zeros(np.array(self.F).shape)

        # create a mask for different elements
        mask = []
        for element in self.elements:
            mask.append((list(map(lambda x: x == element, self.numbers)) * np.ones(self.numbers.shape)).T)
        mask = np.array(mask).T
        
        for i in range(len(self.atoms_set)):
            fps = self.fps[i]
            outputs = self.NN.feedforward(weights, fps, mask[i])
            dEdfp = self.NN.dEdfp(weights, mask[i])
            E_predict.append(np.sum(outputs))

            dfpidX = self.dfpdX[i]
            d0, d1, d2, d3 = dfpidX.shape
            
            # compute forces
            F_predict[i] -= np.dot(dEdfp.reshape(1, -1), dfpidX.reshape((d0 * d1, d2 * d3))).reshape((d2, d3))
        
        E_predict = np.array(E_predict)


        return E_predict, F_predict

    
    def error(self, weights):
        E_predict, F_predict = self.predict_value(weights)
        
        E_MSE = np.mean(np.square(self.E - E_predict))
        F_MSE = np.mean(np.square(self.F - F_predict))

        return E_MSE, F_MSE
        
    def objective(self, weights):

        E_MSE, F_MSE = self.error(weights)

        return self.E_coeff * E_MSE + self.F_coeff * F_MSE
