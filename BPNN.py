import autograd.numpy as np
from autograd import jacobian
from autograd import elementwise_grad
from nn import *
from Gaussian import *

class BPNN:
    def __init__(self, db, params, n_radial = 2, n_angular = 1, mode = 'Behler'):
        self.db = db
        self.E, self.F, self.atoms_set, self.numbers, self.num_atoms = self.data()
        self.elements = np.array(params['elements'])
        
        # generate NNs for every element
        self.NN = []
        self.sizes = params['sizes']
        # insert the input layer
        self.sizes.insert(0, n_radial * len(self.elements) + n_angular)
        # insert the output layer
        self.sizes.append(1)
        for i in range(len(self.elements)):
            self.NN.append(neuralnetwork(self.sizes))
        
        # generate fingerprints
        self.fp_params = params['fp_params']
        self.mode = mode
        self.n_radial = n_radial 
        self.n_angular = n_angular
        self.fps, self.dfpdX = self.generate_fps()

        self.E_coeff = params['E_coeff']
        self.F_coeff = params['F_coeff']

    def data(self):
        E = []
        F = []
        atoms = []
        numbers = []
        for d in self.db.select():
            E.append(d.energy)
            F.append(d.forces)
            atoms.append(d.toatoms())
            numbers.append(d.toatoms().numbers)
        num_atoms = len(atoms[0].positions)
        return np.array(E), np.array(F), atoms, np.array(numbers), num_atoms


    def generate_fps(self):
        fps = []
        dfpdX = []
        for atoms in self.atoms_set:
            single_fps = []
            single_dfpdX = None
            for idx in range(self.n_radial):
                g2 = G2(atoms.positions, atoms.cell, atoms.numbers, self.elements, self.fp_params['radial'][idx])
                single_fps.extend(g2.T)

                try:
                    single_dfpdX = np.concatenate((single_dfpdX, self.dG2dR(atoms.positions, atoms.cell, atoms.numbers, self.elements, self.fp_params['radial'][idx])), axis = 1)
                except:
                    single_dfpdX = self.dG2dR(atoms.positions, atoms.cell, atoms.numbers, self.elements, self.fp_params['radial'][idx])

            for idx in range(self.n_angular):
                g4 = G4(atoms.positions, atoms.cell, self.fp_params['angular'][idx])
                single_fps.append(g4)
                single_dfpdX.append(self.dG4dR(atoms.positions, atoms.cell, self.fp_params['angular'][idx]))
            fps.append(np.array(single_fps).T)
            dfpdX.append(np.array(single_dfpdX))

        return np.array(fps), np.array(dfpdX)

    # def dG2dR(self, positions, cell, fp_params):
    #     dGdR = elementwise_grad(G2, argnum = 0)
    #     return dGdR(positions, cell, fp_params)

    # def dG4dR(self, positions, cell, fp_params):
    #     dGdR = elementwise_grad(G4, argnum = 0)
    #     return dGdR(positions, cell, fp_params)

    def dG2dR(self, positions, cell, numbers, elements, fp_params):
        dGdR = jacobian(G2, argnum = 0)
        return dGdR(positions, cell, numbers, elements, fp_params)

    def dG4dR(self, positions, cell, fp_params):
        dGdR = jacobian(G4, argnum = 0)
        return dGdR(positions, cell, fp_params)

    def predict_value(self, weights):
        E_predict = []
        dEdfp = []
        F_predict = np.zeros(np.array(self.F).shape)
        for idx, element in enumerate(self.elements):
            mask = list(map(lambda x: x == element, self.numbers)) * np.ones(self.numbers.shape)
            nn = self.NN[idx]
            weight = weights[idx * len(nn.weights): (idx + 1) * len(nn.weights)] 
            nn.weights = weight
            for j in range(len(self.atoms_set)):
                fps = self.fps[j]
                
                E_predict.append(np.sum(nn.feedforward(weight, fps, mask[j][:, None])))
                # print(elementwise_grad(nn.feedforward, argnum = 1)(weight, fps, mask[j][:, None]))
                # print(nn.dEdfp(weight, mask[j][:, None]))
                # dEdfp.append(elementwise_grad(nn.feedforward, argnum = 1)(weight, fps, mask[j][:, None]))
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
            # F_predict[i] = -np.dot(dEdfp[i].reshape(1, -1), dfpidX.T.reshape((shape[-1] * shape[-2], -1))).reshape((d2, d3))
            d0, d1, d2, d3 = dfpidX.shape
            F_predict[i] = -np.dot(dEdfp[i].reshape(1, -1), dfpidX.reshape((d0 * d1, d2 * d3))).reshape((d2, d3))


        return E_predict, F_predict

    # def predict_value(self, weights):
    #     E_predict = []
    #     dEdfp = []
    #     F_predict = np.zeros(np.array(self.F).shape)
    #     for idx, element in enumerate(self.elements):
    #         mask = list(map(lambda x: x == element, self.numbers)) * np.ones(self.numbers.shape)
    #         nn = self.NN[idx]
    #         weight = weights[idx * len(nn.weights): (idx + 1) * len(nn.weights)] 
    #         nn.weights = weight
    #         for j in range(len(self.atoms_set)):
    #             fps = self.fps[j] * mask[j][:, None]
                
    #             E_predict.append(np.sum(nn.feedforward(weight, fps, mask[j][:, None])))
    #             # dEdfp.append(nn.dEdfp(weight, mask[j][:, None])) 
        
    #     E_predict = np.array(E_predict)
    #     d0 = len(E_predict)
    #     E_predict = E_predict.reshape(len(self.elements), d0 // len(self.elements))
    #     E_predict = np.sum(E_predict, axis = 0)


    #     return E_predict

    def error(self, weights):
        E_predict, F_predict = self.predict_value(weights)
        
        E_MSE = np.mean(np.square(self.E - E_predict))
        F_MSE = np.mean(np.square(self.F - F_predict))

        return E_MSE, F_MSE
        
    def objective(self, weights):

        E_MSE, F_MSE = self.error(weights)

        return self.E_coeff * E_MSE + self.F_coeff * F_MSE
    