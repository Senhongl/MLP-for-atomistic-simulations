import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from SPNN import *
from BPNN import *
import os

class representation:
    def __init__(self, db, params, n_radial, n_angular, fileName, max_iteration = 600, model = 'SPNN'):
        self.fileName = fileName
        if not os.path.exists(fileName):
            os.makedirs(fileName)
        logfile = open(f'./{fileName}/{fileName}.txt', 'w+')
        t0 = time.time()
        
        self.model = model
        # initialize the model
        if model == 'SPNN':
            self.nn = SPNN(db, params, n_radial, n_angular)
        elif model == 'BPNN':
            self.nn = BPNN(db, params, n_radial, n_angular)
         
        logfile.write(f'It takes {time.time()-t0}s to generate the fingerprints.\n')
        logfile.write('|Epoch|  Energy MSE   |  Forces MSE   |   total loss  |   Time (s)    |\n')
        logfile.close()
        
        self.Nfeval = 0
        
        self.max_iteration = max_iteration
        
        
    def callbackF(self, weights):
        logfile = open(f'./{self.fileName}/{self.fileName}.txt', 'a')
        E_MSE, F_MSE = self.nn.error(weights)
        loss = self.nn.objective(weights)
        logfile.write('|{:5}|{:15.5f}|{:15.5f}|{:15.5f}|{:15.1f}| \n'.format(
            self.Nfeval, E_MSE, F_MSE, loss, time.time() - self.t0))
        logfile.close()
        self.Nfeval += 1
        
    def __call__(self):
        
        if self.model == 'BPNN':
            init_weights = []
            for nn in self.nn.NN:
                init_weights.extend(nn.weights)
            init_weights = np.array(init_weights)
        else:
            init_weights = self.nn.NN.weights
            
        self.t0 = time.time()
        self.res = minimize(self.nn.objective, init_weights, callback = self.callbackF, options = {'maxiter': self.max_iteration})

    def visualization(self):
        
        if self.model == 'BPNN':
            weights = []
            for nn in self.nn.NN:
                weights.extend(nn.weights)
            weights = np.array(weights)
        else:
            weights = self.nn.NN.weights
            
        E_predict, F_predict = self.nn.predict_value(weights)

        E_label = self.nn.E
        E_NN = E_predict
        plt.scatter(E_label,E_NN)
        plt.plot(E_NN,E_NN)
        plt.xlabel('EMT predicted energy (eV)')
        plt.ylabel('NN predicted energy (eV)')
        plt.savefig(f'./{self.fileName}/Energy.png')
        plt.show()
        
        F_NN = [f for Force in F_predict for F in Force for f in F]
        F_label = [f for Force in self.nn.F for F in Force for f in F]

        plt.clf()
        plt.scatter(F_label,F_NN)
        plt.plot(F_NN,F_NN)
        plt.xlabel('EMT predicted forces (eV/$\AA$)')
        plt.ylabel('NN predicted forces (eV/$\AA$)')
        plt.savefig(f'./{self.fileName}/Forces.png')
        plt.show()
