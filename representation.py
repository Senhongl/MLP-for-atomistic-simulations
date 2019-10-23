import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from newBPNN import *
import os

class representation:
    def __init__(self, db, params, n_radial, n_angular, fileName, max_iteration = 600):
        self.fileName = fileName
        if not os.path.exists(fileName):
            os.makedirs(fileName)
        logfile = open(f'./{fileName}/{fileName}.txt', 'w+')
        t0 = time.time()
        self.bpnn = BPNN(db, params, n_radial, n_angular)
        logfile.write(f'It takes {time.time()-t0}s to generate the fingerprints.\n')
        logfile.write('|Epoch|  Energy MSE   |  Forces MSE   |   total loss  |   Time (s)    |\n')
        logfile.close()
        
        self.Nfeval = 0
        
        self.max_iteration = max_iteration
        
        
    def callbackF(self, weights):
        logfile = open(f'./{self.fileName}/{self.fileName}.txt', 'a')
        E_MSE, F_MSE = self.bpnn.error(weights)
        loss = self.bpnn.objective(weights)
        logfile.write('|{:5}|{:15.5f}|{:15.5f}|{:15.5f}|{:15.1f}| \n'.format(
            self.Nfeval, E_MSE, F_MSE, loss, time.time() - self.t0))
        logfile.close()
        self.Nfeval += 1
        
    def __call__(self):
        # weights = []
        # for nn in self.bpnn.NN:
        #     weights.extend(nn.weights)
        # weights = np.array(weights)
        self.t0 = time.time()
        self.res = minimize(self.bpnn.objective, self.bpnn.NN.weights, callback = self.callbackF, options = {'maxiter': self.max_iteration})
        
    def visualization(self):
        # weights = []
        # for nn in self.bpnn.NN:
        #     weights.extend(nn.weights)
        # weights = np.array(weights)
        E_predict, F_predict = self.bpnn.predict_value(self.bpnn.NN.weights)

        E_label = self.bpnn.E
        E_NN = E_predict
        plt.scatter(E_label,E_NN)
        plt.plot(E_NN,E_NN)
        plt.xlabel('EMT predicted energy (eV)')
        plt.ylabel('NN predicted energy (eV)')
        plt.savefig(f'./{self.fileName}/Energy.png')
        plt.show()
        
        F_NN = [f for Force in F_predict for F in Force for f in F]
        F_label = [f for Force in self.bpnn.F for F in Force for f in F]

        plt.clf()
        plt.scatter(F_label,F_NN)
        plt.plot(F_NN,F_NN)
        plt.xlabel('EMT predicted forces (eV/$\AA$)')
        plt.ylabel('NN predicted forces (eV/$\AA$)')
        plt.savefig(f'./{self.fileName}/Forces.png')
        plt.show()