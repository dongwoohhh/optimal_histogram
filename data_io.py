import os,sys
import numpy as np
import matplotlib.pyplot as plt

class Data():
    def __init__(self, dataDir):
        self.dataDir = dataDir
    def load_data(self,dataName):
        x = np.load(os.path.join(self.dataDir, dataName+'.npy'))
        return x
    def save_data(self,dataName, x):
        np.save(os.path.join(self.dataDir,dataName+'.npy'),x)
    def generate_random_data(self, nPoints, xRange,dataType, nMixture=2):
        if dataType == 'gaussian':
            x = self.gaussian_sample(nPoints,xRange,xRange)
            x = x - np.amin(x)
            #print ('min & max :',np.amin(x),np.amax(x))
        if dataType == 'mixture':
            x = np.empty((0))
            for it in range(1,nMixture+1):
                x_tmp = self.gaussian_sample(nPoints//nMixture,xRange//(it*it), xRange)
                x  = np.append(x, x_tmp)
            # Our value start from 1 (as paper)
            x = x - np.amin(x) + 1
        return x
    def gaussian_sample(self,nPoints,center, dev):
        center = center#np.random.randint(xRange // 4)
        deviation = dev // 4
        #print(center,deviation, nPoints)
        x = np.random.normal(center,deviation,nPoints).astype(np.int32)
        return x
if __name__ == '__main__':
    data =Data('./')
    d = data.generate_random_data(5000, 64, 'mixture',4)
    #data.save_data('foo',d)
    #d= data.load_data('here')
    plt.hist(d,100)
    fig = plt.show()