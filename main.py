import os, sys
print(sys.version)
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from data_io import Data 
from preprocess import Preprocess
from optHist import OptHist

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./')
parser.add_argument('--num_points', type=int,default=5000)
parser.add_argument('--range', type=int,default=100)
parser.add_argument('--num_buckets', type=int,default=3)
parser.add_argument('--data_name', type=str, default=None)
parser.add_argument('--toy_example',type=bool,default=False)


config = parser.parse_args()
# run the optimal histogram for inputs or toy example on presentation
if config.toy_example ==False:
    data = Data(config.data_path)
    if config.data_name == None:
        x = data.generate_random_data(config.num_points,config.range,'mixture',4)    
    else:
        x = data.load_data(config.data_name)   
    print('num points :', x.shape[0])
    
    proc = Preprocess(x)
    oh = OptHist(proc)

    tic = time.time()
    sse, cut, hist = oh.findBucket(config.num_buckets)
    print('Running time :', time.time()-tic)
    print('Sum Squared Error :',sse)
    print('Optimal Cuts :',cut)
    x_hist, y_hist = oh.visualize_histogram(cut,hist)
    plt.hist(x,400)
    plt.plot(x_hist,y_hist)
    plt.show()
else :
    x = np.array([2,2,2,2,3,3,3,3,3,4,4,5,6,6,6,6,6,6,6,6,6,7,8,8,8,8,9,9,9])
    proc = Preprocess(x)
    print('value\n',proc.value)
    print('freq\n',proc.freq)
    print('P\n',proc.P)
    print('PP\n',proc.PP)
    #print(proc.SSE(1,2))
    oh = OptHist(proc)
    sse, cut, hist = oh.findBucket(3)
    print('cut list\n', cut)
    print('hist')
    print(hist)
    x1,y1 = oh.visualize_histogram(cut,hist)
    plt.hist(x)
    plt.plot(x1,y1)
    plt.show()


