import numpy as np
from preprocess import Preprocess
class OptHist():
    def __init__(self,Preprocess):
        self.SSE = Preprocess.SSE
        self.avg = Preprocess._Avg
        self.maxValue = Preprocess.maxValue
        self.P = Preprocess.P
    def findBucket(self,numBucket): 
        dp_table = np.ones((self.maxValue+1,numBucket+1),dtype=np.float64) * float("inf")
        # cut list should be 2-d list
        cut_list_bef=np.zeros((self.maxValue+1,numBucket+1),dtype=np.int64) # 1-indexing
        cut_list_aft=np.zeros((self.maxValue+1,numBucket+1),dtype=np.int64)
        for b in range(1,numBucket+1):
            for i in range(1,self.maxValue+1):
                if b == 1:
                    #print(self.SSE(1,i))
                    dp_table[i,b] = self.SSE(1,i)
                    cut_list_aft[i,b]= 0
                    #print(cut_list_aft)
                else:
                    sse_list = [float("inf")] ## initially append inf for 1-indexing
                    if i < b :
                        sse_list.append(float("inf"))
                    else:
                        for j in range(1,i):
                            sse_list.append(dp_table[j,b-1] + self.SSE(j+1,i))
                        sse_list = np.asarray(sse_list)
                        cut_tmp = np.argmin(sse_list)
                        dp_table[i,b] = sse_list[cut_tmp]
                        cut_list_aft[i,1:b] = cut_list_bef[cut_tmp,1:b]
                        cut_list_aft[i,b] = cut_tmp
            cut_list_bef = cut_list_aft
            cut_list_aft = np.zeros((self.maxValue+1,numBucket+1),dtype=np.int64)
            
        sse_opt = dp_table[self.maxValue, numBucket]
        cut_list = np.append(cut_list_bef[self.maxValue,1:],self.maxValue)
        #print(cut_list)
        histogram = self.compute_histogram(cut_list,numBucket)

        return sse_opt, cut_list, histogram

    def compute_histogram(self,cut_list,numBucket):
        histogram = np.zeros((numBucket)) # column(0) : representative value (avg), column(1) : frequency in bucket
        for i in range(numBucket):
            idx1 = cut_list[i] + 1
            idx2 = cut_list[i + 1]
            #print ('({}, {})'.format(idx1, idx2))
            #print(self.avg(idx1, idx2))
            
            histogram[i] = self.avg(idx1, idx2) * (idx2 - idx1 + 1)
        return histogram

if __name__ == '__main__':
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