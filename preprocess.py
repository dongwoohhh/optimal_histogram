import numpy as np



class Preprocess():
    def __init__(self,x):
        self.x = x
        self._compute_table()
        self._compute_PnPP()

    def _compute_table(self):
        # O(nlogn), sorting
        value, freq = np.unique(self.x, return_counts=True)
        value = value.astype(np.int64)
        #print(value)
        #print(freq)
        self.maxValue = np.amax(value)
        self.value = np.arange(self.maxValue + 1)
        self.freq = np.zeros(self.maxValue + 1,dtype=np.int64)
        self.freq[value] = freq
        
        #print('Value :\n',self.value)
        #print('Freq. :\n',self.freq)

    def _compute_PnPP(self):
        # O(n), cumulativee sum, cumulative square sum
        self.P = np.cumsum(self.freq).astype(np.float64)
        self.PP = np.cumsum(self.freq * self.freq).astype(np.float64)

        #print('P :\n',self.P)
        #print('PP :\n',self.PP)
    def _Fsquare(self,i,j):
        # O(1)
        fSquare = self.PP[j] - self.PP[i-1]
        #print(fSquare)
        return fSquare

    def _Avg(self,i,j):
        # O(1)
        avg = (self.P[j] - self.P[i-1]) / (j - i + 1)
        #print(avg)
        return avg 
    def SSE(self,i,j):
        # O(1)
        sumSquaredError = self._Fsquare(i,j) - (j - i + 1) * np.square(self._Avg(i,j))
        #print(sumSquareError)
        return sumSquaredError


if __name__ == '__main__':
    from data_io import Data

    data = Data('./')
    x = data.generate_random_data(50,32,'mixture',4)
    #print (np.sort(x))
    proc = Preprocess(x)
    print(proc.SSE(16,24))
    print(proc.freq[16:25])
    avg = np.mean(proc.freq[16:25])
    var = np.sum((np.square(proc.freq[16:25]-avg)))
    print(avg)
    #print('square freq', np.sum(np.square(proc.freq[16:25])))
    print(var)

    print(proc.SSE(16,16))
