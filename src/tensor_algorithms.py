'''
Created on Jan 21, 2010
@author: Willy
'''

import numpy
from scipy import io

if __name__ == '__main__':
    def normalize_als(X): 
        X -= X.min()
        X[numpy.isnan(X)] = 0
        X += numpy.finfo(float).eps
        X /= X.max()

    def loadmat(file):
        file_variable = {'claus.mat': 'X', 'CBCL_tensor.mat': 'Vtensor',
                          'fluordata.mat': 'fluor'}[file]

        X = io.loadmat('../tests/bench/' + file, struct_as_record=False, squeeze_me=True)

        if file == 'fluordata.mat': return X[file_variable].data        
        return X[file_variable]
    
    from pylab import figure, show
    
    from tensor import Tensor, tucker_als, KTensor

    als, file, R, finalfit = ((KTensor,    'claus.mat', 5, 18.05799552),
                              (tucker_als, 'claus.mat', 4, 1.0041440994))[0]

    X = loadmat(file)
    normalize_als(X)
    X = Tensor(X)

    if 0:
        import timeit
        number = 2
        time = timeit.Timer('Xh, fit = als(X, R, verbose = 0)',
                            "from __main__ import als, X, R").timeit(number) / number
    else:
        time = None

    Xh, fit = tucker_als(X, R=4)
    #Xh, fit = als(X, R, init = 'eigs')
    Xh = KTensor(X = X, R = R, init = 'eigs')
    
    if abs(Xh.fit - finalfit) < 1e-8:
        print "------\n+PASS 0.68[{0} sec]+\n------".format(time)
    else:
        print \
"---------------\n\
FAIL: Final fit missmatch. Returned {0}, Expected {1}\n\
---------------".format(Xh.fit, finalfit)

    if 1:
        # Display results
        fig = figure()
        [fig.add_subplot(X.data.ndim, 1, k+1).plot(Xh.u[k]) for k in range(X.data.ndim)]
        show()
