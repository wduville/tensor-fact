'''
Created on May 6, 2010

@author: Willy
'''

from tensor import Tensor
from __init__ import _als_check_params, _setup_initial_U, mttkrp

class KTensor:
    """
    Tensor stored as a Kruskal operator (decomposed)
    """

    def __init__(self, **kwargs):
        """ Create a KTensor from
            -lambda and U
            -Tensor
        """
        def typechecking():
            #if lambda_.ndim != 2:
            #    print "Error: LAMBDA must be a column vector."; exit()
            for i in range(len(kwargs["U"])):
                """Missing size checking"""
                if kwargs["U"][i].ndim != 2:
                    raise ValueError("Matrix U'' is not a matrix!")

        if 'lmda' in kwargs and 'U' in kwargs:
            typechecking()
    
            self.lambda_ = kwargs["lmda"]
            self.u = kwargs["U"]
            self.shape = tuple([a.shape[0] for a in self.u])

        elif isinstance(kwargs['X'], Tensor):
            self = self.cp_als(**kwargs)
        else:
            raise ValueError("Matrix U'' is not a matrix!")
    
    def ndims(self):
        """ returns the number of dimensions of tensor T """
        return len(self.u)

    def innerprod(self, Y):
        """ Inner product of a tensor and a K-tensor
        
        Parameters
        ----------
        A : K-Tensor
        Y : Tensor
    
        Returns
        -------
        scalar : double
        """
        
        if Y.shape != self.shape:
            raise ValueError("Error: X and Y must be the same size")

        if isinstance(Y, KTensor):
            raise NotImplementedError("innerprod of 2 K-Tensor is not implemented")
        
        elif isinstance(Y, Tensor):# or isinstance(Y, sptensor) or isinstance(Y, ttensor):
            #res = 0
            #for r in range(self.lambda_.size):
            #    vect = [self.u[n][:,r] for n in range(self.ndims())]
            #    res += self.lambda_[r] * Y.ttv(vect)
            res = [self.lambda_[r] * Y.ttv([n[:,r] for n in self.u])
                   for r in range(self.lambda_.size)]
        return sum(res)

    def norm(self):
        """ The Frobenius norm for a K-Tensor
        """
        from numpy import sqrt, outer, dot

        # Compute the matrix of correlation coefficients
        coefMatrix = outer(self.lambda_, self.lambda_.T)
        for i in range(self.ndims()):
            coefMatrix *= dot(self.u[i].T, self.u[i])

        return sqrt(coefMatrix.sum())

    def arrange(self):
        """ Normalizes the columns of each matrix, absorbing the
           excess weight into lambda and then sorts everything so that the
           lambda values are in decreasing order. """

        """ Ensure that the matrices are normalized """
        from scipy.linalg import norm
        from numpy import sort

        for r in range(len(self.lambda_)):
            for n in range(self.ndims()):
                mynorm = norm(self.u[n][:,r])
                self.u[n][:,r] /= mynorm
                self.lambda_[r] *= mynorm

        """ sort """
        idx = [sort(self.lambda_)[::-1].tolist().index(self.lambda_[i]) for i in range(self.lambda_.size)]
        self.lambda_ = sort(self.lambda_)[::-1]
        for i in range(self.ndims()):
            self.u[i] = self.u[i][:,idx]

    def fixsigns(self):
        """ Makes it so that the largest magnitude entries for
           each vector in each factor of K are positive, provided that the
           sign on *pairs* of vectors in a rank-1 component can be flipped.
        """
        from numpy import sign, max, where, floor, zeros

        for r in range(len(self.lambda_)):
            val = zeros(self.ndims())
            idx = zeros(self.ndims())
            sgn = zeros(self.ndims())
            for n in range(self.ndims()):
                val[n] = max(abs(self.u[n][:,r]))
                idx[n]
                sgn[n] = sign(self.u[n][idx[n],r])
                pass

            negidx = where(sgn == -1)
            nflip = 2 * floor(len(negidx)/2);

            for i in range(1, int(nflip)):
                n = negidx(i);
                self.u[n][:,r] =  -self.u[n][:,r];



    def cp_als(self, X, R, fitchangetol=1.0e-5, maxiters=200, verbose=1, init='eigs'):
        """CP_ALS
    
        Compute a CP decomposition of any type of tensor.
        """
        
        from numpy import ones, dot, sqrt
        from scipy.linalg import pinv, norm
        from scipy.sparse import spdiags, issparse
    
        dimorder = range(X.ndims())
    
        _als_check_params(X, R * ones(X.ndims()), maxiters, dimorder)
        U = _setup_initial_U(X, R * ones(X.ndims()), init, dimorder, verbose)
    
        fit = 0
        normX = norm(X.data)
        normresidual = lambda normX, X, K:\
            sqrt(normX**2 + K.norm()**2 - 2*K.innerprod(X))
    
        if verbose: print 'CP_ALS:\n'
    
        # Main Loop: Iterate until convergence
        for iter in range(maxiters):
            # Iterate over all N modes of the tensor
            for n in dimorder:
                # Compute the matrix of coefficients for linear system
                Y = ones((R, R))
                for i in range(n)+range(n+1, X.ndims()):
                    Y *= dot(U[i].T, U[i])
    
                # Calculate Unew = X_(n) * khatrirao(all U except n, 'r')
                Unew = mttkrp(X, U, n)
                Unew = dot(Unew, pinv(Y))
    
                # Normalize each vector to prevent singularities in coefmatrix
                if iter == 0: mylambda = sqrt((Unew ** 2).sum(axis=0)) # b2-norm
                else: mylambda = Unew.max(axis=0); mylambda[mylambda < 1] = 1 # max-norm
    
                Unew *= spdiags(1/mylambda, 0, R, R)
                U[n] = Unew
                if issparse(Unew):
                    raise ValueError("Don't know yet how to handle this") #U{n} = full(Unew);   % for the case R=1

            K = KTensor(lmda = mylambda, U = U)
            self = K
            self.lambda_ = mylambda
            self.u = U    
            self.shape = tuple([a.shape[0] for a in self.u])
            #Compute fit
            newfit = 1 - normresidual(normX, X, self) / normX
            fitdelta = abs(fit-newfit); fit = newfit
            if verbose: print(" Iter %2d: fit = %e fitdelta = %7.1e"%(iter+1, newfit, fitdelta))
            if iter and fitdelta < fitchangetol: break
        else:
            print "Limit of %i reached without convergence" % maxiters
            
    
        self.arrange(); self.fixsigns()
        self.fit = normresidual(normX, X, self)
    
        if verbose: print " Final fit = %e" % self.fit
        #return self

