'''
Created on Jan 21, 2010

@author: Willy
'''


import numpy


class Tensor:
    
    data = None;
    shape = None;
    
    def __init__(self, data, shape = None):
        """Constructor for tensor object.
        dat can be numpy.array or list.
        shape can be numpy.array, list, tuple of integers"""
        if(data.__class__ == list):
            data = numpy.array(data);

        if(shape != None):
            if(len(shape) == 0):
                raise ValueError("Second argument must be a row vector.");
            
            if(shape.__class__ == numpy.ndarray):
                if(shape.ndim != 2 and shape[0].size != 1):
                    raise ValueError("Second argument must be a row vector.");
            shape = tuple(shape);
        else:
            shape = tuple(data.shape);
        

        if (len(shape) == 0):
            if (data.size != 0):
                raise ValueError("Empty tensor cannot contain any elements");
            
        ### TO RECODE (tools deremoval)
        ###elif (tools.prod(shape) != data.size):
        ###    raise ValueError("Size of data does not match specified size of tensor");
            
        self.shape = shape;
        self.data = data.reshape(self.shape);
    

    def reshape(self, size):
        """ !duvill_w! """
        self.data = self.data.reshape(size)
        self.shape = self.data.shape 
    
    def copy(self):
        """ returns the deepcopy of tensor object."""
        return Tensor(self.data.copy(), self.shape);
    @property
    def ndims(self):
        """ The Tensor's number of dimensions """
        return len(self.shape);

    
    def permute(self, order):
        """ returns a tensor permuted by the order specified. """
        
        def find(nda, obj):
            """returns the index of the obj in the given nda(ndarray, list, or tuple)"""
            for i in range(0, len(nda)):
                if(nda[i] == obj):
                    return i;
            return -1;
        
        if (order.__class__ == list):
            order = numpy.array(order);
            
        if(self.ndims != len(order)):
            raise ValueError("Invalid permutation order: ndims %i, order %s"
                             % (self.ndims, order))
           
        sortedorder = order.copy();
        sortedorder.sort();
        ##sortedorder = order.sorted();
        
        if not ((sortedorder == numpy.arange(self.data.ndim)).all()):
            ##raise ValueError("Invalid permutation order");
            raise ValueError("Invalid permutation order: %s != %s" % (sortedorder, numpy.arange(self.data.ndim)))
        
        neworder = numpy.arange(len(order)).tolist();
        newshape = list(self.shape);
        newdata = self.data.copy();

        for i in range(0,len(order)-1):
            index = find(neworder, order[i]);
            newdata = newdata.swapaxes(i,index);
            
            temp = newshape[i];
            newshape[i] = newshape[index];
            newshape[index] = temp;
            temp = neworder[i];
            neworder[i] = neworder[index];
            neworder[index] = temp;
        
        newshape = tuple(newshape);
        return Tensor(newdata,newshape);
    
    def ipermute(self, order):
        """ returns a tensor permuted by the inverse of the order specified. """
        
        def find(nda, obj):
            """returns the index of the obj in the given nda(ndarray, list, or tuple)"""
            for i in range(0, len(nda)):
                if(nda[i] == obj):
                    return i;
            return -1;
        #calculate the inverse of iorder
        iorder = [];
        for i in range(0, len(order)):
            iorder.extend([find(order, i)]);
        
        #returns the permuted tensor by the inverse
        return self.permute(iorder);
        
    def norm(self):
        """ Frobenius norm of a Tensor
        """
        from scipy.linalg import norm

        return norm(self.data);

    def innerprod(self, Y):
        raise NotImplementedError("this innerprod is not implemented")
 
    def ttv(self, vect, dims = []):
        """ mode-n tensor-vector product
        
        Parameters
        ----------
        self : Tensor, ndims: N
        vect : sequence of vectors
        dims : specifies the dimension (or mode) of X along which vect should be multiplied
        
        Returns
        -------
        Y : Tensor, ndims: N-1
        """
        if isinstance(vect, list):
            if [vect[p].size for p in range(len(vect))] != list(self.shape):
                raise ValueError ("len(vect) = %i differ from shape(X, mode_n) = %i"
                                  % ( [vect[p].size for p in range( len(vect))], list(self.shape)))
            
            dims = range(len(vect))
            vidx = range(len(vect))

            """ Permute it so that the dimensions we're working with come last """
            remdims = numpy.setdiff1d(range(self.ndims), dims)
            if self.ndims > 1:
                c = self.permute(numpy.concatenate([remdims, numpy.array(dims)]))
                c = c.data
                
            n = self.ndims - 1
            for i in range(len(dims) - 1, -1, -1):
                if n == 0:
                    c = c.reshape( [1, self.shape[n]] )
                else:
                    c = c.reshape( [reduce(numpy.multiply, self.shape[0:n]), self.shape[n]] )
                c = numpy.dot(c, vect[vidx[i]])
                n -= 1
            
            return c
        else:
            raise NotImplementedError("Don't know how to ttv this vect type")

            #if mode_n < 1 or mode_n > self.ndims():
            #    raise ValueError ("Mode-N must be between 1 and %i (tensor order) " % self.ndims())

            
        """
        #Y2 = tensor(data = numpy.swapaxes(self.data, mode_n, -1))
        Y3 = numpy.zeros((Y2.shape))
        for j in range(self.dimsize(0)):
            for r in range(self.dimsize(1)):
                for p in range(self.dimsize(2)):
                    Y3[j,r,p] += self.data[j,r,p] * vect[p]
        return(Y3)
        """
        
    def ttm(self, mat, dims=None, transpose=False, excludedim=False):
        """ mode-n tensor-matrix product
        
        Parameters
        ----------
        self : Tensor
        mat : ndarray
            Single matrix or a list of matrices to be sequentially multiplied along all dimensions.
        dims : specifies the dimension (or mode) of X along which mat should be multiplied
        option : if 't', performs the same computations as above except the matrices are transposed
        excludedim : if True, multiply along all mode but those specified in the dims parameter
    
        Returns
        -------
        a : ndarray
            The filled array.
        """

        def tt_dimscheck2(dims = None):
            newdims = dims
            
            if(dims == None):
                dims = range(self.ndims)    
            
            if excludedim:
                newdims = range(self.ndims)
                newdims.remove(dims)
                dims = newdims
                
            if(dims.__class__ == int): dims = [dims]
            
            if dims < 0:
                newdims = range(self.ndims)
                newdims.remove(-dims)
                dims = newdims
                
            return dims, dims
        
        #Handle when arrs is a list of arrays
        if(mat.__class__ == list):
            if(len(mat) == 0):
                raise ValueError("the given list of arrays is empty!");

            if True:
                dims, vidx = tt_dimscheck2(dims)
            else:
                from mlabwrap import mlab
                #print "list:{0} ndims{1} len(mat){2}".format(dims, self.ndims(), len(mat))
                dims, vidx = mlab.tt_dimscheck([i+1 for i in dims], self.ndims, len(mat), nout = 2)
                if len(dims.T[0]) > 1:
                    dims, vidx = [(int(k)-1, int(l)-1) for k,l in (dims.T[0], vidx.T[0])]
                else:
                    dims, vidx = [int(dims.T[0]) - 1], [int(vidx.T[0]) - 1]
                #print dims,vidx
                
            Y = self.ttm(mat[vidx[0]], dims[0], transpose)
            for i in range(1, len(dims)):
                Y = Y.ttm(mat[vidx[i]], dims[i], transpose)
            return Y
        
        #Begin Check
        if(mat.ndim != 2):
            raise ValueError ("matrix in 2nd armuent must be a matrix!");
        
        if(dims.__class__ == list):
            if(len(dims) != 1):
                raise ValueError("Error in number of elements in dims");
            else:
                dims = dims[0];
                
        if(dims < 0 or dims > self.ndims):  ## BUG correction, before: if(dims < 0 or dims > self.ndims()):
            raise ValueError ("Dimension N must be between 1 and num of dimensions: val = %i" % dims);
        #!End checks

        
        N = self.ndims;
        shp = self.shape;
        order = [dims]+range(0, dims)+range(dims+1, N)
        
        ## tenmat !
        newdata = self.permute(order)
        newdata = newdata.data.reshape(shp[dims], self.data.size/shp[dims])
        #from tenmat import tenmat2
        #newdata = tenmat2(self, numpy.array(order) + 1 )
        ##!end tenmat

        if transpose:
            newdata = numpy.dot(mat.transpose(), newdata)
            p = mat.shape[1]
        else:
            newdata = numpy.dot(mat, newdata)
            p = mat.shape[0]

        newshp = [p] + list(shp[0:dims]) + list(shp[dims+1:N])

        Y = Tensor(newdata, newshp)
        Y = Y.ipermute(order)
        return Y
    
#    def ttm2(self, mat, n, option = None, excludedim = False):
#        import numpy as np
#        
#        if isinstance(mat, list):
#            (dims,vidx) = tools.tt_dimscehck(self.ndims(), n, len(mat))
#            Y = self.ttm(mat[vidx[0]],dims[0],option);
#            for i in range(1, len(dims)):
#                Y = Y.ttm(mat[vidx[i]],dims[i],option = option);
#                
#            return Y 
#        
#        if mat.ndims() != 2:
#            raise ValueError("M is not a matrix: mat.ndims = %i" % mat.ndims())
#        
#        if option == 't': mat = mat.T
#        
#        N = self.ndims()
#        shp = self.shape
#        
#        order = [n]+range(1, n)+range(n+1, N)
#        
#        newdata = self.permute(order).data;
#        newdata = newdata.reshape(shp[n], reduce(np.multiply,
#                                                 [shp[i] for i in order[1:]]))
#
#        newdata = np.dot(mat, newdata);
#        p = mat.shape[0]
#        
#        newshp = [p]
        
        
    def matricization(self, rdims = None, cdims = None, tsize = None):
        """ mode-n unfolding of a tensor
        
        reimplementation of the tenmat class
        
        Parameters
        ----------
        K : Tensor
        rdims : int
        cdims : list
        tsize : int
        
        Returns
        -------
        M : Matrix
        
        """

        import numpy as np
        
        if tsize is None: tsize = self.shape
        nn = np.mgrid[1:self.ndims+1]
        
        if bool(rdims) ^ bool(cdims):
            if rdims is None: rdims = np.setdiff1d(nn, np.array(cdims))
            if cdims is None: cdims = np.setdiff1d(nn, np.array(rdims))

        else:
            raise ValueError("You have to specify either rdims or cdims")
        rdims = np.array(rdims, ndmin = 1)
        cdims = np.array(cdims, ndmin = 1)
        rcdims = np.hstack((rdims, cdims))
        
        if np.setdiff1d(rcdims, nn) or np.setdiff1d(nn, rcdims):
            raise ValueError("Incorrect specification of dimensions." +
                             "\n RDIMS = %s\n CDIMS = %s" % (rdims, cdims))

        T = self.permute(rcdims-1);

        row = reduce(np.multiply, [tsize[i-1] for i in rdims])
        col = reduce(np.multiply, [tsize[i-1] for i in cdims])
        
        return T.data.reshape([row, col])
        #self.rindices = rdims
        #self.cindices = cdims
    
    def nvecs(self, n, r, flipsign = True):
        """ Compute the leading mode-n vectors for a tensor
        
        computes the r leading eigenvalues of Xn*Xn'
        (where Xn is the mode-n matricization of X), which provides
        information about the mode-n fibers. In two-dimensions, the r
        leading mode-1 vectors are the same as the r left singular vectors
        and the r leading mode-2 vectors are the same as the r right
        singular vectors.
        
        Parameters
        ----------
        X : Tensor
        n : int, mode-n matricization of X
        r : int, nnumber of leading eigenvalues to return
        flipsign : bool, make each column's largest element positive / Make the largest magnitude element be positive
        
        Returns
        -------
        M : Matrix
        
        """
        from numpy import dot
        from scipy.sparse.linalg.eigen.arpack import eigen_symmetric
        
        #from tenmat import tenmat2
        #Xn = tenmat2(self, n).data
        Xn = self.matricization(n)
        Y = dot(Xn, Xn.T)

        v = eigen_symmetric(Y, r, which = 'LM')

        if flipsign:
            """ not implemented """
            pass
        return v[1]

    def __str__(self):
        str = "tensor of size {0}\n".format(self.shape);
        str += self.data.__str__();
        return str;

class KTensor:
    """ Tensor stored as a Kruskal operator (decomposed) """
    lambda_ = None
    u = None
    shape = None

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
            self.fit = None

        elif isinstance(kwargs['X'], Tensor):
            self = self.cp_als(**kwargs)
        else:
            raise ValueError("Matrix U'' is not a matrix!")
    
    @property
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

            #K = KTensor(lmda = mylambda, U = U)
            #self = K
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


class TTensor:
    core = None;
    u = None;
    
    def __init__(self, core, uIn):
        """ Create a tucker tensor object with the core and matrices.
        NOTE: uIn has to be a list of arrays/matrices"""
        
        if not isinstance(core, Tensor) :
            raise ValueError("core is not a tensor");
        
        #Handle if the uIn is not a list
        if(uIn.__class__ != list):
            newuIn = [];
            for x in uIn:
                newuIn.extend([x]);
            uIn = newuIn;
           
        newuIn = []; 
        for i in range(0, len(uIn)):
            newuIn.extend([uIn[i].copy()]);
        uIn = newuIn;
        
        # check that each U is indeed a matrix
        for i in range(0,len(uIn)):
            if (uIn[i].ndim != 2):
                raise ValueError("{0} is not a 2-D matrix!".format(uIn[i]));
        
        # Size error checking
        k = core.shape;
        
        if (len(k) != len(uIn)):
            raise ValueError("Number of dims of Core andthe number of matrices are different");
        
        for i in range(0,len(uIn)):
            if (k[i] != len((uIn[i])[0])):
                raise ValueError(
                    "{0} th dimension of Core is different from the number of columns of uIn[i]"
                    .format(i));
         
        self.core = core.copy();
        self.u = uIn;
        
        #save the shape of the TTensor
        shape = [];
        for i in range(0, len(self.u)):
            shape.extend([len(self.u[i])]);
        self.shape = tuple(shape);
        # constructor end #

    def size(self):
        """NEVER USED"""
        ret = 1;
        for i in range(0, len(self.shape)):
            ret = ret * self.data.shape[i];
        return ret;
    
    def copy(self):
        return TTensor(self.core, self.u);

        
    def __str__(self):
        ret = "TTensor of size {0}\n".format(self.shape);
        ret += "Core = {0} \n".format(self.core.__str__());
        for i in range(0, len(self.u)):
            ret += "u[{0}] =\n{1}\n".format(i, self.u[i]);
        
        return ret;


def tucker_als(X, R, fitchangetol=1.0e-5, maxiters=200, verbose=1, init='eigs'):
    """ **TUCKER_ALS** Higher-order orthogonal iteration
    
    It parses options and calls :func:`minify_sources` or :func:`combine_sources`
    if apropriate.
    
    :param sources: Paths of source files
    :param ext: Type of files
    :param fs_root: root of file (normally public dir)
    :type sources: string
    :type ext: js or css
    :type fs_root: string

    :returns: List of paths to minified sources
    """
    from numpy import ones, sqrt
    from scipy.linalg import norm
    
    dimorder = range(X.ndims)
    if isinstance(R, int): R = R * ones(X.ndims, dtype = int)

    _als_check_params(X, R, maxiters, dimorder)
    U = _setup_initial_U(X, R, init, dimorder)
    
    fit = 0
    normX = norm(X.data)
    normresidual = lambda normX, core:\
        sqrt(normX**2 - core.norm()**2)
    
    if verbose: print 'Alternating Least-Squares:\n'
    
    # Main Loop: Iterate until convergence"
    for iter in range(maxiters):
        # Iterate over all N modes of the tensor
        for n in dimorder:
            Utilde = X.ttm(U, n, transpose = True, excludedim = True)
            U[n] = Utilde.nvecs(n+1, R[n])

        # Assemble the current approximation
        core = Utilde.ttm(U, n, 't');

        # Compute fit, fraction explained by model
        newfit = 1 - normresidual(normX, core) / normX
        fitdelta = abs(fit-newfit); fit = newfit
        if verbose: print(" Iter %2d: fit = %e fitdelta = %7.1e"%(iter+1, newfit, fitdelta))
        if iter and fitdelta < fitchangetol: break
    else:
        print "Limit of %i reached without convergence" % maxiters

    return TTensor(core, U), normresidual(normX, core)

def _setup_initial_U(X, R, init, dimorder, verbose = 1):
    """ for Tucker and PARAFAC decomposition
    
    Set up and error checking on initial guess for U.
    
    Parameters
    ----------
    R has to be an array of ndims
    """
    from numpy.random import rand
    from scipy import io
    
    if verbose: print('Initialisation using %s:' % init)
            
    Uinit = [0]*X.ndims
    for n in dimorder[1:]:
        if init == 'random':
            Uinit[n] = rand(X.data.shape[n], R[n])
            
        elif init == 'eigs':
            if verbose: print('  Computing %i leading e-vectors for factor %i'%(R[n], n+1))
            Uinit[n] = X.nvecs(n+1, R[n])
        
        ## TO REMOVE AFTER SEVERE CHECK
        elif init == 'check_eigs':
            Uinit = [x[0] for x in io.loadmat('../bench/claus_eigs_init.mat',
                                              struct_as_record=True)['Ainit']]
            Uinit[0] = 0 
            break
        else:
            raise ValueError('This init method is not supported: %s' % init)
    return Uinit

def _als_check_params(X, R, maxiters, dimorder):
    """ Verbosely warn & error checking on tucker and cp
    parameters
    """
    
    if not isinstance(R, int) and len(R) != X.ndims:
        raise ValueError("R must have the same size as X's ndims")
    
    if maxiters < 0: raise ValueError("maxiters must be positive")
    
    dimorder.sort()
    if range(X.ndims) != dimorder:
        raise ValueError('dimorder=%s must include all elements of 1:X.ndims=%s'%(dimorder,range(X.ndims)))


def mttkrp(X, U, n):
    """Matricized tensor times Khatri-Rao product for tensor
    
    Calculates the matrix product of the n-mode matricization
    of X with the Khatri-Rao product of all entries in U, a list
    of matrices, except the nth.
    
    Parameters
    ----------
    X : Tensor
    U : list of matrices
    n : ...

    Returns
    -------
    C : ...
    """
    from numpy import dot
    
    b = range(n) + range(n+1, X.ndims)
    Xn = X.permute([n]+b)
    Xn.reshape((X.shape[n], X.data.size/X.shape[n]))
    Z = khatrirao(U[b[0]], U[b[1]])
    return dot(Xn.data, Z)


def khatrirao(A, B):
    """ Khatri-Rao product of a and b
    
    Parameters
    ----------
    A : array, shape (I, J)
    B : array, shape (T, J)

    Returns
    -------
    C : array, shape (I*T, J)

    Examples
    --------
    >>> from numpy import array
    >>> from tensor import khatrirao
    >>> khatrirao(array([[1,2],[2,1]]), array([[1,1],[1,1]]))
    array([[1, 2],
           [1, 2],
           [2, 1],
           [2, 1]])
    """
    """
    Todo:
        double check: kron algo
        Enable reverse parameter?: khatrirao(a, reverse = 1)
    """
    from scipy import kron, array, newaxis
    
    #Begin simple checks
    if A.shape[1] != B.shape[1]:
        raise ArithmeticError("khatrirao: Matrices don't have the same number of columns")
    #!End simple checks
    
    C = array([kron(A[:,p][newaxis], B[:,p][newaxis]) for p in range(A.shape[1])])
    return C[:,0,:].T 
