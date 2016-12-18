'''
Created on May 6, 2010

@author: Willy
'''
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


def tucker_als(self, X, R, fitchangetol=1.0e-5, maxiters=200, verbose=1, init='eigs'):
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
    
    dimorder = range(X.ndims())
    if isinstance(R, int): R = R * ones(X.ndims(), dtype = int)

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
