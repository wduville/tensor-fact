from tensor import Tensor
from ttensor import tucker_als
from ktensor import KTensor


def _setup_initial_U(X, R, init, dimorder, verbose = 1):
    """ for Tucker and PARAFAC decomposition
    
    Set up and error checking on initial guess for U.
    
    Parameters
    ----------
    R has to be an array of ndims
    """
    from numpy.random import rand
    from scipy import io
    
    Uinit = [0]*X.ndims()
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
    
    if not isinstance(R, int) and len(R) != X.ndims():
        raise ValueError("R must have the same size as X's ndims")
    
    if maxiters < 0: raise ValueError("maxiters must be positive")
    
    dimorder.sort()
    if range(X.ndims()) != dimorder:
        raise ValueError('dimorder=%s must include all elements of 1:X.ndims=%s'%(dimorder,range(X.ndims())))


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
    
    b = range(n) + range(n+1, X.ndims())
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
