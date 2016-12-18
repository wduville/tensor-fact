'''
Created on May 6, 2010

@author: Willy
'''
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
            
        if(self.ndims() != len(order)):
            raise ValueError("Invalid permutation order: ndims %i, order %s"
                             % (self.ndims(), order))
           
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
            remdims = numpy.setdiff1d(range(self.ndims()), dims)
            if self.ndims() > 1:
                c = self.permute(numpy.concatenate([remdims, numpy.array(dims)]))
                c = c.data
                
            n = self.ndims() - 1
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
                dims = range(self.ndims())    
            
            if excludedim:
                newdims = range(self.ndims())
                newdims.remove(dims)
                dims = newdims
                
            if(dims.__class__ == int): dims = [dims]
            
            if dims < 0:
                newdims = range(self.ndims())
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
                dims, vidx = mlab.tt_dimscheck([i+1 for i in dims], self.ndims(), len(mat), nout = 2)
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
                
        if(dims < 0 or dims > self.ndims()):  ## BUG correction, before: if(dims < 0 or dims > self.ndims()):
            raise ValueError ("Dimension N must be between 1 and num of dimensions: val = %i" % dims);
        #!End checks

        
        N = self.ndims();
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
        nn = np.mgrid[1:self.ndims()+1]
        
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