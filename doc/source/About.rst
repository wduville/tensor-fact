About --- Draft & Tests
=======================

:Maintainer: Willy Duville
:Authors: Willy Duville <wduville@brain.riken.jp>, Ahn Huy Phan <phan@brain.riken.jp>, Andrzej Cichocki <cia@brain.riken.jp>
:Website: http://www.bsp.brain.riken.jp

Many modern applications generate large amounts of data with multiple aspects and high dimensionality for which tensors (i.e., multi-way arrays)
provide a natural representation.

.. toctree::

	Todolist.rst

.. rubric:: Software involved

#. Python --- http://www.python.org
#. Numpy
#. Scipy
#. Sphinx (easy_install sphinx)

#. Aptana/Pydev
#. Spyder

#. SetupTools
#. `numpydoc <http://numpy.scipy.org/svn/numpy/trunk/doc/HOWTO_DOCUMENT.txt>`_ (easy_install numpydoc)

* http://projects.scipy.org/numpy/wiki/CodingStyleGuidelines

Other Python Numerical and Scientific projects:
	http://wiki.python.org/moin/NumericAndScientific
	
Acknowledgements
----------------
#. Many of those algorithms were borrowed or inspired from the work of KOLDA et. al. ...
#. This project meant to implement algorithms of the Andrzej CICHOCKI's book `*Nonnegative Matrix and Tensor Factorizations* <http://www.tramy.us/>`__.
   Large parts of this documentation originate from this book.
#. By the `Laboratory for Advanced Brain Signal Processing <http://www.bsp.brain.riken.jp/>`_, RIKEN Brain Science Institute

.. image:: images/Wiley-NonnegativeMatrixandTensorFactorizations(2009)_Page_001.png
	:width: 200

	
Testings
--------

+-------------+---------------------------------+-------+
| Operation   | Result                          | Notes |
+=============+=================================+=======+
| ``x or y``  | if *x* is :term:`environment`,  | \(1)  |
|             | then *y*, else *x*              |       |
+-------------+---------------------------------+-------+
| ``x and y`` | if *x* is false, then *x*, else | \(2)  |
|             | *y*                             |       |
+-------------+---------------------------------+-------+
| ``not x``   | if *x* is false, then ``True``, | \(3)  |
|             | else ``False``                  |       |
+-------------+---------------------------------+-------+


.. math::
   :nowrap:

   \begin{eqnarray}
      y    & = & ax^2 + bx + c \\
      f(x) & = & x^2 + 2xy + y^2
   \end{eqnarray}

Imports::

	.. literalinclude:: ../../src/tensor.py
		:pyobject: khatrirao
		:language: python
		:linenos:
		:start-after: #!End

Glossary
--------

.. glossary::

   Numpy
	  http://www.numpy.org

   Scipy
	  Python's Open Source Library of Scientific Tools. `www.scipy.org <http://www.scipy.org>`_

   environment
      A structure where information about all documents under the root is
      saved, and used for cross-referencing.  The environment is pickled
      after the parsing stage, so that successive runs only need to read
      and parse new and changed documents.

   source directory
      The directory which, including its subdirectories, contains all
      source files for one Sphinx project.
