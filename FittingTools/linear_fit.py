"""

Tools for linear least square fitting also called linear regression.


"""

from numpy import *
import numpy as np

# for chi2 probability
import scipy.stats as SS

import pdb

from parameters import *

# setup the general fit function
def linfit(function, y, x = None, y_err = None,  nplot = 100):
    # functions is a function that return the value of each element function 
    # e.g. for a polynomial
    # p = a0 + a1*x + a2*x^2 + a3*x^3
    # function(x) returns array([1., x, x^2, x^3])
    #
    # the shape of the array has to be :
    #    (number_of_parameters, number_of_datapoints)
    #
    # more options could be passed to the fitting function to get more output
    if x is None: x = arange(y.shape[0])
    #
    # pdb.set_trace()
    A = function(x).transpose()
    if y_err is None:
        weight = ones_like(y)
    else:
        weight = 1./y_err**2
    Aw = (function(x)*weight).transpose()
    # store the parameter values in a list to be passed to the fitting function 
    M = dot(Aw.transpose(),A)
    # invert the matrix
    Mi = linalg.inv(M)
    b=dot(Aw.transpose(),y)
    par = dot(Mi,b)
    # for the calculation of errors
    # sigma = sqrt( diag(Mi) )
    # final total chi square
    # number of degrees of freedom
    yfit = dot(par,function(x))
    n_dof = len(y) - len(par)
    diff = yfit - y
    chi2 = sum( power(diff, 2 )*weight )
    chi2_red = chi2/n_dof
    # get the confidence leve = prob for a chi2 larger that the one obtained
    CL = 1. - SS.chi2.cdf(chi2, n_dof)
    # create an array with calculated values for plotting
    if (nplot > 0):
        xpl = linspace(x.min(), x.max(), nplot+1)
        ypl = dot(par,function(xpl))
    stat = {'chisquare': chi2, \
                'red. chisquare':chi2_red, \
                'conf. level':CL, \
                'deg. of freedom':n_dof, \
                'cov. matrix': Mi, \
                'fitted values':yfit, \
                'difference':diff, \
                'xpl':xpl, \
                'ypl':ypl, \
                'parameters':par}

    return stat


class linefit:
    """
    
    simple line fit for a set of data. Example::
    
        >>> R = linefit(x,y, [yerr = errors])      # do the fit
        >>> print R.res['parameters']              # print the parameters
        >>> plot(R.xpl, R.ypl)                     # plot the fitted line
        >>> R.line(x)                              # evaluate the fitted function at x 

    x and y are arrays (:func:`numpy.array`)
    R is an object containing the results of the fit with the following additional useful members:

    ============= ================================
    Member        Meaning
    ============= ================================
    offset        offset of the line
    slope         slope of the line
    sigma_o       error in  offset 
    sigma_s       error in slope
    chi_red       reduce chi square
    CL            probability to find a chi square larger than the fit result (ideal 50%)
    cov           covariance matrix (2D :func:`numpy.array`)
    res           dictionary with all fit information
    ============= ================================
    
    NOTE: to use the covariance matrix you should scale it with the reduced chi square
    

    """
    def __init__(self, x, y, yerr=None, quiet = False):
        self.res = linfit(self.__line_f__, y, x=x, nplot = 2, y_err = yerr)
        self.chi_red = self.res['red. chisquare']
        self.CL = self.res['conf. level']
        # if no error is given scale covariance matrix
        self.sigma_o = sqrt(self.res['cov. matrix'][0,0])
        self.sigma_s = sqrt(self.res['cov. matrix'][1,1])
        self.par = self.res['parameters']
        self.cov = self.res['cov. matrix']
        if yerr is None:
            # data w/o error scale cov. matrix to get chi2 of 1
            self.cov *= self.chi_red
            self.sigma_o *=np.sqrt(self.chi_red)
            self.sigma_s *=np.sqrt(self.chi_red)
            self.common_error = np.sqrt(self.chi_red)
        self.offset = self.par[0]
        self.slope =  self.par[1]
        # for plotting
        self.xpl = self.res['xpl']
        self.ypl = self.res['ypl']
        # print fit information
        if not quiet:
            print "chisq/dof = ", self.chi_red
            print "offset  = ", self.offset, " +/- ", self.sigma_o
            print "slope = ", self.slope, " +/- ", self.sigma_s

    def __getitem__(self,x):
        return self.res[x]

    def __line_f__(self,x):
        return array([ones_like(x),x])

    def line(self,x):
        return dot(self.par, self.__line_f__(x))

# end of class linefit

class polyfit:
    """
    
    Polynomial fit of order n, example::
    
        >>> R = polyfit(x,y, order, [yerr = errors]) # perform the fit 
        >>> print R['parameters']                    # R is a dictionary containing the results of the fit
        >>> plot(R.xpl, R.ypl)                       # plot the fitted curve

    x and y are arrays (:func:`numpy.array`)
    If you want to use predefined parameters to store the results::

        >>> C0 = P.Parameter(0., 'const'); C1 = P.Parameter(0., 'lin'); C2 = P.Parameter(0.,'quad') # create Parameter objects
        >>> R = polyfit(x,y, order, parameters=[C0,C1,C2])                                          # perform the fit
        >>> R.show_parameters()                                                                     # will show the fit result


    ============= ================================
    Member        Meaning
    ============= ================================
    chi_red       reduce chi square
    CL            probability to find a chi square larger than the fit result (ideal 50%)
    par           parameters (:func:`numpy.array`)
    parameters    parameter (list of :class:`~LT_Fit.parameter.Parameter` objects)
    sig_par       parameter errors (variances)
    cov           covarance matric (2D :func:`numpy.array`)
    res           dictionary with all fit information
    poly(x)       evaluate the fitted function at x
    ============= ================================

    NOTE: to use the covariance matrix you should scale it with the reduced chi square

    """
    def __init__(self,x, y, yerr=None, order = 2, np_scale = 5, parameters = None, quiet = False):
        # plot 5 calculated points per exp. point
        self.npoints = len(x)*np_scale
        self.order = order
        # store the list of parameter objects
        if parameters is not None:
            if len(parameters) != self.order + 1:
                print 'parameter list is inconsistent with polynomial order !'
                parameters = None
        self.parameters = parameters
        # do the fit
        self.res = linfit(self.__my_poly__, y, x=x, nplot = self.npoints, y_err = yerr)
        self.chi_red = self.res['red. chisquare']
        self.CL = self.res['conf. level']
        self.par = self.res['parameters']
        self.cov = self.res['cov. matrix']
        if yerr is None:
            # data w/o error scale cov. matrix to get chi2 of 1
            self.cov *= self.chi_red
            self.common_error = np.sqrt(self.chi_red)        
        self.sig_par = sqrt(self.cov.diagonal())
        self.__setup_parameters__()
        # for plotting
        self.xpl = self.res['xpl']
        self.ypl = self.res['ypl']
        if not quiet:
            print "chisq/dof = ", self.chi_red
            for i,v in enumerate(self.res['parameters']):
                print 'parameter [',i,'] = ',v, " +/- ", self.sig_par[i]
        # that's it

    def __getitem__(self,x):
        return self.res[x]

    def __my_poly__(self, x):
        # function for fitting a polynomial
        # this is correnly not implemented
        # general polynomial
        #def gen_poly(x, powers):
        #    xx = x**powers[0]
        #    for p in powers[1:]:
        #        xx = vstack( (xx, x**p) )
        #    return xx
        # regilar polynomial: a0 + a1*x + a2*x^2 + a3*x^3 + ... 
        # order 0
        xx = ones_like(x)
        xs = ones_like(x)
        # remaining orders
        for i in range(self.order):
            xx *= x
            xs = vstack( (xs, xx) )
        return xs
    def __setup_parameters__(self):
        # store the parameters for results in a list of parameter objects
        if self.parameters is None:
            self.parameters = []
            for i,p in enumerate(self.par):
                self.parameters.append( Parameter(p, "a%d"%(i), self.sig_par[i] ) )
        else:
            for i,p in enumerate(self.par):
                self.parameters[i].set(p, err = self.sig_par[i])
        # done

    def show_parameters(self):
        """

        show the current parameter data

        """
        for p in self.parameters:
            print p

    def poly(self,x):
        """

        The fitted function with the current parameters. It can be used in further calculations::

            >>> z = R.poly(x) # z contains the value of the fitted function at x

        """
        
        # return the fitted function value for variable x
        return dot(self.par, self.__my_poly__(x))

class gen_linfit:
    """

    general linear fit of::
    
        f(x) = a0*f0(x) + a1*f1(x) + ...
    
    where::
    
        f = [f0, f1, f2, ..]

    is a list of functions provided by the user.

    An example of making a list of function and using it in a fit::

        >>> f0 = lambda x: sin(x)                                                             # define lambda functions
        >>> f1 = lambda x: sin(2.*x)
        >>> f2 = lambda x: sin(4.*x)
        >>> a0 = P.Parameter(0.,'ax'); a1 = P.Parameter(0.,'a2x'); a2 = P.Parameter(0.,'a4x') # define Parameter objects (optional)
        >>> R = gen_linfit([f0, f1, f2], x, y, parameters = [a0, a1, a2], yerr = sig_y)       # do the fit
        >>> plot(R.xpl, R.ypl)                                                                # plot the fit
        >>> R.show_parameters()                                                               # print the parameters
        
    R is a gen_linfit object containing the fit results and the fitted function

    ============= ================================
    Member        Meaning
    ============= ================================
    chi_red       reduce chi square
    CL            probability to find a chi square larger than the fit result (ideal 50%)
    par           parameters (numpy array)
    parameters    parameter (list of Parameter objects)
    sig_par       parameter errors (variances)
    cov           covariance matrix (2D :func:`numpy.array`)
    res           dictionary with all fit information
    func(x)       evaluate the fitted function at x
    ============= ================================

    NOTE: to use the covariance matrix you should scale it with the reduced chi square

    """
    def __init__(self, functions, x, y,  parameters = None, yerr = None, np_scale = 5, quiet = False):
        # plot 5 calculated points per exp. point
        self.functions = []
        # vectorize function
        for f in functions:
            self.functions.append( vectorize(f) )
        self.npoints = len(x)*np_scale
        self.res = linfit(self.__fit_func__, y, x=x, nplot = self.npoints, y_err = yerr)
        # store fit results
        self.chi_red = self.res['red. chisquare']
        self.CL = self.res['conf. level']
        self.par = self.res['parameters']
        self.cov = self.res['cov. matrix']
        # only multipy with chired if there were no errors given
        if yerr is None:
            self.cov *= self.chi_red
            self.common_error = np.sqrt(self.chi_red)
        self.sig_par = sqrt(self.cov.diagonal()*self.chi_red)
        # store the list of parameter objects
        if parameters is not None:
            if len(parameters) != len(functions):
                print 'parameter list is inconsistent with function list !'
                parameters = None
        self.parameters = parameters
        self.__setup_parameters__()
        # for plotting
        self.xpl = self.res['xpl']
        self.ypl = self.res['ypl']
        if not quiet:
            print "chisq/dof = ", self.chi_red
            for i,v in enumerate(self.res['parameters']):
                print 'parameter [',i,'] = ',v, " +/- ", self.sig_par[i]
        # that's it

    def __getitem__(self,x):
        return self.res[x]

    def __setup_parameters__(self):
        # store the for results in a list of parameter objects
        if self.parameters is None:
            self.parameters = []
            for i,p in enumerate(self.par):
                self.parameters.append( Parameter(p, "parameter_%d"%(i), self.sig_par[i] ) )
        else:
            for i,p in enumerate(self.par):
                self.parameters[i].set(p, err = self.sig_par[i])
        # done

    def __fit_func__(self, x):
        # function for general linear fitting
        y=self.functions[0](x)
        for f in self.functions[1:]:
            y = vstack( (y, f(x)) )
        # pdb.set_trace()
        return y
    
    def func(self,x):
        """
        
        return the fitted function value for variable x

        """
        return dot(self.par, self.__fit_func__(x))

    def show_parameters(self):
        """

        print the parameters

        """
        for p in self.parameters:
            print p
# done
