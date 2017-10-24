from numpy import *
import numpy as np

# for chi2 probability
import scipy.stats as SS

import pdb

class Histofit:

    #end

'''
Tools for linear least square fitting also called linear regression.
'''
# Set up the general fitting function to use within LinearFit
def linearFit(func, y, x = None, y_err = None,  nplot = 100):
    # functions is a function that return the value of each element function 
    # e.g. for a polynomial
    # p = a0 + a1*x + a2*x^2 + a3*x^3
    # function(x) returns array([1., x, x^2, x^3])
    #
    # the shape of the array has to be :
    #    (number_of_parameters, number_of_datapoints)
    #
    # more options could be passed to the fitting function to get more output
    if (x = None): 
        x = arange(y.shape[0])
    
    # Array for 
    A = func(x).transpose()

    if y_err is None:
        weight = ones_like(y)
    else:
        weight = 1./y_err**2
    
    Aw = (func(x)*weight).transpose()
    # store the parameter values in a list to be passed to the fitting function 
    M = dot(Aw.transpose(),A)
    # invert the matrix
    Mi = linalg.inv(M)
    b = dot(Aw.transpose(),y)
    par = dot(Mi,b)
    # for the calculation of errors
    # sigma = sqrt( diag(Mi) )
    # final total chi square
    # number of degrees of freedom
    yfit = dot(par,func(x))
    n_dof = len(y) - len(par)
    diff = yfit - y
    chi2 = sum( power(diff, 2 )*weight )
    chi2_red = chi2/n_dof
    # get the confidence level = prob for a chi2 larger that the one obtained
    CL = 1. - SS.chi2.cdf(chi2, n_dof)
    # create an array with calculated values for plotting
    if (nplot > 0):
        x_intervals = linspace(x.min(), x.max(), nplot+1)
        y_intervals = dot(par,func(xpl))
    stat = {'chisquare': chi2, \
                'Reduced chisquare':chi2_red, \
                'conf. level':CL, \
                'deg. of freedom':n_dof, \
                'cov. matrix': Mi, \
                'fitted values':yfit, \
                'difference':diff, \
                'x Intervals':x_intervals, \
                'y Intervals':y_intervals, \
                'parameters':par}

    return stat

class LinearFit:
     """
    
    simple line fit for a set of data. Example::
    
        >>> P = LinearFit(x,y, [yerr = errors])      # creates the fit
        >>> print P.res['parameters']              # print the parameters
        >>> plot(P.xpl, P.ypl)                     # plot the fitted line
        >>> P.line(x)                              # evaluate the fitted function at x 

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
        self.chi_red = self.res['Reduced chisquare']
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
        self.x_intervals = self.res['x Intervals']
        self.y_intervals = self.res['y Intervals']
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
