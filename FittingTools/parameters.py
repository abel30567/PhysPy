"""

These are the fit parameter objects. They store the *name*, the *value*
and the *error* of each parameter and are used in various fitting tools.

-----------------------------------------------

"""
# parameter class
class Parameter:
    """
    Define a parameter as follows::

       >>> import LT_Fit.Parameter as P
       >>> mypar = P.Parameter(0.)
       >>> mypar = P.Parameter(0., 'my_first_parameter')
       >>> mypar = P.Parameter(0., name = 'my_first_parameter')
       >>> mypar = P.Parameter(0., name = 'my_first_parameter', err = 0.1)

    Other possible operations::

       >>> mypar()     # if the parameter is called it returns it's current value only
       >>> mypar.get() # returns the tuple (name, value, error)
       >>> print mypar # print a string representation of the name, value and error

    """
    
    def __init__(self, value, name = 'var', err = 0.):
        self.value = value
        self.err = err
        self.name = name

    def __str__(self):
        return "%s = %r +/- %r"%(self.name, self.value, self.err)

    def set(self, value, name = None, err = None):
        """
        Set the value of a parameter::

           >>> mypar.set(50.) # this value is now 50
           
        ============   =====================================================
        Keyword        Meaning
        ============   =====================================================
        value          set the value
        name           assign a name (string)
        err            set the error
        ============   =====================================================
        
        """
        self.value = value
        if name != None:
            self.name = name
        if err != None:
            self.err = err

    def get(self):
        """
        
        return the current value

        """
        return self.name, self.value, self.err

    def __call__(self):
        # this is to use is like : param()
        return self.value
