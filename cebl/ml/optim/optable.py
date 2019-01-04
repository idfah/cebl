class Optable:
    """Base class/interface for classes that can be optimized using the routines
    in the optable module.  This class holds the abstract methods that should/can
    be overridden and the documentation regarding their arguments and usage.
    """

    def parameters(self):
        """Return a 1d numpy array view of the parameters to optimize.  This
        view will be modified in place.  This method MUST be overridden by
        ALL implementations of Optable.
        """
        raise NotImplementedError('parameters not implemented.')

    def error(self):
        """Return a scalar error metric for the current state of the
        Optable implementation.  This method MUST be overridden by ALL
        implementations of Optable.
        """
        raise NotImplementedError('error not implemented.')

    def gradient(self, returnError=True):
        """Return a 1d numpy array holding the gradient of the parameters
        to optimize.  This method must be overridden in order to use
        optimization routines that require a first-order gradient.
        """
        raise NotImplementedError('gradient not implemented.')

    def gradient2(self, returnError=True):
        """Return a 1d numpy array holding the 2nd order gradient of the
        parameters to optimize.  This method must be overridden in order
        to use optimization routines that require a second-order gradient.
        """
        raise NotImplementedError('gradient2 not implemented.')
