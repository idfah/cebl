"""Miscellaneous functions that are generally useful.
"""

import sys

from .arr import *
from .attr import *
from .cache import *
from .clsm import *
from .comp import *
from .crc import *
from .ds import *
from .embed import *
from .errm import *
#from .fasttanh import *
#from .fastmult import *
from .func import *
from .pack import *
from .parallel import *
from .shuffle import *
from .stats import *

#if sys.platform.startswith('linux'):
#    from .fasttanh import tanh
#else:
#    from numpy import tanh
from numpy import tanh

