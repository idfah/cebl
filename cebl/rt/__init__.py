# hack to prevent segfault in wxPython3.x, forces wxAgg backend XXX - idfah
# https://github.com/matplotlib/matplotlib/issues/3316
# note, the cebl startup script also sets the backend.  This is just in case
# cebl is started from the console.
import matplotlib.pyplot as plt
plt.switch_backend("WXAgg")

from . import events
from . import logging
from . import main
from . import manager
from . import pages
from . import sources
from . import widgets
