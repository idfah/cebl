"""Real-time BCI Modules, called Pages.
"""

pageList = []

from .config import Config
pageList.append(Config)

from .filt import Filter
pageList.append(Filter)

from .trace import Trace
pageList.append(Trace)

from .power import Power
pageList.append(Power)

from .specgram import Spectrogram
pageList.append(Spectrogram)

from .textstim import TextStim
pageList.append(TextStim)

from .p300bot import P300Bot
pageList.append(P300Bot)

from .p300grid import P300Grid
pageList.append(P300Grid)

#from .bciplayer import BCIPlayer
#pageList.append(BCIPlayer)

from .mentaltasks import MentalTasks
pageList.append(MentalTasks)

from .motorpong import MotorPong
pageList.append(MotorPong)

from .pieern import PieERN
pageList.append(PieERN)

from .freesom import FreeSOM
pageList.append(FreeSOM)
