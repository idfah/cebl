"""Real-time data acquisition sources and drivers.
"""

import sys

sourceList = []
defaultSource = 'Random'

if sys.platform.startswith('linux'):
    from .biosemi import ActiveTwoConfigPanel, ActiveTwo
    sourceList.append(ActiveTwo)

    from .neuropulse import Mindset24RConfigPanel, Mindset24R
    sourceList.append(Mindset24R)

    #from .gtec import GMobiLabPlusConfigPanel, GMobiLabPlus
    #sourceList.append(GMobiLabPlus)

    #from .gtec import GNautilusConfigPanel, GNautilus
    #sourceList.append(GNautilus)

from .erptest import ERPTestConfigPanel, ERPTest
sourceList.append(ERPTest)

from .mttest import MTTestConfigPanel, MTTest
sourceList.append(MTTest)

from .openbci import OpenBCIConfigPanel, OpenBCI
sourceList.append(OpenBCI)

from .random import RandomConfigPanel, Random
sourceList.append(Random)

from .replay import ReplayConfigPanel, Replay
sourceList.append(Replay)

from .wavegen import WaveGenConfigPanel, WaveGen
sourceList.append(WaveGen)
