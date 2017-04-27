"""Real-time data acquisition sources and drivers.
"""

sourceList = []
defaultSource = 'Random'

from biosemi import ActiveTwoConfigPanel, ActiveTwo
sourceList.append(ActiveTwo)

from erptest import ERPTestConfigPanel, ERPTest
sourceList.append(ERPTest)

from openbci import OpenBCIConfigPanel, OpenBCI
sourceList.append(OpenBCI)

from neuropulse import Mindset24RConfigPanel, Mindset24R
sourceList.append(Mindset24R)

from random import RandomConfigPanel, Random
sourceList.append(Random)

from replay import ReplayConfigPanel, Replay
sourceList.append(Replay)

from wavegen import WaveGenConfigPanel, WaveGen
sourceList.append(WaveGen)
