from cebl import ml

from .strans import STrans, STransConfigPanel

MaxSignalFractionName = 'Max Signal Fraction'


class MaxSignalFraction(STrans):
    def __init__(self, *args, **kwargs):
        STrans.__init__(self, *args, stransClass=ml.MSF, name=MaxSignalFractionName,
                configPanelClass=STransConfigPanel, **kwargs)
