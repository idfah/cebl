from cebl import ml

from .strans import STrans, STransConfigPanel

PrincipalComponentsName = "Principal Components"


class PrincipalComponents(STrans):
    def __init__(self, *args, **kwargs):
        STrans.__init__(self, *args, stransClass=ml.PCA, name=PrincipalComponentsName,
                configPanelClass=STransConfigPanel, **kwargs)
