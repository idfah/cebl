"""Real-time filters.
"""

filterChoices = {}

from .filt import *

from .bandpass import FIRBandpass, FIRBandpassName
filterChoices[FIRBandpassName] = FIRBandpass

from .bandpass import IIRBandpass, IIRBandpassName
filterChoices[IIRBandpassName] = IIRBandpass

from .demean import Demean, DemeanName
filterChoices[DemeanName] = Demean

from .ica import IndependentComponentsName, IndependentComponents
filterChoices[IndependentComponentsName] = IndependentComponents

from .moving import MovingAverage, MovingAverageName
filterChoices[MovingAverageName] = MovingAverage

from .msf import MaxSignalFraction, MaxSignalFractionName
filterChoices[MaxSignalFractionName] = MaxSignalFraction

from .pca import PrincipalComponents, PrincipalComponentsName
filterChoices[PrincipalComponentsName] = PrincipalComponents

from .reference import Reference, ReferenceName
filterChoices[ReferenceName] = Reference

from .wiener import Wiener, WienerName
filterChoices[WienerName] = Wiener

from .bsp3 import BiosemiP3, BiosemiP3Name
filterChoices[BiosemiP3Name] = BiosemiP3

from .bsmtpsd import BiosemiMTPSD, BiosemiMTPSDName
filterChoices[BiosemiMTPSDName] = BiosemiMTPSD

from .bsmtcna import BiosemiMTCNA, BiosemiMTCNAName
filterChoices[BiosemiMTCNAName] = BiosemiMTCNA

from .gnmtpsd import GNautilusMTPSD, GNautilusMTPSDName
filterChoices[GNautilusMTPSDName] = GNautilusMTPSD

from .gnmttde import GNautilusMTTDE, GNautilusMTTDEName
filterChoices[GNautilusMTTDEName] = GNautilusMTTDE

from .gnp3 import GNautilusP3, GNautilusP3Name
filterChoices[GNautilusP3Name] = GNautilusP3

from .test import Test, TestName
filterChoices[TestName] = Test
