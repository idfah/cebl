"""Data structures.
"""

#XXX: this could be done with munch
class Holder:
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
