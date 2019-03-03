"""Tools for working with iterators.
"""
import collections


def isiterable(x):
    """Return True if we can iterate over x, otherwise False.
    """
    return isinstance(x, collections.Iterable)
