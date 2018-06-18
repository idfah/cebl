"""Parallel processing utilities.
"""

import multiprocessing as mp


class ReadWriteLock(object):
    """Read-Write lock with writer preference.
    """
    def __init__(self):
        self.rLock = mp.Lock()
        self.wLock = mp.Lock()

        self.rcLock = _CountLock(self.wLock)
        self.wcLock = _CountLock(self.rLock)

        self.reLock = mp.Lock()

    def acquireReadLock(self):
        self.reLock.acquire()
        self.rLock.acquire()
        self.rcLock.acquire()
        self.rLock.release()
        self.reLock.release()

    def releaseReadLock(self):
        self.rcLock.release()

    def acquireWriteLock(self):
        self.wcLock.acquire()
        self.wLock.acquire()

    def releaseWriteLock(self):
        self.wLock.release()
        self.wcLock.release()

    def getReadLock(self):
        return _RWReadLock(self)

    def getWriteLock(self):
        return _RWWriteLock(self)

class _CountLock(object):
    def __init__(self, lock):
        self.lock = lock
        self.counter = mp.Value('i', 0)
        self.counterLock = mp.Lock()

    def acquire(self):
        self.counterLock.acquire()
        self.counter.value += 1
        if self.counter.value == 1:
            self.lock.acquire()
        self.counterLock.release()

    def release(self):
        self.counterLock.acquire()
        self.counter.value -= 1
        if self.counter.value == 0:
            self.lock.release()
        self.counterLock.release()

class _RWReadLock(ReadWriteLock):
    def __init__(self, rwLock):
        self.rwl = rwLock

    def acquire(self):
        self.rwl.acquireReadLock()

    def release(self):
        self.rwl.releaseReadLock()

    def __enter__(self):
        self.rwl.acquireReadLock()

    def __exit__(self, kind, value, tb):
        self.rwl.releaseReadLock()

class _RWWriteLock(ReadWriteLock):
    def __init__(self, rwLock):
        self.rwl = rwLock

    def acquire(self):
        self.rwl.acquireWriteLock()

    def release(self):
        self.rwl.releaseWriteLock()

    def __enter__(self):
        self.rwl.acquireWriteLock()

    def __exit__(self, kind, value, tb):
        self.rwl.releaseWriteLock()
