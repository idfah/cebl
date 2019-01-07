from collections import OrderedDict as odict


class Cache:
    def __init__(self, maxSize=10):
        self.maxSize = maxSize
        self.enabled = True
        self.store = odict()

    def setMaxSize(self, maxSize):
        self.maxSize = maxSize

    def getMaxSize(self):
        return self.maxSize

    def getSize(self):
        return len(self.store)

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def clear(self):
        self.store = odict()

    def __getitem__(self, key):
        if not self.enabled or self.maxSize < 1:
            return None

        if key in self.store:
            value = self.store.pop(key)
            self.store[key] = value
            return value

        return None

    def __setitem__(self, key, value):
        if not self.enabled or self.maxSize < 1:
            return

        if key in self.store:
            self.store.pop(key)

        self.store[key] = value

        if len(self.store) > self.maxSize:
            self.store.popitem(last=False)

    def __contains__(self, value):
        return value in self.store

    def __repr__(self):
        return repr([i for i in self.store.items()])

    def __str__(self):
        return str([i for i in self.store.items()])

    def __len__(self):
        return len(self.store)

if __name__ == '__main__':
    cache = Cache(3)

    print('adding a: 1')
    cache['a'] = 1

    print('adding b: 2')
    cache['b'] = 2

    print('adding c: 3')
    cache['c'] = 3

    print(cache)

    print('getting b')
    print(cache['b'])

    print('adding a')
    cache['a'] = 10

    print(cache)

    print('adding d')
    cache['d'] = 4

    print('getting x')
    print(cache['x'])

    print(cache)
    print(len(cache))

    print('d' in cache)
    print('c' in cache)
