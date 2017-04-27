CC=gcc
SHELL=/bin/bash

NUMPYROOT=${HOME}/local/numpy

CFLAGS=-Wall -march=core2 -fPIC -shared -fopenmp -O2
CPPFLAGS=-I$(NUMPYROOT)/lib64/python2.7/site-packages/numpy/core/include -I/usr/include/python2.7
LDFLAGS=-lgomp -lpthread -lm -lpython2.7

MODS = gnautilus.so

all: $(MODS)

%.c: %.pyx
	cython $<

%.so: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(MODS)

.PHONY: all clean
