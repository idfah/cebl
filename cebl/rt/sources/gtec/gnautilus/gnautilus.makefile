CC=gcc
SHELL=/bin/bash

NUMPYROOT=${HOME}/local/numpy

CFLAGS=-Wall -march=core2 -fPIC -shared -fopenmp -O2
CPPFLAGS=-I$(NUMPYROOT)/lib64/python3.5/site-packages/numpy/core/include -I/usr/include/python3.5m
LDFLAGS=-lgomp -lpthread -lm -lpython3.5m

MODS = gnautilus.so

all: $(MODS)

%.c: %.pyx
	cython3 $<

%.so: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(MODS)

.PHONY: all clean
