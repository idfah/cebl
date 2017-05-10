CC=icc
SHELL=/bin/bash

NUMPYROOT=${HOME}/local/numpy
INTELROOT=${HOME}/local/intel

CFLAGS=-Wall -march=core2 -fPIC -shared -qopenmp -O3 #-fp-model precise #-no-fast-transcendentals
CPPFLAGS=-I$(NUMPYROOT)/lib64/python2.7/site-packages/numpy/core/include -I/usr/include/python2.7
LDFLAGS=-L$(INTELROOT)/lib/intel64 -liomp5 -lpthread -limf -lpython2.7

MODS = wxlibplot.so

all: $(MODS)

%.c: %.pyx
	cython $<

%.so: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(MODS)

.PHONY: all clean
