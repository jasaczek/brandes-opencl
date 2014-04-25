CPPFLAGS = -Wall -I /usr/local/cuda-5.5/include/ -std=c++0x
LDFLAGS = -L /usr/lib64/nvidia
LDLIBS = -lOpenCL -lstdc++

EXECS = brandes brandes_verifier

all: $(EXECS)

brandes.o: brandes.cpp brandes.hpp

brandes_verifier.o: brandes_verifier.cpp

.PHONY: clean TARGET
clean:
	rm -f $(EXECS) *.o *~
