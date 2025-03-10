CC = gcc 
CFLAGS = -O2 -Wall

ifeq ($(shell uname -s),Darwin)
    LDFLAGS = -framework OpenCL
else
    LDFLAGS = -L/usr/local/cuda/lib64 -lOpenCL
endif

# might not be necessary, uncomment if you need it
# INCFLAGS = -I/usr/local/cuda/include
MATHFLAG = -lm

all: run
run: main.c kernel.cl ../../opencl_utils.h
	$(CC) $(CFLAGS) -o run main.c $(INCFLAGS) $(LDFLAGS) $(MATHFLAG)
clean:
	rm -f run
