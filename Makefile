INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcusolver -lcublas -lcublasLt -lcuda

GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_86 --ptxas-options=-v -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

MUJEXAMPLEEXE = muj_cusolver_example.exe

all: clean mujexample

mujexample: cuRandomSVD_wrapper.o Makefile
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) -shared -Xcompiler -fPIC -o libcuRSVD.so cuRandomSVD_wrapper.o

cuRandomSVD_wrapper.o: 
	$(NVCC) -c cuRandomSVD_wrapper.cu -Xcompiler -fPIC $(NVCCFLAGS)

clean:	
	rm -f *.o *.so *.~ $(MUJEXAMPLEEXE)


