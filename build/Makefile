# 用到的编译器
CC = g++
NVCC = nvcc
MPI_CC = mpicc

# 编译器 flags
CFLAGS = -Wall -Wextra -std=c++14
NVCCFLAGS = -std=c++14
MPICC_FLAGS = -Wall -Wextra -std=c++14

# CUDA flags 和 libraries
CUDAFLAGS = -arch=sm_75
CUDALIBS = -I/opt/cuda/targets/x86_64-linux/include -L/opt/cuda/lib64 -lcudart -lcuda

# MPI flags and libraries
DISTRIBUTEDLIBS = -lnccl -lmpi

# 路径
SRCDIR = ../zorch/csrc
BUILDDIR = ../build
TARGET = ../zorch/libtensor.so

# Files
SRCS := $(filter-out $(SRCDIR)/distributed.cpp, $(wildcard $(SRCDIR)/*.cpp))
CU_SRCS = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(SRCS))
CU_OBJS = $(patsubst $(SRCDIR)/%.cu, $(BUILDDIR)/%.cu.o, $(CU_SRCS))
MPI_SRCS := $(SRCDIR)/distributed.cpp
MPI_OBJS := $(BUILDDIR)/distributed.o


# Rule to build the target
$(TARGET): $(OBJS) $(MPI_OBJS) $(CU_OBJS)
	$(NVCC) --shared -o $(TARGET) $(OBJS) $(MPI_OBJS) $(CU_OBJS) $(CUDALIBS) $(DISTRIBUTEDLIBS)

# Rule to compile C++ source files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -fPIC -c $< -o $@ $(CUDALIBS)

# Rule to compile CUDA source files
$(BUILDDIR)/%.cu.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CUDAFLAGS) -Xcompiler -fPIC -c $< -o $@

# Rule to compile distributed.cpp with mpiCC
$(BUILDDIR)/distributed.o: $(SRCDIR)/distributed.cpp
	$(MPI_CC) $(MPICC_FLAGS) -fPIC -c $< -o $@

# Clean rule
clean:
	rm -f $(BUILDDIR)/*.o $(TARGET)