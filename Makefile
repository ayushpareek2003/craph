# CRAPH GPU Graph Traversal Library Makefile
# Supports both CPU and CUDA components

# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -fPIC
NVCCFLAGS = -std=c++17 -O2 -arch=sm_60 -Xcompiler -fPIC
INCLUDES = -I./inc

# Directories
SRCDIR = src
INCDIR = inc
LIBDIR = lib
OBJDIR = obj
TESTDIR = tests

# Source files
CPU_SOURCES = $(SRCDIR)/sparse_structs.cpp
CUDA_SOURCES = $(SRCDIR)/kernels/vertex_centric.cu $(SRCDIR)/kernels/edge_centric.cu

# Object files
CPU_OBJECTS = $(CPU_SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)

# Library name
LIBRARY_NAME = libcraph.a
LIBRARY_PATH = $(LIBDIR)/$(LIBRARY_NAME)

# Default target
all: $(LIBRARY_PATH) tests

# Create directories
$(OBJDIR):
	mkdir -p $(OBJDIR)
	mkdir -p $(OBJDIR)/kernels

$(LIBDIR):
	mkdir -p $(LIBDIR)

# Build static library
$(LIBRARY_PATH): $(LIBDIR) $(CPU_OBJECTS) $(CUDA_OBJECTS)
	ar rcs $@ $(CPU_OBJECTS) $(CUDA_OBJECTS)
	@echo "Static library built: $@"

# Compile CPU source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Build tests
tests: $(LIBRARY_PATH)
	$(MAKE) -C $(TESTDIR) LIB_PATH=../$(LIBRARY_PATH) all

# Install library
install: $(LIBRARY_PATH)
	@echo "Installing library..."
	# Add installation commands here if needed

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(LIBDIR)
	$(MAKE) -C $(TESTDIR) clean

# Clean everything including tests
distclean: clean
	rm -f simple_test comprehensive_test sparse sparse_test

# Show help
help:
	@echo "CRAPH GPU Graph Traversal Library Build System"
	@echo "=============================================="
	@echo "Available targets:"
	@echo "  all          - Build library and tests"
	@echo "  library      - Build only the static library"
	@echo "  tests        - Build and run tests"
	@echo "  install      - Install library"
	@echo "  clean        - Remove build artifacts"
	@echo "  distclean    - Remove all generated files"
	@echo "  help         - Show this help"

# Phony targets
.PHONY: all tests install clean distclean help

# Dependencies
$(CPU_OBJECTS): $(INCDIR)/sparse_Structs.hpp
$(CUDA_OBJECTS): $(INCDIR)/kernels/vertex_centric.cuh $(INCDIR)/kernels/edge_centric.cuh 