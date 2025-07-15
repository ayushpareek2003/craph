# CRAPH

CRAPH is a lightweight C++ library for high-performance graph traversal on GPUs using CUDA.  
It provides flexible **edge-centric** and **vertex-centric** traversal kernels and modular components for sparse graph representations.


##  Features

**Vertex-centric and Edge-centric traversals** for different workloads
-  CUDA-accelerated parallel primitives
-  Modular sparse graph structures (CSR, CSC, COO)
-  Comprehensive test suite
-  Multiple build systems (Make, CMake)
-  Automatic CUDA memory management

---

##  Getting Started

###  Prerequisites

- **CUDA Toolkit** (CUDA 10.0+ recommended)
- **C++17** compatible compiler (e.g., `g++`, `clang++`)
- **CMake** (optional, for CMake build)

### 📦 Build Options

#### Option 1: Using Make (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/CRAPH.git
cd CRAPH

# Build the library and tests
make

# Run tests
make -C tests run_simple
make -C tests run_comprehensive
make -C tests run_cuda
```

#### Option 2: Using CMake

```bash
# Clone the repository
git clone https://github.com/your-repo/CRAPH.git
cd CRAPH

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run tests
ctest --verbose
```

### 🧪 Running Tests

```bash
# CPU-only tests
./tests/simple_test
./tests/comprehensive_test

# CUDA tests (requires GPU)
./tests/cuda_test
```
### Basic Usage

```cpp
#include "craph_cuda.hpp"

// Load graph
craph::CSR csr("graph.txt", true);

// Create CUDA wrapper
craph::CudaCSR cuda_csr(csr);

// Initialize BFS
craph::CudaBFS bfs(csr.GetGraph().offsets.size() - 1);
bfs.setSource(0);

// Run top-down BFS
bfs.topDownBFS(cuda_csr, 1);

// Get results
std::vector<unsigned int> levels = bfs.getLevels();
```

### Supported Graph Formats

- **CSR (Compressed Sparse Row)**: Efficient for out-edge traversal
- **CSC (Compressed Sparse Column)**: Efficient for in-edge traversal  
- **COO (Coordinate)**: Simple edge-list format

### Traversal Algorithms

- **Top-Down BFS**: Push-based traversal using CSR
- **Bottom-Up BFS**: Pull-based traversal using CSC
- **Edge-Centric BFS**: Edge-parallel traversal using COO

---

## 🛠️ Build Targets

### Make Targets

```bash
make              # Build library and tests
make library      # Build only static library
make tests        # Build and run tests
make clean        # Remove build artifacts
make distclean    # Remove all generated files
make help         # Show available targets
```

### CMake Targets

```bash
cmake --build .           # Build all targets
cmake --build . --target craph        # Build library only
cmake --build . --target simple_test  # Build specific test
ctest                    # Run all tests
```

### Build Verification

```bash
# Check library creation
ls -la lib/libcraph.a

# Verify CUDA compilation
nvcc --version
```
