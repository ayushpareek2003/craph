# CRAPH

**CRAPH** is a lightweight C++ library for high-performance graph traversal on GPUs using CUDA.  
It provides flexible **edge-centric** and **vertex-centric** traversal kernels and modular components for sparse graph representations.

---

## 📂 Project Structure

CRAPH/
├── inc/ # Header files
│ ├── kernels/ # Kernel headers
│ │ ├── edge_centric.cuh
│ │ └── vertex_centric.cuh
│ └── sparse_Structs.hpp
├── src/ # Source files
│ ├── kernels/ # Kernel implementations
│ │ ├── edge_centric.cu
│ │ └── vertex_centric.cu
│ └── sparse/ # Sparse graph structures
│ └── sparse_structs.cpp
├── tests/ # Unit and example tests
│ ├── simple_test.cpp
│ └── test_sparse_formats.cpp
├── graph.txt # Example input graph
├── Makefile # Build configuration
└── README.md

yaml
Copy
Edit

---

## ⚡ Features

- 🧵 **Vertex-centric and Edge-centric traversals** for different workloads.
- ⚙️ CUDA-accelerated parallel primitives.
- 🗂️ Modular sparse graph structures.
- ✅ Example tests and input graph.
- 📦 Simple Makefile build system.

---

## 🚀 Getting Started

### 🔧 Prerequisites

- CUDA Toolkit installed (CUDA 10.0+ recommended)
- C++17 compatible compiler (e.g., `g++`, `clang++`)

---

### 📦 Build

```bash
# Clone the repository
git clone https://github.com/ayushpareek2003/CRAPH.git
cd CRAPH

# Build the library and tests
make
🧪 Run Example
bash
Copy
Edit
# Run the simple test
./simple_test
Edit graph.txt to test your own graphs.

🧩 File Overview
File/Folder	Description
inc/kernels/	CUDA kernel headers
src/kernels/	CUDA kernel implementations
inc/sparse_Structs.hpp	Sparse graph structures
tests/	Example unit tests
graph.txt	Sample graph input