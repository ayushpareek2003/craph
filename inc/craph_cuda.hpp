#ifndef CRAPH_CUDA_HPP
#define CRAPH_CUDA_HPP

#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <memory>
#include "sparse_Structs.hpp"
#include "kernels/vertex_centric.cuh"
#include "kernels/edge_centric.cuh"

namespace craph {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device memory management class
class CudaMemoryManager {
public:
    template<typename T>
    static T* allocateDevice(size_t size) {
        T* d_ptr;
        CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(T)));
        return d_ptr;
    }

    template<typename T>
    static void copyToDevice(T* d_ptr, const T* h_ptr, size_t size) {
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    static void copyFromDevice(T* h_ptr, const T* d_ptr, size_t size) {
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    template<typename T>
    static void freeDevice(T* d_ptr) {
        CUDA_CHECK(cudaFree(d_ptr));
    }
};

// CUDA wrapper for CSR format
class CudaCSR {
private:
    CSR* d_csr;
    int* d_offsets;
    int* d_indices;
    float* d_values;
    size_t num_vertices;
    size_t num_edges;

public:
    CudaCSR(const CSR& csr) {
        const auto& graph = csr.GetGraph();
        num_vertices = graph.offsets.size() - 1;
        num_edges = graph.indices.size();

        // Allocate device memory
        d_offsets = CudaMemoryManager::allocateDevice<int>(graph.offsets.size());
        d_indices = CudaMemoryManager::allocateDevice<int>(graph.indices.size());
        
        if (graph.weighted) {
            d_values = CudaMemoryManager::allocateDevice<float>(graph.values.size());
        }

        // Copy data to device
        CudaMemoryManager::copyToDevice(d_offsets, graph.offsets.data(), graph.offsets.size());
        CudaMemoryManager::copyToDevice(d_indices, graph.indices.data(), graph.indices.size());
        
        if (graph.weighted) {
            CudaMemoryManager::copyToDevice(d_values, graph.values.data(), graph.values.size());
        }

        // Create device CSR structure
        d_csr = CudaMemoryManager::allocateDevice<CSR>(1);
        CSR h_csr_temp = csr; // Copy constructor
        h_csr_temp.g.offsets.clear();
        h_csr_temp.g.indices.clear();
        h_csr_temp.g.values.clear();
        CudaMemoryManager::copyToDevice(d_csr, &h_csr_temp, 1);
    }

    ~CudaCSR() {
        if (d_offsets) CudaMemoryManager::freeDevice(d_offsets);
        if (d_indices) CudaMemoryManager::freeDevice(d_indices);
        if (d_values) CudaMemoryManager::freeDevice(d_values);
        if (d_csr) CudaMemoryManager::freeDevice(d_csr);
    }

    int* getOffsetsPtr() const { return d_offsets; }
    int* getIndicesPtr() const { return d_indices; }
    size_t getNumVertices() const { return num_vertices; }
    size_t getNumEdges() const { return num_edges; }
};

// CUDA wrapper for CSC format
class CudaCSC {
private:
    CSC* d_csc;
    int* d_offsets;
    int* d_indices;
    float* d_values;
    size_t num_vertices;
    size_t num_edges;

public:
    CudaCSC(const CSC& csc) {
        const auto& graph = csc.GetGraph();
        num_vertices = graph.offsets.size() - 1;
        num_edges = graph.indices.size();

        // Allocate device memory
        d_offsets = CudaMemoryManager::allocateDevice<int>(graph.offsets.size());
        d_indices = CudaMemoryManager::allocateDevice<int>(graph.indices.size());
        
        if (graph.weighted) {
            d_values = CudaMemoryManager::allocateDevice<float>(graph.values.size());
        }

        // Copy data to device
        CudaMemoryManager::copyToDevice(d_offsets, graph.offsets.data(), graph.offsets.size());
        CudaMemoryManager::copyToDevice(d_indices, graph.indices.data(), graph.indices.size());
        
        if (graph.weighted) {
            CudaMemoryManager::copyToDevice(d_values, graph.values.data(), graph.values.size());
        }

        // Create device CSC structure
        d_csc = CudaMemoryManager::allocateDevice<CSC>(1);
        CSC h_csc_temp = csc; // Copy constructor
        h_csc_temp.g.offsets.clear();
        h_csc_temp.g.indices.clear();
        h_csc_temp.g.values.clear();
        CudaMemoryManager::copyToDevice(d_csc, &h_csc_temp, 1);
    }

    ~CudaCSC() {
        if (d_offsets) CudaMemoryManager::freeDevice(d_offsets);
        if (d_indices) CudaMemoryManager::freeDevice(d_indices);
        if (d_values) CudaMemoryManager::freeDevice(d_values);
        if (d_csc) CudaMemoryManager::freeDevice(d_csc);
    }

    int* getOffsetsPtr() const { return d_offsets; }
    int* getIndicesPtr() const { return d_indices; }
    size_t getNumVertices() const { return num_vertices; }
    size_t getNumEdges() const { return num_edges; }
};

// CUDA wrapper for COO format
class CudaCOO {
private:
    COO* d_coo;
    int* d_offsets;
    int* d_indices;
    float* d_values;
    size_t num_vertices;
    size_t num_edges;

public:
    CudaCOO(const COO& coo) {
        const auto& graph = coo.GetGraph();
        num_vertices = coo.GetGraph().offsets.size();
        num_edges = graph.indices.size();

        // Allocate device memory
        d_offsets = CudaMemoryManager::allocateDevice<int>(graph.offsets.size());
        d_indices = CudaMemoryManager::allocateDevice<int>(graph.indices.size());
        
        if (graph.weighted) {
            d_values = CudaMemoryManager::allocateDevice<float>(graph.values.size());
        }

        // Copy data to device
        CudaMemoryManager::copyToDevice(d_offsets, graph.offsets.data(), graph.offsets.size());
        CudaMemoryManager::copyToDevice(d_indices, graph.indices.data(), graph.indices.size());
        
        if (graph.weighted) {
            CudaMemoryManager::copyToDevice(d_values, graph.values.data(), graph.values.size());
        }

        // Create device COO structure
        d_coo = CudaMemoryManager::allocateDevice<COO>(1);
        COO h_coo_temp = coo; // Copy constructor
        h_coo_temp.g.offsets.clear();
        h_coo_temp.g.indices.clear();
        h_coo_temp.g.values.clear();
        CudaMemoryManager::copyToDevice(d_coo, &h_coo_temp, 1);
    }

    ~CudaCOO() {
        if (d_offsets) CudaMemoryManager::freeDevice(d_offsets);
        if (d_indices) CudaMemoryManager::freeDevice(d_indices);
        if (d_values) CudaMemoryManager::freeDevice(d_values);
        if (d_coo) CudaMemoryManager::freeDevice(d_coo);
    }

    int* getOffsetsPtr() const { return d_offsets; }
    int* getIndicesPtr() const { return d_indices; }
    size_t getNumVertices() const { return num_vertices; }
    size_t getNumEdges() const { return num_edges; }
};

// BFS traversal class
class CudaBFS {
private:
    unsigned int* d_level;
    unsigned int* d_vertex_visited;
    size_t num_vertices;

public:
    CudaBFS(size_t vertices) : num_vertices(vertices) {
        d_level = CudaMemoryManager::allocateDevice<unsigned int>(vertices);
        d_vertex_visited = CudaMemoryManager::allocateDevice<unsigned int>(1);
        
        // Initialize level array with INT_MAX
        std::vector<unsigned int> h_level(vertices, UINT_MAX);
        CudaMemoryManager::copyToDevice(d_level, h_level.data(), vertices);
        
        // Initialize vertex_visited
        unsigned int h_vertex_visited = 0;
        CudaMemoryManager::copyToDevice(d_vertex_visited, &h_vertex_visited, 1);
    }

    ~CudaBFS() {
        if (d_level) CudaMemoryManager::freeDevice(d_level);
        if (d_vertex_visited) CudaMemoryManager::freeDevice(d_vertex_visited);
    }

    void setSource(unsigned int source) {
        std::vector<unsigned int> h_level(num_vertices, UINT_MAX);
        h_level[source] = 0;
        CudaMemoryManager::copyToDevice(d_level, h_level.data(), num_vertices);
    }

    std::vector<unsigned int> getLevels() const {
        std::vector<unsigned int> h_level(num_vertices);
        CudaMemoryManager::copyFromDevice(h_level.data(), d_level, num_vertices);
        return h_level;
    }

    // Top-down BFS using CSR
    void topDownBFS(const CudaCSR& csr, unsigned int current_level) {
        int block_size = 256;
        int grid_size = (num_vertices + block_size - 1) / block_size;
        
        craph::top_down_bfs_kernel<<<grid_size, block_size>>>(
            csr.getOffsetsPtr(), csr.getIndicesPtr(), csr.getNumVertices(),
            d_level, d_vertex_visited, current_level
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Bottom-up BFS using CSC
    void bottomUpBFS(const CudaCSC& csc, unsigned int current_level) {
        int block_size = 256;
        int grid_size = (num_vertices + block_size - 1) / block_size;
        
        craph::bottom_up_bfs_kernel<<<grid_size, block_size>>>(
            csc.getOffsetsPtr(), csc.getIndicesPtr(), csc.getNumVertices(),
            d_level, d_vertex_visited, current_level
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Edge-centric BFS using COO
    void edgeCentricBFS(const CudaCOO& coo, unsigned int current_level) {
        int block_size = 256;
        int grid_size = (coo.getNumEdges() + block_size - 1) / block_size;
        
        craph::edge_centric_bfs<<<grid_size, block_size>>>(
            coo.getOffsetsPtr(), coo.getIndicesPtr(), coo.getNumEdges(),
            d_level, d_vertex_visited, current_level
        );
        CUDA_CHECK(cudaGetLastError());
    }
};

} // namespace craph

#endif // CRAPH_CUDA_HPP 