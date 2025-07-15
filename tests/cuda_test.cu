#include "../inc/craph_cuda.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "CRAPH CUDA Graph Traversal Test" << std::endl;
    std::cout << "===============================" << std::endl;
    
    // Check CUDA availability
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return -1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Set device
    cudaSetDevice(0);
    
    std::string file_path = "../graph.txt";
    bool is_weighted = true;
    
    try {
        // Load graph in different formats
        std::cout << "\n1. Loading graph in CSR format..." << std::endl;
        craph::CSR csr(file_path, is_weighted);
        std::cout << "CSR loaded successfully. Vertices: " << csr.GetGraph().offsets.size() - 1 
                  << ", Edges: " << csr.GetGraph().indices.size() << std::endl;
        
        std::cout << "\n2. Loading graph in CSC format..." << std::endl;
        craph::CSC csc(file_path, is_weighted);
        std::cout << "CSC loaded successfully. Vertices: " << csc.GetGraph().offsets.size() - 1 
                  << ", Edges: " << csc.GetGraph().indices.size() << std::endl;
        
        std::cout << "\n3. Loading graph in COO format..." << std::endl;
        craph::COO coo(file_path, is_weighted);
        std::cout << "COO loaded successfully. Vertices: " << coo.GetGraph().offsets.size() 
                  << ", Edges: " << coo.GetGraph().indices.size() << std::endl;
        
        // Create CUDA wrappers
        std::cout << "\n4. Creating CUDA wrappers..." << std::endl;
        craph::CudaCSR cuda_csr(csr);
        craph::CudaCSC cuda_csc(csc);
        craph::CudaCOO cuda_coo(coo);
        std::cout << "CUDA wrappers created successfully" << std::endl;
        
        // Initialize BFS
        size_t num_vertices = csr.GetGraph().offsets.size() - 1;
        craph::CudaBFS bfs(num_vertices);
        
        // Set source vertex
        unsigned int source = 0;
        bfs.setSource(source);
        std::cout << "\n5. Starting BFS from vertex " << source << std::endl;
        
        // Test top-down BFS
        std::cout << "\n6. Testing Top-Down BFS (CSR)..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (unsigned int level = 1; level <= 3; ++level) {
            bfs.topDownBFS(cuda_csr, level);
            std::vector<unsigned int> levels = bfs.getLevels();
            
            std::cout << "Level " << level << " vertices: ";
            for (size_t i = 0; i < num_vertices; ++i) {
                if (levels[i] == level) {
                    std::cout << i << " ";
                }
            }
            std::cout << std::endl;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Top-Down BFS completed in " << duration.count() << " microseconds" << std::endl;
        
        // Test bottom-up BFS
        std::cout << "\n7. Testing Bottom-Up BFS (CSC)..." << std::endl;
        bfs.setSource(source); // Reset
        start = std::chrono::high_resolution_clock::now();
        
        for (unsigned int level = 1; level <= 3; ++level) {
            bfs.bottomUpBFS(cuda_csc, level);
            std::vector<unsigned int> levels = bfs.getLevels();
            
            std::cout << "Level " << level << " vertices: ";
            for (size_t i = 0; i < num_vertices; ++i) {
                if (levels[i] == level) {
                    std::cout << i << " ";
                }
            }
            std::cout << std::endl;
        }
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Bottom-Up BFS completed in " << duration.count() << " microseconds" << std::endl;
        
        // Test edge-centric BFS
        std::cout << "\n8. Testing Edge-Centric BFS (COO)..." << std::endl;
        bfs.setSource(source); // Reset
        start = std::chrono::high_resolution_clock::now();
        
        for (unsigned int level = 1; level <= 3; ++level) {
            bfs.edgeCentricBFS(cuda_coo, level);
            std::vector<unsigned int> levels = bfs.getLevels();
            
            std::cout << "Level " << level << " vertices: ";
            for (size_t i = 0; i < num_vertices; ++i) {
                if (levels[i] == level) {
                    std::cout << i << " ";
                }
            }
            std::cout << std::endl;
        }
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Edge-Centric BFS completed in " << duration.count() << " microseconds" << std::endl;
        
        std::cout << "\nAll CUDA tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 