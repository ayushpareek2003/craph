#include "../inc/sparse_Structs.hpp"
#include <iostream>

int main() {
    std::cout << "Simple Sparse Matrix Test" << std::endl;
    std::cout << "=========================" << std::endl;
    
    std::string file_path = "../src/graph.txt";
    bool is_weighted = true;
    
    std::cout << "\n1. Testing CSR Format:" << std::endl;
    craph::CSR csr(file_path, is_weighted);
    csr.Print();
    
    std::cout << "\n2. Testing CSC Format:" << std::endl;
    craph::CSC csc(file_path, is_weighted);
    csc.Print();
    
    std::cout << "\n3. Testing COO Format:" << std::endl;
    craph::COO coo(file_path, is_weighted);
    coo.Print();
    
    std::cout << "\n4. Testing Inheritance (polymorphism):" << std::endl;
    craph::SparseMatrix* matrix = new craph::CSR(file_path, is_weighted);
    matrix->Print();
    delete matrix;
    
    std::cout << "\n✅ All tests completed!" << std::endl;
    return 0;
} 