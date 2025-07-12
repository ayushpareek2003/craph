#include "../inc/sparse_Structs.hpp"
#include <iostream>
#include <cassert>

namespace craph {

void test_csr_format() {
    std::cout << "\n=== Testing CSR Format ===" << std::endl;
    
    // Test weighted CSR
    std::cout << "Testing weighted CSR:" << std::endl;
    CSR csr_weighted("graph.txt", true);
    csr_weighted.Print();
    
    // Test unweighted CSR
    std::cout << "\nTesting unweighted CSR:" << std::endl;
    CSR csr_unweighted("graph.txt", false);
    csr_unweighted.Print();

    // Verify graph data
    const CSgraph& g = csr_weighted.GetGraph();
    assert(g.weighted == true);
    assert(!g.offsets.empty());
    assert(!g.indices.empty());
    std::cout << " CSR tests passed" << std::endl;
}

void test_csc_format() {
    std::cout << "\n=== Testing CSC Format ===" << std::endl;
    
    // Test weighted CSC
    std::cout << "Testing weighted CSC:" << std::endl;
    CSC csc_weighted("graph.txt", true);
    csc_weighted.Print();
    
    // Test unweighted CSC
    std::cout << "\nTesting unweighted CSC:" << std::endl;
    CSC csc_unweighted("graph.txt", false);
    csc_unweighted.Print();
    
    // Verify graph data
    const CSgraph& g = csc_weighted.GetGraph();
    assert(g.weighted == true);
    assert(!g.offsets.empty());
    assert(!g.indices.empty());
    std::cout << " CSC tests passed" << std::endl;
}

void test_coo_format() {
    std::cout << "\n=== Testing COO Format ===" << std::endl;
    
    // Test weighted COO
    std::cout << "Testing weighted COO:" << std::endl;
    COO coo_weighted("graph.txt", true);
    coo_weighted.Print();
    
    // Test unweighted COO
    std::cout << "\nTesting unweighted COO:" << std::endl;
    COO coo_unweighted("graph.txt", false);
    coo_unweighted.Print();
    
    // Verify graph data
    const CSgraph& g = coo_weighted.GetGraph();
    assert(g.weighted == true);
    assert(!g.offsets.empty());
    assert(!g.indices.empty());
    std::cout << " COO tests passed" << std::endl;
}

void test_inheritance() {
    std::cout << "\n=== Testing Inheritance ===" << std::endl;
    
    // Test polymorphism
    SparseMatrix* matrices[3];
    matrices[0] = new CSR("graph.txt", true);
    matrices[1] = new CSC("graph.txt", true);
    matrices[2] = new COO("graph.txt", true);
    
    std::cout << "Testing polymorphic Print() calls:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "\nMatrix " << (i+1) << ":" << std::endl;
        matrices[i]->Print();
    }
    
    // Clean up
    for (int i = 0; i < 3; i++) {
        delete matrices[i];
    }
    
    std::cout << " Inheritance tests passed" << std::endl;
}

//        i dont think so we need this as for now

// void test_error_handling() {
//     std::cout << "\n=== Testing Error Handling ===" << std::endl;
    
//     // Test with non-existent file
//     std::cout << "Testing with non-existent file:" << std::endl;
//     try {
//         CSR csr_error("graph.txt", true);
//         std::cout << "Warning: Should have failed to open file" << std::endl;
//     } catch (...) {
//         std::cout << " Error handling works" << std::endl;
//     }
    
//     std::cout << " Error handling tests passed" << std::endl;
// }

}
 // namespace craph

int main() {
    std::cout << "Starting Sparse Matrix Format Tests..." << std::endl;
    
    try {
        craph::test_csr_format();
        craph::test_csc_format();
        craph::test_coo_format();
        craph::test_inheritance();
        // craph::test_error_handling();
        
        std::cout << "All tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
} 