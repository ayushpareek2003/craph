#ifndef SPARSE_HPP
#define SPARSE_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <set>

namespace craph {

//struct to store SPARSE format 
struct CSgraph{
    std::vector<int> offsets;
    std::vector<int> indices;
    std::vector<float> values;  
    bool weighted = false;
};

// Base class for all sparse matrix formats
class SparseMatrix {
public:
    SparseMatrix(const std::string& path_to_file, bool is_weighted = false);
    virtual ~SparseMatrix() = default;
    
    const CSgraph& GetGraph() const;
    void Print() const; // Common print function for all formats

protected:
    std::string path_to_file;
    bool weighted;
    CSgraph g;
    int numVertices;
    int numEdges;
    std::set<int> mapping;
    virtual CSgraph migrate(std::ifstream& file) = 0; // Pure virtual function
};

// CSR (Compressed Sparse Row) class
class CSR : public SparseMatrix {
public:
    CSR(const std::string& path_to_file, bool is_weighted = false);
    
   
private:
    CSgraph migrate(std::ifstream& file) override;
};

// CSC (Compressed Sparse Column) class 
class CSC : public SparseMatrix {
public:
    CSC(const std::string& path_to_file, bool is_weighted = false);
    
private:
    CSgraph migrate(std::ifstream& file) override;
};

class COO : public SparseMatrix {
public:
    COO(const std::string& path_to_file, bool is_weighted = false);
    
private:
    CSgraph migrate(std::ifstream& file) override;   
};

}  // namespace craph

#endif // SPARSE_HPP
