#ifndef SPARSE_HPP
#define SPARSE_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

namespace craph {

// Utility print function for global graph `g`


//struct to store SPARSE format 
struct CSgraph{
    std::vector<int> offsets;
    std::vector<int> indices;
    std::vector<float> values;  
    bool weighted = false;
};

void Print(CSgraph &g);

// CSR (Compressed Sparse Row) class
class CSR {
public:
    CSR(const std::string& path_to_file, bool is_weighted = false);
    void Print() const;
    const CSgraph& GetGraph() const;

private:
    std::string path_to_file;
    bool weighted;
    CSgraph g;

    CSgraph migrate(std::ifstream& file);
};

// CSC (Compressed Sparse Column) class - stub
class CSC {
public:
    CSC(std::string& path_to_file, bool& is_weighted);

private:
    std::string path_to_file;
    bool is_weighted;
    CSgraph g;
};

}  // namespace craph

#endif // SPARSE_HPP
