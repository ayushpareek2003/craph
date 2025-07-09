#include "../inc/sparse_Structs.hpp"
namespace craph {

void SparseMatrix::Print() const {
    std::cout << "Offsets: ";
    for (int r : g.offsets) std::cout << r << " ";
    std::cout << "\nIndices: ";
    for (int c : g.indices) std::cout << c << " ";
    if (g.weighted) {
        std::cout << "\nValues: ";
        for (float w : g.values) std::cout << w << " ";
    }
    std::cout << "\n";
}

// Base class constructor - doesn't call virtual functions
SparseMatrix::SparseMatrix(const std::string& path_to_file, bool is_weighted)
    : path_to_file(path_to_file), weighted(is_weighted) {
}

const CSgraph& SparseMatrix::GetGraph() const {
    return g;
}

CSR::CSR(const std::string& path_to_file, bool is_weighted)
    : SparseMatrix(path_to_file, is_weighted) {
    std::ifstream file(path_to_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path_to_file << "\n";
        return;
    }
    g = migrate(file);
}
    
CSgraph CSR::migrate(std::ifstream& file) {
    std::string line;
    std::vector<int> out_degrees;
    int max_node = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int src, dst;
        float weight;

        iss >> src >> dst;
        max_node = std::max(max_node, std::max(src, dst)); 
        if (out_degrees.size() <= src)
            out_degrees.resize(src + 1, 0);
        ++out_degrees[src];
    }

    std::vector<int> row_offsets(max_node + 2, 0);
    for (int i = 0; i <= max_node; ++i)
        row_offsets[i + 1] = row_offsets[i] + (i < out_degrees.size() ? out_degrees[i] : 0);

    std::fill(out_degrees.begin(), out_degrees.end(), 0);
    std::vector<int> col_indices(row_offsets[max_node + 1]);
    std::vector<float> values(weighted ? row_offsets[max_node + 1] : 0);

    file.clear();
    file.seekg(0);

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int src, dst;
        float weight = 1.0f;
        if (weighted) {
            iss >> src >> dst >> weight;
        } else {
            iss >> src >> dst;
        }

        int pos = row_offsets[src] + out_degrees[src]++;
        col_indices[pos] = dst;
        if (weighted) {
            values[pos] = weight;
        }
    }

    return {row_offsets, col_indices, values, weighted};
}

CSC::CSC(const std::string& path_to_file, bool is_weighted)
    : SparseMatrix(path_to_file, is_weighted) {
    std::ifstream file(path_to_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path_to_file << "\n";
        return;
    }
    g = migrate(file);
}

CSgraph CSC::migrate(std::ifstream &file){
    std::string line;
    std::vector<int> out_degrees;

    int max_node=0;
    int sparseCount=0;
    while(std::getline(file,line)){
        if(line.empty()){
            continue;
        }
        std::istringstream iss(line);
        int src,dst;
        float weight;
        iss>>src>>dst;
        max_node=std::max(max_node,std::max(src,dst));
        sparseCount++;
        if(out_degrees.size()<=dst){
            out_degrees.resize(dst+1,0);
        }
        out_degrees[dst]++;
    }
    std::vector<int> inScan(max_node+1,0);
    std::vector<int> col_offsets(max_node+2,0);
    std::vector<int> row_indices(sparseCount,0);
    std::vector<float> values(sparseCount,0);
    for(int i=1;i<=max_node;i++){
        inScan[i]=inScan[i-1]+out_degrees[i-1];
        col_offsets[i+1]=col_offsets[i]+(i < out_degrees.size() ? out_degrees[i] : 0);
    }
    file.clear();
    file.seekg(0);

    while(std::getline(file,line)){
        if(line.empty()){
            continue;
        }
        std::istringstream iss(line);
        int src,dst;
        float weight;
        if(weighted){
            iss>>src>>dst>>weight;
        }
        else{
            iss>>src>>dst;
        }
        
        row_indices[inScan[dst]]=src;
        if(weighted){
            values[inScan[dst]]=weight;
        }
        inScan[dst]++;
    }
    return {col_offsets,row_indices,values,weighted};
}

COO::COO(const std::string& path_to_file, bool is_weighted)
    : SparseMatrix(path_to_file, is_weighted) {
    std::ifstream file(path_to_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path_to_file << "\n";
        return;
    }
    g = migrate(file);
}

CSgraph COO::migrate(std::ifstream& file) {
    std::string line;
    std::vector<std::pair<int, int>> edges;
    std::vector<float> weights;
    int max_node = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int src, dst;
        float weight = 1.0f;
        
        if (weighted) {
            iss >> src >> dst >> weight;
        } else {
            iss >> src >> dst;
        }
        
        max_node = std::max(max_node, std::max(src, dst));
        edges.push_back({src, dst});
        if (weighted) {
            weights.push_back(weight);
        }
    }

    std::vector<int> row_indices, col_indices;
    std::vector<float> values;
    
    for (const auto& edge : edges) {
        row_indices.push_back(edge.first);
        col_indices.push_back(edge.second);
    }
    
    if (weighted) {
        values = weights;
    }

    return {row_indices, col_indices, values, weighted};
}

}  // namespace craph

int main() {
    std::string file_path = "src/graph.txt";  // Updated path
    bool is_weighted = true;

    craph::CSC tr(file_path, is_weighted);

    tr.Print(); // Now using the inherited Print method

    return 0;
}
