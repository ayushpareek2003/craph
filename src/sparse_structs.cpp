#include "../inc/sparse_Structs.hpp"


namespace craph {

void Print(CSgraph g){
        std::cout << "row_offsets: ";
        for (int r : g.offsets) std::cout << r << " ";
        std::cout << "\ncol_indices: ";
        for (int c : g.indices) std::cout << c << " ";
        if (g.weighted) {
            std::cout << "\nvalues: ";
            for (float w : g.values) std::cout << w << " ";
        }
        std::cout << "\n";
    }


CSR::CSR(const std::string& path_to_file, bool is_weighted)
    : path_to_file(path_to_file), weighted(is_weighted) {
    std::ifstream file(path_to_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path_to_file << "\n";
        return;
    }
    g = migrate(file);
    }


const CSgraph& CSR::GetGraph() const {
    return g;
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

// class CSC{
//     public:
//     CSC(std::string &path_to_file,bool &is_weighted){

//     }

//     private:
//     std::string path_to_file;
//     bool is_weighted;
//     CSgraph g;
// };

}  // namespace craph

int main() {
    std::string file_path = "graph.txt";  // Change to your actual file
    bool is_weighted = false;

    craph::CSR tr(file_path, is_weighted);

    Print(tr.GetGraph());

    return 0;
}
