#include "../inc/kernels/edge_centric.cuh"

namespace craph {

__global__ void edge_centric_bfs(
    int* d_offsets,
    int* d_indices,
    int num_edges,
    unsigned int* level,
    unsigned int* newVertexVisited, 
    unsigned int currLevel
) {
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < num_edges) {
        unsigned int previous = d_offsets[edge];
        if (level[previous] == currLevel) {
            unsigned int neighbor = d_indices[edge];
            if (level[neighbor] == UINT_MAX) {
                level[neighbor] = currLevel;
                *newVertexVisited = 1;
            }
        }
    }
}

} // namespace craph
