#ifndef edge_centric_cuh
#define edge_centric_cuh

#include <cuda_runtime.h>
#include <climits>

namespace craph {

// Edge-centric BFS kernel
__global__ void edge_centric_bfs(
    int* d_offsets,
    int* d_indices,
    int num_edges,
    unsigned int* level,
    unsigned int* newVertexVisited, 
    unsigned int currLevel
);

} // namespace craph

#endif // edge_centric_cuh
