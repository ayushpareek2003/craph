#ifndef vertex_centric_cuh
#define vertex_centric_cuh

#include <cuda_runtime.h>
#include <climits>

#define LOCAL_FRONTIER_CAPACITY 32

namespace craph {

// Top-down BFS kernel (push style)
__global__ void top_down_bfs_kernel(
    int* d_offsets,
    int* d_indices,
    int num_vertices,
    unsigned int *level,
    unsigned int* vertex_visited,
    unsigned int current_level
);

// Top-down frontiers kernel
__global__ void top_down_frontiers_kernel(
    int* d_offsets,
    int* d_indices,
    unsigned int *level,
    unsigned int current_level,
    unsigned int* prevFrontier,
    unsigned int* __lenprevFrontier,
    unsigned int* currFrontier,
    unsigned int* __lencurrFrontier
);

// Bottom-up BFS kernel (pull style)
__global__ void bottom_up_bfs_kernel(
    int* d_offsets,
    int* d_indices,
    int num_vertices,
    unsigned int *level,
    unsigned int* vertex_visited,
    unsigned int current_level
);

// Optimized approach kernel
__global__ void optimised_approach(
    int* d_csc_offsets,
    int* d_csc_indices,
    int* d_csr_offsets,
    int* d_csr_indices,
    int num_vertices,
    unsigned int *level,
    unsigned int* vertex_visited,
    unsigned int current_level
);

} // namespace craph

#endif // vertex_centric_cuh