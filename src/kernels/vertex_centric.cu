#include "../inc/kernels/vertex_centric.cuh"

namespace craph {

__global__ void top_down_bfs_kernel(
    int* d_offsets,
    int* d_indices,
    int num_vertices,
    unsigned int *level,
    unsigned int* vertex_visited,
    unsigned int current_level
) {
    unsigned int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id < num_vertices) {
        if (level[vertex_id] != current_level - 1) {
            for (int vertex = d_offsets[vertex_id]; vertex < d_offsets[vertex_id + 1]; vertex++) {
                unsigned int next_Vertex = d_indices[vertex];
                if (level[next_Vertex] != INT_MAX) {
                    level[next_Vertex] = current_level;
                    *vertex_visited = 1;
                }
            }
        }
    }
}

__global__ void top_down_frontiers_kernel(
    int* d_offsets,
    int* d_indices,
    unsigned int *level,
    unsigned int current_level,
    unsigned int* prevFrontier,
    unsigned int* __lenprevFrontier,
    unsigned int* currFrontier,
    unsigned int* __lencurrFrontier
) {
    __shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    
    if (threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }

    __syncthreads();

    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < *__lenprevFrontier) {
        unsigned int vertex = prevFrontier[idx];
        for (int edge = d_offsets[vertex]; edge < d_offsets[vertex + 1]; edge++) {
            unsigned int dst = d_indices[edge];
            if (atomicCAS(&level[dst], UINT_MAX, current_level) == UINT_MAX) {
                unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
                
                if (currFrontierIdx_s < LOCAL_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = dst;
                } else {
                    numCurrFrontier_s = LOCAL_FRONTIER_CAPACITY;
                    unsigned int IdxUpdate = atomicAdd(__lencurrFrontier, 1);
                    currFrontier[IdxUpdate] = dst;
                }
            }
        }
    }

    __syncthreads();
    unsigned int currStartIdx;
    if (threadIdx.x == 0) {
        currStartIdx = atomicAdd(__lencurrFrontier, numCurrFrontier_s);
    }
    __syncthreads();
    for (unsigned int fillerIdx = threadIdx.x; fillerIdx < numCurrFrontier_s; fillerIdx += blockDim.x) {
        unsigned int fillerIdxTemp = currStartIdx + fillerIdx;
        currFrontier[fillerIdxTemp] = currFrontier_s[fillerIdx];
    }
}

__global__ void bottom_up_bfs_kernel(
    int* d_offsets,
    int* d_indices,
    int num_vertices,
    unsigned int *level,
    unsigned int* vertex_visited,
    unsigned int current_level
) {
   unsigned int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
   if (vertex_id < num_vertices) {
        if (level[vertex_id] == INT_MAX) {
            for (int vertex = d_offsets[vertex_id]; vertex < d_offsets[vertex_id + 1]; vertex++) {
                unsigned int next_vertex = d_indices[vertex];
                if (level[next_vertex] == current_level - 1) {
                    level[vertex_id] = current_level;
                    *vertex_visited = 1;
                    break;
                }
            }
        }
   }
}

__global__ void optimised_approach( 
    int* d_csc_offsets,
    int* d_csc_indices,
    int* d_csr_offsets,
    int* d_csr_indices,
    int num_vertices,
    unsigned int *level,
    unsigned int* vertex_visited,
    unsigned int current_level
) {
    // Implementation coming soon
    // Intuition is for the initial cases we use top down and later bottom up
}

} // namespace craph


