#ifndef vertex_centric_cuh
#define vertex_centric_cuh

#include <cuda_runtime.h>
#include <../sparse_Structs.hpp>

class VertexCentric_kernel{
    public:
        __global__ static void top_down_bfs_kernel( //Also known as  push style BFS
            CSR *d_csr,
            unsigned int *level,
            unsigned int* vertex_visited,
            unsigned int current_level,
        );

        __global__ static void top_down_frontiers_kernel(
             CSR *d_csr,
            unsigned int *level,
            unsigned int current_level,
            unsigned int* prevFrontier,
            unsigned int* __lenprevFrontier,
            unsigned int* currFrontier,
            unsigned int* __lencurrFrontier
        )

        __global__ static void bottom_up_bfs_kernel( //Also known as pull style BFS
            CSC *d_csc,
            unsigned int *level,
            unsigned int* vertex_visited,
            unsigned int current_level,
        );
        
        __global__ static void optimised_approach( // Optimised approach, intially top_down, then bottom_up
            CSC *d_csc,
            CSR *d_csr,
            unsigned int *level,
            unsigned int* vertex_visited,
            unsigned int current_level,
        );
};

#endif // vertex_centric_cuh