#include "../inc/kernels/vertex_centric.cuh"

__global__ void VertexCentric_kernel::top_down_bfs_kernel(
    CSR *d_csr,
    unsigned int *level,
    unsigned int* vertex_visited,
    unsigned int current_level
) {
    unsigned int vertex_id=blockIdx.x * blockDim.x + threadIdx.x;
    if(vertex_id < d_csr.numVertices){
        if(level[vertex_id]!=current_level-1){
            for(int vertex=d_csr.g.offsets[vertex_id];vertex<d_csr.g.offsets[vertex_id+1];vertex++){
                unsigned int next_Vertex=d_csr.g.indices[vertex];
                if(level[next_Vertex]!=INT_MAX){
                    level[next_Vertex]=current_level;
                    *vertex_visited=1;
                }
            }
        }

    }
}

__global__ void VertexCentric_kernel::bottom_up_bfs_kernel(
    CSC *d_csc,
    unsigned int *level,
    unsigned int* vertex_visited,
    unsigned int current_level
){
   unsigned int vertex_id=blockIdx.x*blockDim.x + threadIdx.x;
   if(vertex_id < d_csc.numVertices){
        if(level[vertex_id]==INT_MAX){
            for(int vertex=d_csc.g.offsets[vertex_id];vertex<d_csc.g.offsets[vertex_id];vertex++){
                unsigned int next_vertex=d_csc.g.indices[vertex];
                if(level[next_vertex]==current_level-1){
                    level[vertex_id]=current_level;
                    *vertex_visited=1;
                    break;
                }
            }
        }

   }

}

__global__ static void optimised_approach( 
            CSC *d_csc,
            CSR *d_csr,
            unsigned int *level,
            unsigned int* vertex_visited,
            unsigned int current_level
){
    //implement soon
}


