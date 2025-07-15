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

__global__ void VertexCentric_kernel::bottom_up_Frontier_kernel(
    CSR* d_csr,
    unsigned int *level,
    unsigned int current_level,
    unsigned int* prevFrontier,
    unsigned int* __lenprevFrontier,
    unsigned int* currFrontier,
    unsigned int* __lencurrFrontier
){
    __shared__ unsigned int currFrontier s[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    
    if(threadIdx.x==0){
        numCurrFrontier_s=0;
    }

    __syncthreads();

    unsigned int idx=blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < __lenprevFrontier){
        unsigned int vertex=prevFrontier[idx];
        for(int edge=d_csr.offsets[vertex];edge<d_csr.offsets[vertex+1];edge++){
            unsigned int dst=d_csr.indices[edge];
            if(atomicCAS(&level[dst],UINT_MAX,current_level)==UINT_MAX){
                unsigned int currFrontierIdx_s=atomicadd(&numCurrFrontier_s,1);
                
                if(currFrontierIdx_s< LOCAL_FRONTIER_CAPACITY){
                    currFrontier_s[currFrontierIdx_s]=dst;
                }
                else{
                    numCurrFrontier_s=LOCAL_FRONTIER_CAPACITY;
                    unsigned int IdxUpdate=atomicadd(&__lencurrFrontier,1);
                    currFrontier[IdxUpdate]=dst;
                }
            }
        }
    }

    __syncthreads();
    unsigned int currStartIdx;
    if(threadIdx.x==0){
        currStartIdx= atomicadd(__lencurrFrontier,numCurrFrontier_s);
    }
    __syncthreads();
    for(unsigned int fillerIdx=threadIdx.x; fillerIdx < numCurrFrontier_S ; fillerIdx+=blockDim.x){
        unsigned int fillerIdxTemp= currStartIdx + fillerIdx;
        currFrontier[fillerIdxTemp]=currFrontier_s[fillerIdx];
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
    // intution is for the intial cases we uses top down and later bottom up
}


