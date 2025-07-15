#include "../inc/kernels/edge_centric.cuh"

__global__ void Edge_centric_kernels::edge_centric_bfs(COO* graph,unsigned int* level,unsigned int* newVertexVisited, unsigned int currLevel){
    unsigned int edge=blockIdx.x*blockDim.x + threadIdx.x;
    if(edge< graph.numEdges){
        unsigned int previous=graph.offsets[edge];
        if(level[previous]==currLevel){
            unsigned int neighbor=graph.indices[edge];
            if(level[neighbor]==UINT_MAX){
                level[neighbor]=currLevel;
                *newVertexVisited=1;
            }
        }
    }

}
