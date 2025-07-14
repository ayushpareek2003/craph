#ifndef edge_centric_cuh
#define edge_centric_cuh

#include <cuda_runtime.h>
#include <../sparse_Structs.hpp>


class Edge_centric_kernels{

    __global__ static void edge_centric_bfs(COO* graph,unsigned int* level,unsigned int* newVertexVisited, unsigned int currLevel);
    
    
};
