//#define __global __attribute__((address_space(1)))

__kernel void brandesKernelForward(
        __global  unsigned int* ptrs_arr,
        __global  unsigned int* adjs_arr,
        __global  char* prec_arr,
        __global  unsigned int* sigma_arr,
        __global int* dist,
        __global char* cont,
        const unsigned int level) {
    unsigned int u = get_global_id(0);
    unsigned int ptr, v;
    if (dist[u] == level) {
        for (ptr = ptrs_arr[u]; ptr < ptrs_arr[u + 1]; ++ptr) {
            v = adjs_arr[ptr];
            if (dist[v] == -1) {
                dist[v] = level + 1;
                *cont = 1;
            }
            if (dist[v] == level + 1) {
                prec_arr[ptr] = 1;
                atomic_add(&sigma_arr[v], sigma_arr[u]);
            }
        }
    }
}

__kernel void brandesKernelBackward(
        __global  unsigned int* ptrs_arr,
        __global  unsigned int* adjs_arr,
        __global  char* prec_arr,
        __global int* dist,
        __global float* delta_arr,
        const unsigned int level) {
    unsigned int u = get_global_id(0);
    unsigned int ptr, v;
    if (dist[u] == level) {
        for (ptr = ptrs_arr[u]; ptr < ptrs_arr[u + 1]; ++ptr) {
            v = adjs_arr[ptr];
            if (prec_arr[ptr] == 1) {
                delta_arr[u] = delta_arr[u] + delta_arr[v];
            }
        }
    }
}
