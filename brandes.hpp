#ifndef BRANDES_H_
#define BRANDES_H_

#include <CL/cl.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>

// Problem variables.
size_t real_vertex_num;
size_t vertex_num;
size_t edges_num;
cl_ulong kernel_execution_time;
cl_ulong memory_transfer_time;
cl_uint* ptrs_arr;
cl_uint* adjs_arr;
cl_float* bc_arr;

// Buffers
cl_mem   ptrs_arr_buffer;
cl_mem   adjs_arr_buffer;
cl_mem	 prec_arr_buffer;
cl_mem	 sigma_arr_buffer;
cl_mem	 dist_buffer;
cl_mem	 delta_arr_buffer;
cl_mem	 cont_buffer;

// Device info.
cl_uint maxDims;
size_t maxWorkGroupSize;
size_t maxWorkItemSizes[3];
cl_uint addressBits;


cl_context          context;
cl_device_id        *devices;
cl_command_queue    commandQueue;

cl_program program;

// Forward and backward kernel of Brandes algorithm.
cl_kernel  kernelForward;
cl_kernel  kernelBackward;

int initializeCL(void);

std::string convertToString(const char * filename);

/*
 * This is called once the OpenCL context, memory etc. are set up,
 * the program is loaded into memory and the kernel handles are ready.
 * 
 * It sets the values for kernels' arguments and enqueues calls to the kernels
 * on to the command queue and waits till the calls have finished execution.
 *
 * It also gets kernel start and end time if profiling is enabled.
 */
int runBFS(void);

/* Releases OpenCL resources (Context, Memory etc.) */
int cleanupCL(void);

/* Releases program's resources */
void cleanupHost(void);


#endif  /* #ifndef BRANDES_H_ */
