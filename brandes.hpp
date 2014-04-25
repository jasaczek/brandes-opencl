#ifndef BRANDES_H_
#define BRANDES_H_

#include <CL/cl.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>

cl_uint *input;
cl_uint *output;
cl_uint multiplier;

cl_uint width;

cl_mem   inputBuffer;
cl_mem	 outputBuffer;

cl_context          context;
cl_device_id        *devices;
cl_command_queue    commandQueue;

cl_program program;

/* This program uses only one kernel and this serves as a handle to it */
cl_kernel  kernel;

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
int runCLKernels(void);

/* Releases OpenCL resources (Context, Memory etc.) */
int cleanupCL(void);

/* Releases program's resources */
void cleanupHost(void);

/*
 * Prints no more than 256 elements of the given array.
 * Prints full array if length is less than 256.
 *
 * Prints Array name followed by elements.
 */
void print1DArray(
		 const std::string arrayName, 
         const unsigned int * arrayData, 
         const unsigned int length);


#endif  /* #ifndef BRANDES_H_ */
