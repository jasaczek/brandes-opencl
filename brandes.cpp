#include "brandes.hpp"

#include <algorithm>
#include <math.h>
#include <vector>
#include "brandes_utils.hpp"

using std::vector;
using std::cout;
using std::endl;;

#define DEBUG
#ifdef DEBUG
#define STATUS_CHK(msg) if(status != CL_SUCCESS) { cout << "STATUS: " << status << ". " << msg; return 1; }
#else
#define STATUS_CHK(msg)
#endif

#define MDEG 4

#define WORK_GROUP_SIZE 128

int initializeHost(const char* inputPath) {
	vector<vector<cl_uint>> edges;
	readGraph(inputPath, edges);

	vertex_num = edges.size();
	real_vertex_num = vertex_num;

	nvir_arr = (cl_uint*) malloc(sizeof(cl_uint) * vertex_num);
	virtual_vertex_num = 0;
	edges_num = 0;
	for(size_t iter = 0; iter < vertex_num; ++iter) {
		nvir_arr[iter] = (edges[iter].size() / MDEG) + ((edges[iter].size() % MDEG) > 0);
		virtual_vertex_num += nvir_arr[iter];
		edges_num += edges[iter].size();
	}

	offset_arr = 	(cl_uint*) 	malloc(sizeof(cl_uint) 	* virtual_vertex_num);
	vmap_arr = 		(cl_uint*) 	malloc(sizeof(cl_uint) 	* virtual_vertex_num);
	ptrs_arr = 		(cl_uint*) 	malloc(sizeof(cl_uint) 	* (vertex_num + 1));
	adjs_arr = 		(cl_uint*) 	malloc(sizeof(cl_uint) 	* edges_num);
	bc_arr = 		(cl_float*) malloc(sizeof(cl_float) * vertex_num);

	cl_uint current_offset = 0;
	size_t progress = 0;
	for(size_t i = 0; i < vertex_num; ++i) {
		ptrs_arr[i] = current_offset;
		// Handle error.
		if(adjs_arr + current_offset != memcpy(adjs_arr + current_offset,
				edges[i].data(),
				sizeof(cl_uint) * edges[i].size())) {
			cout << "Error creating CSR representation of graph" << endl;
			return 1;
		}
		current_offset += edges[i].size();

		bc_arr[i] = 0;

		for(size_t j = 0; j < nvir_arr[i]; ++j) {
			offset_arr[progress] = j;
			vmap_arr[progress] = i;
			++progress;
		}
	}
	ptrs_arr[vertex_num] = current_offset;

	return 0;
}

/*
 * Converts the contents of a file into a string
 */
std::string convertToString(const char *filename) {
	size_t size;
	char*  str;
	std::string s;

	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if (f.is_open()) {
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);

		str = new char[size+1];
		if (!str) {
			f.close();
			return NULL;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';

		s = str;
		delete[] str;
		return s;
	}
	else {
		std::cout << "\nFile containg the kernel code(\".cl\") not found. Please copy the required file in the folder containg the executable.\n";
		exit(1);
	}
	return NULL;
}

/*
 * \brief OpenCL related initialization 
 *        Create Context, Device list, Command Queue
 *        Create OpenCL memory buffer objects
 *        Load CL file, compile, link CL source 
 *		  Build program and kernel objects
 */
int initializeCL(void) {
	cl_int status = 0;
	size_t deviceListSize;

	/*
	 * Have a look at the available platforms and pick either
	 * the AMD one if available or a reasonable default.
	 */

	cl_uint numPlatforms;
	cl_platform_id platform = NULL;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	STATUS_CHK("Error: Getting Platforms. (clGetPlatformsIDs)\n");

	if (numPlatforms > 0) {
		cl_platform_id* platforms = new cl_platform_id[numPlatforms];
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		STATUS_CHK("Error: Getting Platform Ids. (clGetPlatformsIDs)\n");
		for (unsigned int i=0; i < numPlatforms; ++i) {
			char pbuff[100];
			status = clGetPlatformInfo(
					platforms[i],
					CL_PLATFORM_VENDOR,
					sizeof(pbuff),
					pbuff,
					NULL);
			STATUS_CHK("Error: Getting Platform Info.(clGetPlatformInfo)\n")
			platform = platforms[i];
			//if (!strcmp(pbuff, "Advanced Micro Devices, Inc."))
			if (!strcmp(pbuff, "NVIDIA Corporation"))
			{
				break;
			}
		}
		delete platforms;
	}

	if (NULL == platform) {
		std::cout << "NULL platform found so Exiting Application." << std::endl;
		return 1;
	}

	/*
	 * If we could find our platform, use it. Otherwise use just available platform.
	 */
	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	/////////////////////////////////////////////////////////////////
	// Create an OpenCL context
	/////////////////////////////////////////////////////////////////
	context = clCreateContextFromType(cps,
			CL_DEVICE_TYPE_GPU,
			NULL,
			NULL,
			&status);
	STATUS_CHK("Error: Creating Context. (clCreateContextFromType)\n");

	/* First, get the size of device list data */
	status = clGetContextInfo(context,
			CL_CONTEXT_DEVICES,
			0,
			NULL,
			&deviceListSize);
	STATUS_CHK("Error: Getting Context Info (device list size, clGetContextInfo)\n");

	/////////////////////////////////////////////////////////////////
	// Detect OpenCL devices
	/////////////////////////////////////////////////////////////////
	devices = (cl_device_id *)malloc(deviceListSize);
	if (devices == 0) {
		std::cout<<"Error: No devices found.\n";
		return 1;
	}

	/* Now, get the device list data */
	status = clGetContextInfo(
			context,
			CL_CONTEXT_DEVICES,
			deviceListSize,
			devices,
			NULL);
	STATUS_CHK("Error: Getting Context Info (device list, clGetContextInfo)\n");

	/////////////////////////////////////////////////////////////////
	// Create an OpenCL command queue
	/////////////////////////////////////////////////////////////////
	commandQueue = clCreateCommandQueue(
			context,
			devices[0],
			CL_QUEUE_PROFILING_ENABLE,
			&status);
	STATUS_CHK("Creating Command Queue. (clCreateCommandQueue)\n");

	/////////////////////////////////////////////////////////////////
	// Create OpenCL memory buffers
	/////////////////////////////////////////////////////////////////
	offset_arr_buffer = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY,
			sizeof(cl_uint) * virtual_vertex_num,
			NULL,
			&status);
	STATUS_CHK("Error: clCreateBuffer (offset_arr_buffer)\n");

	vmap_arr_buffer = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY,
			sizeof(cl_uint) * virtual_vertex_num,
			NULL,
			&status);
	STATUS_CHK("Error: clCreateBuffer (vmap_arr_buffer)\n");

	nvir_arr_buffer = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY,
			sizeof(cl_uint) * vertex_num,
			NULL,
			&status);
	STATUS_CHK("Error: clCreateBuffer (nvir_arr_buffer)\n");

	ptrs_arr_buffer = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY,
			sizeof(cl_uint) * (vertex_num + 1),
			NULL,
			&status);
	STATUS_CHK("Error: clCreateBuffer (ptrs_arr_buffer)\n");

	adjs_arr_buffer = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY,
			sizeof(cl_uint) * edges_num,
			NULL,
			&status);
	STATUS_CHK("Error: clCreateBuffer (adjs_arr_buffer)\n");

	sigma_arr_buffer = clCreateBuffer(
			context,
			CL_MEM_READ_WRITE,
			sizeof(cl_uint) * vertex_num,
			NULL,
			&status);
	STATUS_CHK("Error: clCreateBuffer (sigma_arr_buffer)\n");

	dist_buffer = clCreateBuffer(
			context,
			CL_MEM_READ_WRITE,
			sizeof(cl_uint) * vertex_num,
			NULL,
			&status);
	STATUS_CHK("Error: clCreateBuffer (dist_buffer)\n");

	delta_arr_buffer = clCreateBuffer(
			context,
			CL_MEM_READ_WRITE,
			sizeof(cl_float) * vertex_num,
			NULL,
			&status);
	STATUS_CHK("Error: clCreateBuffer (delta_arr_buffer)\n");

	bc_arr_buffer = clCreateBuffer(
			context,
			CL_MEM_READ_WRITE,
			sizeof(cl_float) * vertex_num,
			NULL,
			&status);
	STATUS_CHK("Error: clCreateBuffer (bc_arr_buffer)\n");

	cont_buffer = clCreateBuffer(
			context,
			CL_MEM_WRITE_ONLY,
			sizeof(cl_char),
			NULL,
			&status);
	STATUS_CHK("Error: clCreateBuffer (cont_buffer)\n");

	/////////////////////////////////////////////////////////////////
	// Load CL file, build CL program object, create CL kernel object
	/////////////////////////////////////////////////////////////////
	const char * filename  = "brandes_kernels.cl";
	std::string  sourceStr = convertToString(filename);
	const char * source    = sourceStr.c_str();
	size_t sourceSize[]    = { strlen(source) };

	program = clCreateProgramWithSource(
			context,
			1,
			&source,
			sourceSize,
			&status);
	STATUS_CHK("Error: Loading Binary into cl_program (clCreateProgramWithBinary)\n");

	/* create a cl program executable for all the devices specified */
	status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
	STATUS_CHK("Error: Building Program (clBuildProgram)\n");

	/* get a kernel object handle for a kernel with the given name */
	kernelForward = clCreateKernel(program, "brandesKernelForward", &status);
	STATUS_CHK("Error: Creating Forward Kernel from program. (clCreateKernel)\n")

	kernelDeltaInit = clCreateKernel(program, "brandesKernelDeltaInit", &status);
	STATUS_CHK("Error: Creating Delta Init Kernel from program. (clCreateKernel)\n");

	kernelBackward = clCreateKernel(program, "brandesKernelBackward", &status);
	STATUS_CHK("Error: Creating Backward Kernel from program. (clCreateKernel)\n");

	kernelBCUpdate = clCreateKernel(program, "brandesKernelBCUpdate", &status);
	STATUS_CHK("Error: Creating BCUpdate from program. (clCreateKernel)\n");

	return 0;
}

int getDeviceInfo(void) {
	cl_int   status;

	/**
	 * Query device capabilities. Maximum
	 * work item dimensions and the maximum
	 * work item sizes
	 */
	status = clGetDeviceInfo(
			devices[0],
			CL_DEVICE_MAX_WORK_GROUP_SIZE,  // maximum number of work items in a work group
			sizeof(size_t),
			(void*)&maxWorkGroupSize,
			NULL);
	STATUS_CHK("Error: Getting Device Info. (clGetDeviceInfo)\n");

	status = clGetDeviceInfo(
			devices[0],
			CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, // maximum number of dimensions
			sizeof(cl_uint),
			(void*)&maxDims,
			NULL);
	STATUS_CHK("Error: Getting Device Info. (clGetDeviceInfo)\n");

	status = clGetDeviceInfo(
			devices[0],
			CL_DEVICE_MAX_WORK_ITEM_SIZES, // maximum number of work items in each dimension of a work group
			sizeof(size_t)*maxDims,
			(void*)maxWorkItemSizes,
			NULL);
	STATUS_CHK("Error: Getting Device Info. (clGetDeviceInfo)\n");;

	status = clGetDeviceInfo(
			devices[0],
			CL_DEVICE_MAX_WORK_ITEM_SIZES, // maximum number of work items in each dimension of a work group
			sizeof(size_t)*maxDims,
			(void*)maxWorkItemSizes,
			NULL);
	STATUS_CHK("Error: Getting Device Info. (clGetDeviceInfo)\n");

	status = clGetDeviceInfo(devices[0],
			CL_DEVICE_ADDRESS_BITS, //maximum number of work items is bounded by 2^CL_DEVICE_ADDRESS_BITS
			sizeof(cl_uint),
			&addressBits,
			NULL);
	STATUS_CHK("Error: Getting Device Info. (clGetDeviceInfo)");

	return 0;
}

/*
 * Set forward kernel buffer arguments.
 */
int setForwardKernelArgs() {
	cl_int   status;

	status = clSetKernelArg(kernelForward, 0, sizeof(cl_mem), (void *)&offset_arr_buffer);
	STATUS_CHK("Error: Setting forward kernel argument. (offset_arr_buffer)\n");

	status = clSetKernelArg(kernelForward, 1, sizeof(cl_mem), (void *)&vmap_arr_buffer);
	STATUS_CHK("Error: Setting forward kernel argument. (vmap_arr_buffer)\n");

	status = clSetKernelArg(kernelForward, 2, sizeof(cl_mem), (void *)&nvir_arr_buffer);
	STATUS_CHK("Error: Setting forward kernel argument. (nvir_arr_buffer)\n");

	status = clSetKernelArg(kernelForward, 3, sizeof(cl_mem), (void *)&ptrs_arr_buffer);
	STATUS_CHK("Error: Setting forward kernel argument. (ptrs_arr_buffer)\n");

	status = clSetKernelArg(kernelForward, 4, sizeof(cl_mem), (void *)&adjs_arr_buffer);
	STATUS_CHK("Error: Setting forward kernel argument. (adjs_arr_buffer)\n");

	status = clSetKernelArg(kernelForward, 5, sizeof(cl_mem), (void *)&sigma_arr_buffer);
	STATUS_CHK("Error: Setting forward kernel argument. (sigma_arr_buffer)\n");

	status = clSetKernelArg(kernelForward, 6, sizeof(cl_mem), (void *)&dist_buffer);
	STATUS_CHK("Error: Setting forward kernel argument. (dist_buffer)\n");

	status = clSetKernelArg(kernelForward, 7, sizeof(cl_mem), (void *)&cont_buffer);
	STATUS_CHK("Error: Setting forward kernel argument. (cont_buffer)\n");

	status = clSetKernelArg(kernelForward, 8, sizeof(cl_uint), (void *)&virtual_vertex_num);
	STATUS_CHK("Error: Setting forward kernel argument. (virtual_vertex_num)\n");
	return 0;
}

/*
 * Set backward kernel buffer arguments.
 */
int setBackwardKernelArgs() {
	cl_int   status;

	status = clSetKernelArg(kernelBackward, 0, sizeof(cl_mem), (void *)&offset_arr_buffer);
	STATUS_CHK("Error: Setting backward kernel argument. (offset_arr_buffer)\n");

	status = clSetKernelArg(kernelBackward, 1, sizeof(cl_mem), (void *)&vmap_arr_buffer);
	STATUS_CHK("Error: Setting backward kernel argument. (vmap_arr_buffer)\n");

	status = clSetKernelArg(kernelBackward, 2, sizeof(cl_mem), (void *)&nvir_arr_buffer);
	STATUS_CHK("Error: Setting backward kernel argument. (nvir_arr_buffer)\n");

	status = clSetKernelArg(kernelBackward, 3, sizeof(cl_mem), (void *)&ptrs_arr_buffer);
	STATUS_CHK("Error: Setting backward kernel argument. (ptrs_arr_buffer)\n");

	status = clSetKernelArg(kernelBackward, 4, sizeof(cl_mem), (void *)&adjs_arr_buffer);
	STATUS_CHK("Error: Setting backward kernel argument. (adjs_arr_buffer)\n");

	status = clSetKernelArg(kernelBackward, 5, sizeof(cl_mem), (void *)&dist_buffer);
	STATUS_CHK("Error: Setting backward kernel argument. (delta_arr_buffer)\n");

	status = clSetKernelArg(kernelBackward, 6, sizeof(cl_mem), (void *)&delta_arr_buffer);
	STATUS_CHK("Error: Setting backward kernel argument. (delta_arr_buffer)\n");

	status = clSetKernelArg(kernelBackward, 7, sizeof(cl_uint), (void *)&virtual_vertex_num);
	STATUS_CHK("Error: Setting backward kernel argument. (virtual_vertex_num)\n");
	return 0;
}

int setDeltaInitKernelArgs() {
	cl_int   status;

	status = clSetKernelArg(kernelDeltaInit, 0, sizeof(cl_mem), (void *)&sigma_arr_buffer);
	STATUS_CHK("Error: Setting delta init kernel argument. (sigma_arr_buffer)\n");

	status = clSetKernelArg(kernelDeltaInit, 1, sizeof(cl_mem), (void *)&delta_arr_buffer);
	STATUS_CHK("Error: Setting delta init kernel argument. (delta_arr_buffer)\n");

	status = clSetKernelArg(kernelDeltaInit, 2, sizeof(cl_uint), (void *)&vertex_num);
	STATUS_CHK("Error: Setting delta init kernel argument. (vertex_num)\n");

	return 0;
}

int setBCUpdateKernelArgs() {
	cl_int   status;

	status = clSetKernelArg(kernelBCUpdate, 0, sizeof(cl_mem), (void *)&sigma_arr_buffer);
	STATUS_CHK("Error: Setting bcupdate kernel argument. (sigma_arr_buffer)\n");

	status = clSetKernelArg(kernelBCUpdate, 1, sizeof(cl_mem), (void *)&delta_arr_buffer);
	STATUS_CHK("Error: Setting bcupdate kernel argument. (delta_arr_buffer)\n");

	status = clSetKernelArg(kernelBCUpdate, 2, sizeof(cl_mem), (void *)&bc_arr_buffer);
	STATUS_CHK("Error: Setting bcupdate kernel argument. (bc_arr_buffer)\n");

	status = clSetKernelArg(kernelBCUpdate, 3, sizeof(cl_uint), (void *)&vertex_num);
	STATUS_CHK("Error: Setting bcupdate kernel argument. (bc_arr_buffer)\n");

	return 0;
}


/*
 * \brief Run OpenCL program
 *
 *        Bind host variables to kernel arguments
 *		  Run the CL kernel
 */
int runBFS(void) {
	cl_int   status;
	cl_event events[3];
	size_t globalThreads[1];
	size_t globalThreadsNormal[1];
	size_t localThreads[1];
	cl_uint s;
	cl_uint level;
	cl_ulong start_time, end_time;
	char cont;

	kernel_execution_time = 0;
	memory_transfer_time = 0;

	// Starting distance array. Set it to -1 everywhere.
	cl_int* dist_arr = (cl_int*) malloc(vertex_num * sizeof(cl_int));
	memset(dist_arr, -1, sizeof(cl_int) * vertex_num);

	cl_float* delta_arr = (cl_float*) malloc(vertex_num * sizeof(cl_float));
	cl_uint* sigma_arr = (cl_uint*) malloc(vertex_num * sizeof(cl_uint));

	if (virtual_vertex_num % WORK_GROUP_SIZE) {
		globalThreads[0] = virtual_vertex_num + WORK_GROUP_SIZE - (virtual_vertex_num % WORK_GROUP_SIZE);
	} else {
		globalThreads[0] = virtual_vertex_num;
	}
	if (vertex_num % WORK_GROUP_SIZE) {
		globalThreadsNormal[0] = vertex_num + WORK_GROUP_SIZE - (vertex_num % WORK_GROUP_SIZE);
	} else {
		globalThreadsNormal[0] = vertex_num;
	}
	localThreads[0]  = WORK_GROUP_SIZE;

	if (localThreads[0] > maxWorkGroupSize ||  // maxWorkGroupSize is the total number of threads in a work group
			localThreads[0] > maxWorkItemSizes[0] // number of threads in each dimension is also limited
	) {
		std::cout<<"Unsupported: Device does not support requested number of work items in a work group."<<std::endl;
		return 1;
	}

	if (setForwardKernelArgs() == 1) {
		return 1;
	}
	if (setDeltaInitKernelArgs() == 1) {
		return 1;
	}
	if (setBackwardKernelArgs() == 1) {
		return 1;
	}
	if (setBCUpdateKernelArgs() == 1) {
		return 1;
	}

	status = clEnqueueWriteBuffer(commandQueue, ptrs_arr_buffer, CL_FALSE, 0, sizeof(cl_uint) * (vertex_num + 1), ptrs_arr, 0, NULL, &events[1]);
	STATUS_CHK("Error: Writing to buffer. (ptrs_arr_buffer)");
	status = clEnqueueWriteBuffer(commandQueue, offset_arr_buffer, CL_FALSE, 0, sizeof(cl_uint) * virtual_vertex_num, offset_arr, 0, NULL, NULL);
	STATUS_CHK("Error: Writing to buffer. (offset_arr_buffer)");
	status = clEnqueueWriteBuffer(commandQueue, bc_arr_buffer, CL_FALSE, 0, sizeof(cl_float) * vertex_num, bc_arr, 0, NULL, NULL);
	STATUS_CHK("Error: Writing to buffer. (offset_arr_buffer)");
	status = clEnqueueWriteBuffer(commandQueue, vmap_arr_buffer, CL_FALSE, 0, sizeof(cl_uint) * virtual_vertex_num, vmap_arr, 0, NULL, NULL);
	STATUS_CHK("Error: Writing to buffer. (vmap_arr_buffer)");
	status = clEnqueueWriteBuffer(commandQueue, nvir_arr_buffer, CL_FALSE, 0, sizeof(cl_uint) * vertex_num, nvir_arr, 0, NULL, NULL);
	STATUS_CHK("Error: Writing to buffer. (nvir_arr_buffer)");
	status = clEnqueueWriteBuffer(commandQueue, adjs_arr_buffer, CL_FALSE, 0, sizeof(cl_uint) * edges_num, adjs_arr, 0, NULL, &events[2]);
	STATUS_CHK("Error: Writing to buffer. (adjs_arr_buffer)");

	status = clWaitForEvents(1, &events[2]);
	STATUS_CHK("Error: Waiting for kernel run to finish. (clWaitForEvents)\n");
	status = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
	STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
	status = clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
	STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
	memory_transfer_time += end_time - start_time;

	for(s = 0; s < vertex_num; ++s) {
		memset(sigma_arr, 0, sizeof(cl_uint) * vertex_num);
		sigma_arr[s] = 1;
		level = 0;
		dist_arr[s] = 0;

		status = clEnqueueWriteBuffer(commandQueue, dist_buffer, CL_FALSE, 0, sizeof(cl_int) * vertex_num, dist_arr, 0, NULL, &events[1]);
		STATUS_CHK("Error: Writing to buffer. (dist_buffer)");
		status = clEnqueueWriteBuffer(commandQueue, sigma_arr_buffer, CL_FALSE, 0, sizeof(cl_uint) * vertex_num, sigma_arr, 0, NULL, &events[2]);
		STATUS_CHK("Error: Writing to buffer. (sigma_arr_buffer)");

		status = clWaitForEvents(1, &events[2]);
		STATUS_CHK("Error: Waiting for kernel run to finish. (clWaitForEvents)\n");
		status = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
		STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
		status = clGetEventProfilingInfo(events[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
		STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
		memory_transfer_time += end_time - start_time;

		cont = 1;
		while(cont) {
			cont = 0;
			status = clSetKernelArg(kernelForward, 9, sizeof(cl_uint), (void *)&level);
			STATUS_CHK("Error: Setting forward kernel argument. (level)\n");

			status = clEnqueueWriteBuffer(commandQueue, cont_buffer, CL_FALSE, 0, sizeof(cl_char), &cont, 0, NULL, &events[1]);
			STATUS_CHK("Error: Writing to buffer. (cont_buffer)");

			status = clWaitForEvents(1, &events[1]);
			STATUS_CHK("Error: Waiting for kernel run to finish. (clWaitForEvents)\n");
			status = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
			STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
			status = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
			STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
			memory_transfer_time += end_time - start_time;

			status = clEnqueueNDRangeKernel(
					commandQueue,
					kernelForward,
					1, // number of dimensions
					NULL,
					globalThreads,
					localThreads,
					0,
					NULL,
					&events[0]);
			STATUS_CHK("Error: Enqueueing kernel onto command queue. (clEnqueueNDRangeKernel)\n");

			/* wait for the kernel call to finish execution */
			status = clWaitForEvents(1, &events[0]);
			STATUS_CHK("Error: Waiting for kernel run to finish. (clWaitForEvents)\n");

			status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
			STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
			status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
			STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
			kernel_execution_time += end_time - start_time;

			status = clEnqueueReadBuffer(commandQueue, cont_buffer, CL_FALSE, 0, sizeof(cl_char), &cont, 0, NULL, &events[1]);
			STATUS_CHK("Error: Reading from buffer (cont_buffer)");

			status = clWaitForEvents(1, &events[1]);
			STATUS_CHK("Error: Waiting for kernel run to finish. (clWaitForEvents)\n");
			status = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
			STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
			status = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
			STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
			memory_transfer_time += end_time - start_time;

			++level;
		}

		status = clEnqueueNDRangeKernel(
				commandQueue,
				kernelDeltaInit,
				1, // number of dimensions
				NULL,
				globalThreadsNormal,
				localThreads,
				0,
				NULL,
				&events[0]);
		STATUS_CHK("Error: Enqueueing kernel onto command queue. (clEnqueueNDRangeKernel)\n");

		/* wait for the kernel call to finish execution */
		status = clWaitForEvents(1, &events[0]);
		STATUS_CHK("Error: Waiting for kernel run to finish. (clWaitForEvents)\n");

		status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
		STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
		status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
		STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
		kernel_execution_time += end_time - start_time;

		status = clEnqueueReadBuffer(commandQueue, delta_arr_buffer, CL_TRUE, 0, sizeof(cl_float) * vertex_num, delta_arr, 0, NULL, NULL);
		STATUS_CHK("Error: Reading from buffer. (delta_arr_buffer)");

		while (level > 1) {
			level = level - 1;

			status = clSetKernelArg(kernelBackward, 8, sizeof(cl_uint), (void *)&level);
			STATUS_CHK("Error: Setting backward kernel argument. (level)\n");

			status = clEnqueueNDRangeKernel(
					commandQueue,
					kernelBackward,
					1, // number of dimensions
					NULL,
					globalThreads,
					localThreads,
					0,
					NULL,
					&events[0]);
			STATUS_CHK("Error: Enqueueing kernel onto command queue. (clEnqueueNDRangeKernel)\n");

			status = clWaitForEvents(1, &events[0]);
			STATUS_CHK("Error: Waiting for kernel run to finish. (clWaitForEvents)\n");

			clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
			clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
			kernel_execution_time += end_time - start_time;
		}

		status = clSetKernelArg(kernelBCUpdate, 4, sizeof(cl_uint), (void *)&s);
		STATUS_CHK("Error: Setting bcupdate kernel argument. (s)\n");

		status = clEnqueueNDRangeKernel(
				commandQueue,
				kernelBCUpdate,
				1, // number of dimensions
				NULL,
				globalThreadsNormal,
				localThreads,
				0,
				NULL,
				&events[0]);
		STATUS_CHK("Error: Enqueueing kernel onto command queue. (clEnqueueNDRangeKernel)\n");

		/* wait for the kernel call to finish execution */
		status = clWaitForEvents(1, &events[0]);
		STATUS_CHK("Error: Waiting for kernel run to finish. (clWaitForEvents)\n");

		status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
		STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
		status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
		STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
		kernel_execution_time += end_time - start_time;

		dist_arr[s] = -1;
	}

	status = clEnqueueReadBuffer(commandQueue, bc_arr_buffer, CL_FALSE, 0, sizeof(cl_float) * vertex_num, bc_arr, 0, NULL, &events[1]);
	STATUS_CHK("Error: Reading from buffer (cont_buffer)");

	status = clWaitForEvents(1, &events[1]);
	STATUS_CHK("Error: Waiting for kernel run to finish. (clWaitForEvents)\n");
	status = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
	STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
	status = clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
	STATUS_CHK("Error: Get profiling info. (clGetEventProfilingInfo)\n");
	memory_transfer_time += end_time - start_time;

	status = clReleaseEvent(events[0]);
	STATUS_CHK("Error: Release event object. (clReleaseEvent)\n");
	status = clReleaseEvent(events[1]);
	STATUS_CHK("Error: Release event object. (clReleaseEvent)\n");
	status = clReleaseEvent(events[2]);
	STATUS_CHK("Error: Release event object. (clReleaseEvent)\n");

	free(dist_arr);
	free(delta_arr);
	free(sigma_arr);

	return 0;
}


/*
 * \brief Release OpenCL resources (Context, Memory etc.) 
 */
int cleanupCL(void) {
	cl_int status;

	status = clReleaseKernel(kernelForward);
	STATUS_CHK("Error: In clReleaseKernel \n");

	status = clReleaseKernel(kernelDeltaInit);
	STATUS_CHK("Error: In clReleaseKernel \n");

	status = clReleaseKernel(kernelBackward);
	STATUS_CHK("Error: In clReleaseKernel \n");

	status = clReleaseKernel(kernelBCUpdate);
	STATUS_CHK("Error: In clReleaseKernel \n");

	status = clReleaseProgram(program);
	STATUS_CHK("Error: In clReleaseProgram\n");

	status = clReleaseMemObject(offset_arr_buffer);
	STATUS_CHK("Error: In clReleaseMemObject (offset_arr_buffer)\n");

	status = clReleaseMemObject(vmap_arr_buffer);
	STATUS_CHK("Error: In clReleaseMemObject (vmap_arr_buffer)\n");

	status = clReleaseMemObject(nvir_arr_buffer);
	STATUS_CHK("Error: In clReleaseMemObject (nvir_arr_buffer)\n");

	status = clReleaseMemObject(ptrs_arr_buffer);
	STATUS_CHK("Error: In clReleaseMemObject (ptrs_arr_buffer)\n");

	status = clReleaseMemObject(adjs_arr_buffer);
	STATUS_CHK("Error: In clReleaseMemObject (adjs_arr_buffer)\n");

	status = clReleaseMemObject(sigma_arr_buffer);
	STATUS_CHK("Error: In clReleaseMemObject (sigma_arr_buffer)\n");

	status = clReleaseMemObject(dist_buffer);
	STATUS_CHK("Error: In clReleaseMemObject (dist_buffer)\n");

	status = clReleaseMemObject(delta_arr_buffer);
	STATUS_CHK("Error: In clReleaseMemObject (delta_arr_buffer)\n");

	status = clReleaseMemObject(cont_buffer);
	STATUS_CHK("Error: In clReleaseMemObject (delta_arr_buffer)\n");

	status = clReleaseCommandQueue(commandQueue);
	STATUS_CHK("Error: In clReleaseCommandQueue\n");

	status = clReleaseContext(context);
	STATUS_CHK("Error: In clReleaseContext\n");

	return 0;
}


/* 
 * \brief Releases program's resources 
 */
void cleanupHost(void) {
	if (offset_arr != NULL) {
		free(offset_arr);
		offset_arr = NULL;
	}
	if (vmap_arr != NULL) {
		free(vmap_arr);
		vmap_arr = NULL;
	}
	if (nvir_arr != NULL) {
		free(nvir_arr);
		nvir_arr = NULL;
	}
	if (ptrs_arr != NULL) {
		free(ptrs_arr);
		ptrs_arr = NULL;
	}
	if (adjs_arr != NULL) {
		free(adjs_arr);
		adjs_arr = NULL;
	}
	if (bc_arr != NULL) {
		free(bc_arr);
		bc_arr = NULL;
	}
	if (devices != NULL) {
		free(devices);
		devices = NULL;
	}
}

int main(int argc, char * argv[]) {
	// Initialize Host application
	if(initializeHost(argv[1])==1)
		return 1;

	// Initialize OpenCL resources
	if(initializeCL()==1)
		return 1;

	if(getDeviceInfo() == 1) {
		return 1;
	}

	// Run the CL program
	if(runBFS()==1)
		return 1;

	// Print/write results.
	printf("%lu\n", (kernel_execution_time / 1000000L));
	printf("%lu\n", (kernel_execution_time / 1000000L + memory_transfer_time / 1000000L));

	FILE* fout;
	fout = fopen(argv[2], "w+");
	for(size_t i = 0; i < real_vertex_num; ++i) {
		fprintf(fout, "%f\n", bc_arr[i]);
	}
	fclose(fout);

	// Releases OpenCL resources
	if(cleanupCL()==1)
		return 1;

	// Release host resources
	cleanupHost();

	return 0;
}
