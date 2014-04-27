//#define __global __attribute__((address_space(1)))

inline void AtomicAdd(volatile __global float *source, const float operand) {
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;
	do {
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}


__kernel void brandesKernelForward(
		__global  unsigned int* offset_arr,
		__global  unsigned int* vmap_arr,
		__global  unsigned int* nvir_arr,
		__global  unsigned int* ptrs_arr,
		__global  unsigned int* adjs_arr,
		__global  unsigned int* sigma_arr,
		__global int* dist,
		__global char* cont,
		const unsigned int vnum,
		const unsigned int level) {
	unsigned int uvir = get_global_id(0);
	if(uvir < vnum) {
		unsigned int ptr, u, v;
		u = vmap_arr[uvir];
		if (dist[u] == level) {
			for (ptr = ptrs_arr[u] + offset_arr[uvir]; ptr < ptrs_arr[u + 1]; ptr += nvir_arr[u]) {
				v = adjs_arr[ptr];
				if (dist[v] == -1) {
					dist[v] = level + 1;
					*cont = 1;
				}
				if (dist[v] == level + 1) {
					atomic_add(&sigma_arr[v], sigma_arr[u]);
				}
			}
		}
	}
}

__kernel void brandesKernelDeltaInit(
		__global  unsigned int* sigma_arr,
		__global  float* delta_arr,
		const unsigned int vnum
		) {
	unsigned int u = get_global_id(0);
	if (u < vnum) {
		if(sigma_arr[u] == 0) {
			delta_arr[u] = 0;
		} else {
			delta_arr[u] = 1.0f / (float) sigma_arr[u];
		}
	}
}

__kernel void brandesKernelBackward(
		__global  unsigned int* offset_arr,
		__global  unsigned int* vmap_arr,
		__global  unsigned int* nvir_arr,
		__global  unsigned int* ptrs_arr,
		__global  unsigned int* adjs_arr,
		__global int* dist,
		__global float* delta_arr,
		const unsigned int vnum,
		const unsigned int level) {
	unsigned int uvir = get_global_id(0);
	if (uvir < vnum) {
		unsigned int ptr, v, u, offset;
		u = vmap_arr[uvir];
		offset = offset_arr[uvir] + 1;
		if (dist[u] == level) {
			float sum = 0;
			for (ptr = ptrs_arr[u] + offset_arr[uvir]; ptr < ptrs_arr[u + 1]; ptr += nvir_arr[u]) {
				v = adjs_arr[ptr];
				if (dist[v] == level + 1) {
					sum += delta_arr[v];
				}
			}
			AtomicAdd(&delta_arr[u], sum);
		}
	}
}

__kernel void brandesKernelBCUpdate(
		__global  unsigned int* sigma_arr,
		__global  float* delta_arr,
		__global  float* bc_arr,
		const unsigned int vnum,
		const unsigned int cur) {
	unsigned int u = get_global_id(0);
	if (u < vnum && u != cur) {
		if(sigma_arr[u] != 0) {
			bc_arr[u] += delta_arr[u] * ((float) sigma_arr[u]) - 1.0f; 
		}
	}
}
