//Atomic add of floats
inline void atomicFloatAdd(volatile __global float *source, const float operand) {
	union {
		unsigned int intVal;
		float floatVal;
	} newVal, prevVal;
	do {
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile __global int*) source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}


__kernel void templateKernel(__global  unsigned int * output,
                             __global  unsigned int * input,
                             const     unsigned int multiplier)
{
    uint tid = get_global_id(0);
    
    output[tid] = input[tid] * multiplier;
}
