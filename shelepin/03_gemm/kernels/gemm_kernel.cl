__kernel void gemm(const unsigned int m, const unsigned int n, const unsigned int k, __global float* a, __global float* b, __global float* c) {
	const unsigned int i = get_local_id(1);  // m
	const unsigned int j = get_local_id(0);  // k
	const unsigned int global_i = get_global_id(1);  // m
	const unsigned int global_j = get_global_id(0);  // k
	__local float local_a[BLOCK][BLOCK];
	__local float local_b[BLOCK][BLOCK];
	const unsigned int num_tiles = n / BLOCK;
	float result = .0;
	for (unsigned int t = 0; t < num_tiles; t++) {
		const unsigned int tiled_i = BLOCK * t + i;
		const unsigned int tiled_j = BLOCK * t + j;
		local_a[i][j] = a[global_i * n + tiled_j];
		local_b[i][j] = b[tiled_i * k + global_j];
		barrier(CLK_LOCAL_MEM_FENCE);
		for (unsigned int l = 0; l < BLOCK; l++) {
			result += local_a[i][l] * local_b[l][j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	c[global_i * k + global_j] = result;
}
