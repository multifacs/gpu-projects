__kernel void matmul (const unsigned int m, const unsigned int n, const unsigned int k, __global float* a, __global float* b, __global float* c) {
    unsigned int row = get_global_id(1);
    unsigned int col = get_global_id(0);
    float res = .0;
    for (unsigned int l = 0; l < n; l++) {
        res += a[row * n + l] * b[l * k + col];
    }
    c[row * k + col] = res;
}

__kernel void gemm(const unsigned int m, const unsigned int n, const unsigned int k, __global float* a, __global float* b, __global float* c) {
	unsigned int row = get_local_id(1);
	unsigned int col = get_local_id(0);
	
	unsigned int global_row = get_global_id(1);
	unsigned int global_col = get_global_id(0);
	
	__local float loc_a[BLOCK][BLOCK];
	__local float loc_b[BLOCK][BLOCK];
	
	const unsigned int num_tiles = n / BLOCK;
	
	float res = .0;
	for (unsigned int t = 0; t < num_tiles; t++) {
		const unsigned int tiled_i = BLOCK * t + row;
		const unsigned int tiled_j = BLOCK * t + col;
		loc_a[row][col] = a[global_row * n + tiled_j];
		loc_b[row][col] = b[tiled_i * k + global_col];
		barrier(CLK_LOCAL_MEM_FENCE);
		for (unsigned int l = 0; l < BLOCK; l++) {
			res += loc_a[row][l] * loc_b[l][col];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	c[global_row * k + global_col] = res;
}
