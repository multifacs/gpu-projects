__kernel void gemm(const int m, const int n, const int k, __global float* a, __global float* b, __global float* c) {
	const int row = get_local_id(1);
	const int col = get_local_id(0);
	const int global_row = get_global_id(1);
	const int global_col = get_global_id(0);
	
	__local float local_a[16][16];
	__local float local_b[16][16];
	
	const int tiles_n = n / 16;
	
	float val = 0.0;
	for (unsigned int t = 0; t < tiles_n; t++) {
		
		const int tiled_i = 16 * t + row;
		const int tiled_j = 16 * t + col;
		
		local_a[row][col] = a[global_row * n + tiled_j];
		local_b[row][col] = b[tiled_i * k + global_col];

		barrier(CLK_LOCAL_MEM_FENCE);
		for (unsigned int l = 0; l < 16; l++) {
			val += local_a[row][l] * local_b[l][col];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	c[global_row * k + global_col] = val;
}
