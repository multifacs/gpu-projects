__kernel void normal (const int m, const int n, const int k, __global float* a, __global float* b, __global float* c) {
    const int row = get_global_id(1);
    const int col = get_global_id(0);
    
    float val = 0.0;
    
    for (unsigned int l = 0; l < n; l++) {
        val += a[row * n + l] * b[l * k + col];
    }
    
    c[row * k + col] = val;
}
