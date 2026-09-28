#include <cuda_runtime.h>

#include <iostream>

__global__ void doubleValues(float* values, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        values[i] *= 2.0f;
    }
}

int main() {
    constexpr int n = 8;

    float hostValues[n] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };

    float* deviceValues = nullptr;

    cudaMalloc(&deviceValues, n * sizeof(float));

    cudaMemcpy(
        deviceValues,
        hostValues,
        n * sizeof(float),
        cudaMemcpyHostToDevice
    );

    int threadsPerBlock = 4;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    doubleValues<<<blocks, threadsPerBlock>>>(deviceValues, n);

    cudaMemcpy(
        hostValues,
        deviceValues,
        n * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < n; ++i) {
        std::cout << hostValues[i] << " ";
    }

    std::cout << "\n";

    cudaFree(deviceValues);

    return 0;
}