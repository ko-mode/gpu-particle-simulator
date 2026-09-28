#include <cuda_runtime.h>

#include <iostream>

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "No CUDa devices found.\n";
        return 1;
    }

    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);

    std::cout << "GPU: " << properties.name << '\n';
    std::cout << "Compute capability: "
              << properties.major << '.'
              << properties.minor << '\n';

    std::cout << "Streaming multiprocessors: "
              << properties.multiProcessorCount << '\n';

    std::cout << "Warp size: "
              << properties.warpSize << '\n';

    std::cout << "Max threads per block: "
              << properties.maxThreadsPerBlock << '\n';

    std::cout << "Global memory: "
              << properties.totalGlobalMem / (1024 * 1024)
              << " MiB\n";

    return 0;
}
