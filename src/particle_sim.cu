#include <cuda_runtime.h>

#include <iostream>

struct Particle {
    float x; // position (x,y)
    float y;
    float vx; // velocity (vx, vy)
    float vy;
};

__global__ void updateParticles(Particle* particles, int n, float dt, float gravity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        particles[i].vy += gravity * dt;

        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
    }
}

int main() {
    constexpr int n = 4;
    constexpr float dt = 0.016f;
    constexpr float gravity = -9.81f;

    Particle hostParticles[n] = {
        {0.0f, 0.0f, 1.0f, 0.5f},
        {1.0f, 2.0f, -1.0f, 1.0f},
        {3.0f, 1.0f, 0.5f, -0.5f},
        {2.0f, 4.0f, 0.0f, -1.0f}
    };

    Particle* deviceParticles = nullptr;

    cudaMalloc(&deviceParticles, n * sizeof(Particle));

    cudaMemcpy(
        deviceParticles,
        hostParticles,
        n * sizeof(Particle),
        cudaMemcpyHostToDevice
    );

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    constexpr int steps = 60;

    // Keep data on the CPU as long as possible 
    // As transfer time (cudaMemcpy) between the CPU and GPU is relatively expensive compared with doing arithmetic on GPU data
    for (int step = 0; step < steps; ++step) {
        updateParticles<<<blocks, threadsPerBlock>>>(
            deviceParticles,
            n, 
            dt,
            gravity
        );
    };

    cudaMemcpy(
        hostParticles,
        deviceParticles,
        n * sizeof(Particle),
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < n; ++i) {
        std::cout
            << "Particle " << i
            << ": (" << hostParticles[i].x
            << ", " << hostParticles[i].y
            << ")\n";
    };

    cudaFree(deviceParticles);

    return 0;
}