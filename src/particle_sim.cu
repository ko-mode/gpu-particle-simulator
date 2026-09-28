#include <cuda_runtime.h>

#include <iostream>

#include <vector>

struct Particle {
    float x; // position (x,y)
    float y;
    float vx; // velocity (vx, vy)
    float vy;
};

__global__ void updateParticles(
    Particle* particles, 
    int n, float dt, 
    float gravity,
    float floorY,
    float restitution
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        particles[i].vy += gravity * dt;

        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;

        if (particles[i].y < floorY) {
            particles[i].y = floorY;
            particles[i].vy = -particles[i].vy * restitution;
        }
    }
}

int main() {
    constexpr int n = 10000;
    constexpr float dt = 0.016f;
    constexpr float gravity = -9.81f;
    constexpr float floorY = -5.0f;
    constexpr float restitution = 0.8f;

    std::vector<Particle> hostParticles(n);

    for (int i = 0; i < n; ++i) {
        hostParticles[i] = {
            static_cast<float>(i % 100) * 0.1f,
            static_cast<float>(i / 100) * 0.1f,
            1.0f,
            0.0f
        };
    }

    Particle* deviceParticles = nullptr;

    cudaMalloc(&deviceParticles, n * sizeof(Particle));

    cudaMemcpy(
        deviceParticles,
        hostParticles.data(),
        n * sizeof(Particle),
        cudaMemcpyHostToDevice
    );

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    constexpr int steps = 600;

    // Keep data on the GPU as long as possible 
    // As transfer time (cudaMemcpy) between the CPU and GPU is relatively expensive compared with doing arithmetic on GPU data
    for (int step = 0; step < steps; ++step) {
        updateParticles<<<blocks, threadsPerBlock>>>(
            deviceParticles,
            n, 
            dt,
            gravity,
            floorY,
            restitution
        );
    }

    cudaMemcpy(
        hostParticles.data(),
        deviceParticles,
        n * sizeof(Particle),
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < 5; ++i) {
        std::cout
            << "Particle " << i
            << ": (" << hostParticles[i].x
            << ", " << hostParticles[i].y
            << ")\n";
    }

    cudaFree(deviceParticles);

    return 0;
}