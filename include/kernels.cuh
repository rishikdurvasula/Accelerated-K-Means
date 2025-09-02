
#pragma once
#include <cuda_runtime.h>
#include <cmath>

// Assign each point to the nearest centroid and accumulate sums & counts via atomics.
// points: [N, D], centroids: [K, D], sums: [K, D], counts: [K]
__global__ void assign_and_accumulate(const float* __restrict__ points,
                                      const float* __restrict__ centroids,
                                      int* __restrict__ assignments,
                                      float* __restrict__ sums,
                                      int* __restrict__ counts,
                                      int N, int D, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Find nearest centroid
    int best_k = 0;
    float best_dist = INFINITY;

    for (int k = 0; k < K; ++k) {
        float dist = 0.f;
        int base_p = i * D;
        int base_c = k * D;
        for (int d = 0; d < D; ++d) {
            float diff = points[base_p + d] - centroids[base_c + d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_k = k;
        }
    }

    assignments[i] = best_k;

    // Accumulate sums for the best_k centroid
    int base_p = i * D;
    int base_s = best_k * D;
    for (int d = 0; d < D; ++d) {
        atomicAdd(&sums[base_s + d], points[base_p + d]);
    }
    atomicAdd(&counts[best_k], 1);
}

// Recompute centroids from sums and counts. If a centroid has 0 points, keep it unchanged.
__global__ void finalize_centroids(float* __restrict__ centroids,
                                   const float* __restrict__ sums,
                                   const int* __restrict__ counts,
                                   int D, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    int cnt = counts[k];
    if (cnt <= 0) return;

    int base = k * D;
    for (int d = 0; d < D; ++d) {
        centroids[base + d] = sums[base + d] / (float)cnt;
    }
}
