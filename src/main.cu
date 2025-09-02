
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include "kernels.cuh"

static void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(result) << std::endl;
        std::exit(1);
    }
}

// Generate synthetic Gaussian blobs with fixed seed for reproducibility
void generate_blobs(std::vector<float>& points, int N, int D, int K, float spread=0.6f) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> center_dist(-5.f, 5.f);
    std::normal_distribution<float> noise(0.f, spread);

    std::vector<float> centers(K * D);
    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < D; ++d) {
            centers[k*D + d] = center_dist(rng);
        }
    }

    for (int i = 0; i < N; ++i) {
        int k = i % K; // roughly balanced
        for (int d = 0; d < D; ++d) {
            points[i*D + d] = centers[k*D + d] + noise(rng);
        }
    }
}

// CPU single-threaded K-Means (Lloyd's algorithm)
void kmeans_cpu(const std::vector<float>& points, int N, int D, int K, int iters,
                std::vector<int>& assign_out, std::vector<float>& centroids_out) {
    // init centroids: take first K points (deterministic)
    centroids_out.assign(points.begin(), points.begin() + K*D);

    std::vector<float> sums(K * D, 0.f);
    std::vector<int> counts(K, 0);

    assign_out.assign(N, 0);

    for (int it = 0; it < iters; ++it) {
        // reset accumulators
        std::fill(sums.begin(), sums.end(), 0.f);
        std::fill(counts.begin(), counts.end(), 0);

        // assign step
        for (int i = 0; i < N; ++i) {
            int best_k = 0;
            float best_dist = INFINITY;
            for (int k = 0; k < K; ++k) {
                float dist = 0.f;
                for (int d = 0; d < D; ++d) {
                    float diff = points[i*D + d] - centroids_out[k*D + d];
                    dist += diff*diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }
            assign_out[i] = best_k;
            // accumulate
            for (int d = 0; d < D; ++d) {
                sums[best_k*D + d] += points[i*D + d];
            }
            counts[best_k] += 1;
        }

        // update step
        for (int k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                for (int d = 0; d < D; ++d) {
                    centroids_out[k*D + d] = sums[k*D + d] / (float)counts[k];
                }
            }
        }
    }
}

double inertia(const std::vector<float>& points, const std::vector<int>& assign,
               const std::vector<float>& centroids, int N, int D) {
    double sse = 0.0;
    for (int i = 0; i < N; ++i) {
        int k = assign[i];
        double dist = 0.0;
        for (int d = 0; d < D; ++d) {
            double diff = (double)points[i*D + d] - (double)centroids[k*D + d];
            dist += diff*diff;
        }
        sse += dist;
    }
    return sse;
}

int main(int argc, char** argv) {
    int N = 500000; // number of points
    int D = 32;     // dimensions
    int K = 10;     // clusters
    int iters = 10;
    int reps = 1;
    bool verify = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i+1 < argc) N = std::stoi(argv[++i]);
        else if (arg == "--d" && i+1 < argc) D = std::stoi(argv[++i]);
        else if (arg == "--k" && i+1 < argc) K = std::stoi(argv[++i]);
        else if (arg == "--iters" && i+1 < argc) iters = std::stoi(argv[++i]);
        else if (arg == "--reps" && i+1 < argc) reps = std::stoi(argv[++i]);
        else if (arg == "--no-verify") verify = false;
        else if (arg == "--help") {
            std::cout << "Usage: ./cuda_kmeans [--n N] [--d D] [--k K] [--iters I] [--reps R] [--no-verify]\n";
            return 0;
        }
    }

    std::cout << "K-Means  N=" << N << " D=" << D << " K=" << K << " iters=" << iters << " reps=" << reps << std::endl;

    // Host data
    std::vector<float> points((size_t)N * D);
    generate_blobs(points, N, D, K);

    // CPU baseline
    std::vector<int> assign_cpu(N);
    std::vector<float> centroids_cpu(K * D);
    auto t0 = std::chrono::high_resolution_clock::now();
    kmeans_cpu(points, N, D, K, iters, assign_cpu, centroids_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[CPU]   time: " << cpu_ms << " ms" << std::endl;

    // Device allocations
    float *d_points=nullptr, *d_centroids=nullptr, *d_sums=nullptr;
    int *d_assign=nullptr, *d_counts=nullptr;
    size_t pts_bytes = (size_t)N * D * sizeof(float);
    size_t cent_bytes = (size_t)K * D * sizeof(float);
    size_t sums_bytes = cent_bytes;
    size_t counts_bytes = (size_t)K * sizeof(int);

    checkCuda(cudaMalloc(&d_points, pts_bytes), "d_points malloc");
    checkCuda(cudaMalloc(&d_centroids, cent_bytes), "d_centroids malloc");
    checkCuda(cudaMalloc(&d_sums, sums_bytes), "d_sums malloc");
    checkCuda(cudaMalloc(&d_counts, counts_bytes), "d_counts malloc");
    checkCuda(cudaMalloc(&d_assign, (size_t)N * sizeof(int)), "d_assign malloc");

    checkCuda(cudaMemcpy(d_points, points.data(), pts_bytes, cudaMemcpyHostToDevice), "H2D points");

    // Init centroids identically to CPU: first K points
    checkCuda(cudaMemcpy(d_centroids, points.data(), cent_bytes, cudaMemcpyHostToDevice), "H2D initial centroids");

    dim3 block(256);
    dim3 grid_points((N + block.x - 1) / block.x);
    dim3 grid_centroids((K + block.x - 1) / block.x);

    float kernel_ms_total = 0.0f;
    float total_ms_with_transfers = 0.0f;

    for (int r = 0; r < reps; ++r) {
        // Reset centroids to initial for each rep
        checkCuda(cudaMemcpy(d_centroids, points.data(), cent_bytes, cudaMemcpyHostToDevice), "reset centroids");

        cudaEvent_t start_all, stop_all;
        cudaEventCreate(&start_all);
        cudaEventCreate(&stop_all);

        cudaEventRecord(start_all);

        float iter_ms_sum = 0.0f;
        for (int it = 0; it < iters; ++it) {
            checkCuda(cudaMemset(d_sums, 0, sums_bytes), "memset sums");
            checkCuda(cudaMemset(d_counts, 0, counts_bytes), "memset counts");

            cudaEvent_t start_it, stop_it;
            cudaEventCreate(&start_it);
            cudaEventCreate(&stop_it);

            cudaEventRecord(start_it);

            assign_and_accumulate<<<grid_points, block>>>(d_points, d_centroids, d_assign, d_sums, d_counts, N, D, K);
            checkCuda(cudaPeekAtLastError(), "assign_and_accumulate");
            checkCuda(cudaDeviceSynchronize(), "sync after assign");

            finalize_centroids<<<grid_centroids, block>>>(d_centroids, d_sums, d_counts, D, K);
            checkCuda(cudaPeekAtLastError(), "finalize_centroids");
            checkCuda(cudaDeviceSynchronize(), "sync after finalize");

            cudaEventRecord(stop_it);
            cudaEventSynchronize(stop_it);
            float iter_ms = 0.0f;
            cudaEventElapsedTime(&iter_ms, start_it, stop_it);
            iter_ms_sum += iter_ms;

            cudaEventDestroy(start_it);
            cudaEventDestroy(stop_it);
        }

        cudaEventRecord(stop_all);
        cudaEventSynchronize(stop_all);
        float all_ms = 0.0f;
        cudaEventElapsedTime(&all_ms, start_all, stop_all);

        kernel_ms_total += iter_ms_sum;
        total_ms_with_transfers += all_ms;

        cudaEventDestroy(start_all);
        cudaEventDestroy(stop_all);
    }

    kernel_ms_total /= reps;
    total_ms_with_transfers /= reps;

    std::cout << "[GPU]   kernel time (avg): " << kernel_ms_total << " ms" << std::endl;
    std::cout << "[GPU]   end-to-end time (avg): " << total_ms_with_transfers << " ms" << std::endl;

    // Fetch GPU results once for verification
    std::vector<int> assign_gpu(N);
    std::vector<float> centroids_gpu(K * D);

    checkCuda(cudaMemcpy(assign_gpu.data(), d_assign, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost), "D2H assign");
    checkCuda(cudaMemcpy(centroids_gpu.data(), d_centroids, cent_bytes, cudaMemcpyDeviceToHost), "D2H centroids");

    // Compute inertia for both
    double sse_cpu = inertia(points, assign_cpu, centroids_cpu, N, D);
    double sse_gpu = inertia(points, assign_gpu, centroids_gpu, N, D);
    double rel = std::abs(sse_cpu - sse_gpu) / std::max(1.0, std::abs(sse_cpu));

    std::cout << "[Verify] inertia CPU: " << sse_cpu << " | GPU: " << sse_gpu << " | rel diff: " << rel << std::endl;

    // Speedups
    double sp_kernel = cpu_ms / kernel_ms_total;
    double sp_e2e = cpu_ms / total_ms_with_transfers;
    std::cout << "[Speedup] CPU/GPU (kernel): " << sp_kernel << "x" << std::endl;
    std::cout << "[Speedup] CPU/GPU (end-to-end): " << sp_e2e << "x" << std::endl;

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_assign);

    return 0;
}
