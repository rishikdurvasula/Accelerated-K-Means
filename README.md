
# CUDA K-Means — Reproducible, Benchmarkable, and Resume-Proof

This project implements **K-Means clustering** with both:
- A **single-threaded CPU baseline**
- A **CUDA-accelerated** version (GPU), with kernels for assignment and centroid recomputation

It includes **reproducible benchmarking** and **numeric verification** (inertia comparison), so any reviewer can validate your resume claims.

---

## Build

Requirements: CUDA Toolkit (nvcc) and CMake 3.18+

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native ..
cmake --build . -j
```

---

## Run

Arguments (defaults in brackets):
- `--n` number of points [500000]
- `--d` dimensions [32]
- `--k` clusters [10]
- `--iters` Lloyd iterations [10]
- `--reps` repetitions for averaging [1]

Example (large enough to show GPU advantage):
```bash
./cuda_kmeans --n 1000000 --d 32 --k 10 --iters 10
```

Output includes:
```
[CPU]   time:  XXXX ms
[GPU]   kernel time (avg):  YYYY ms
[GPU]   end-to-end time (avg):  ZZZZ ms
[Verify] inertia CPU: AAAAA | GPU: BBBBB | rel diff: RRRR
[Speedup] CPU/GPU (kernel):   S1.x
[Speedup] CPU/GPU (end-to-end): S2.x
```



---

## License
MIT
