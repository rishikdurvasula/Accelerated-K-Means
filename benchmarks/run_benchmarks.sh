
#!/usr/bin/env bash
set -euo pipefail
N=${1:-1000000}
D=${2:-32}
K=${3:-10}
ITERS=${4:-10}
REPS=${5:-1}
./cuda_kmeans --n "$N" --d "$D" --k "$K" --iters "$ITERS" --reps "$REPS"
