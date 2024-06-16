# TOSA MATMUL
iree-benchmark-module -- \
    --module=build/tosa-matmul.vmfb \
    --device=local-task \
    --function=matmul \
    --input="1x8192x8192xf32" \
    --input="1x8192x8192xf32" \
    --task_topology_max_group_count=1 \
    > out/benchmark-tosa-matmul-1.out 2>&1

iree-benchmark-module -- \
    --module=build/tosa-matmul.vmfb \
    --device=local-task \
    --function=matmul \
    --input="1x8192x8192xf32" \
    --input="1x8192x8192xf32" \
    --task_topology_max_group_count=72 \
    > out/benchmark-tosa-matmul-72.out 2>&1

# GENERIC MATMUL
iree-benchmark-module -- \
    --module=build/generic-matmul.vmfb \
    --device=local-task \
    --function=matmul \
    --input="8192x8192xf32" \
    --input="8192x8192xf32" \
    --input="8192x8192xf32" \
    --task_topology_max_group_count=1 \
    > out/benchmark-generic-matmul-1.out 2>&1

iree-benchmark-module -- \
    --module=build/generic-matmul.vmfb \
    --device=local-task \
    --function=matmul \
    --input="8192x8192xf32" \
    --input="8192x8192xf32" \
    --input="8192x8192xf32" \
    --task_topology_max_group_count=72 \
    > out/benchmark-generic-matmul-72.out 2>&1

# LINALG MATMUL
iree-benchmark-module -- \
    --module=build/linalg-matmul.vmfb \
    --device=local-task \
    --function=matmul \
    --input="8192x8192xf32" \
    --input="8192x8192xf32" \
    --input="8192x8192xf32" \
    --task_topology_max_group_count=1 \
    > out/benchmark-linalg-matmul-1.out 2>&1

iree-benchmark-module -- \
    --module=build/linalg-matmul.vmfb \
    --device=local-task \
    --function=matmul \
    --input="8192x8192xf32" \
    --input="8192x8192xf32" \
    --input="8192x8192xf32" \
    --task_topology_max_group_count=72 \
    > out/benchmark-linalg-matmul-72.out 2>&1