# TOSA MATMUL
iree-benchmark-module -- \
    --module=build/tosa-matmul.vmfb \
    --device=local-task \
    --function=matmul \
    --input="1x8192x8192xf32" \
    --input="1x8192x8192xf32" \
    > out/benchmark-tosa-matmul.out 2>&1

# GENERIC MATMUL
iree-benchmark-module -- \
    --module=build/generic-matmul.vmfb \
    --device=local-task \
    --function=matmul \
    --input="8192x8192xf32" \
    --input="8192x8192xf32" \
    --input="8192x8192xf32" \
    > out/benchmark-generic-matmul.out 2>&1

# LINALG MATMUL
iree-benchmark-module -- \
    --module=build/linalg-matmul.vmfb \
    --device=local-task \
    --function=matmul \
    --input="8192x8192xf32" \
    --input="8192x8192xf32" \
    --input="8192x8192xf32" \
    > out/benchmark-linalg-matmul.out 2>&1
