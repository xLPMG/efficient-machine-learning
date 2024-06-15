../iree-build/tools/iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu-features=host \
    --iree-llvmcpu-target-triple=aarch64 \
    linalg-matmul.mlir \
    -o build/linalg-matmul.vmfb

../iree-build/tools/iree-run-module \
    --module=build/linalg-matmul.vmfb \
    --input="3x2xf32=[0 1][2 3][4 5]" \
    --input="2x3xf32=[0 1 2][3 4 5]" \
    --input="3x3xf32=[0 0 0][0 0 0][0 0 0]" \
    > out/linalg-matmul.out 2>&1

echo ""                                    >> out/linalg-matmul.out 
echo "Correct result:"                     >> out/linalg-matmul.out              
echo  "| 0 1 |               |  3  4  5 |" >> out/linalg-matmul.out
echo  "| 2 3 | * | 0 1 2 | = |  9 14 19 |" >> out/linalg-matmul.out
echo  "| 4 5 |   | 3 4 5 |   | 15 24 33 |" >> out/linalg-matmul.out

../iree-build/tools/iree-opt \
    --pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops))" \
    linalg-matmul.mlir \
    -o linalg-matmul-generalized.mlir 