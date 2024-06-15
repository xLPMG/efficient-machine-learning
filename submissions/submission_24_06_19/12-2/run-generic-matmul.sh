../iree-build/tools/iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu-features=host \
    --iree-llvmcpu-target-triple=aarch64 \
    generic-matmul.mlir \
    -o build/generic-matmul.vmfb

../iree-build/tools/iree-run-module \
    --module=build/generic-matmul.vmfb \
    --input="3x2xf32=[0 1][2 3][4 5]" \
    --input="2x3xf32=[0 1 2][3 4 5]" \
    --input="2x3xf32=[0 0 0][0 0 0]" \
    > out/generic-matmul.out 2>&1

echo ""                                    >> out/generic-matmul.out 
echo "Correct result:"                     >> out/generic-matmul.out              
echo  "| 0 1 |               |  3  4  5 |" >> out/generic-matmul.out
echo  "| 2 3 | * | 0 1 2 | = |  9 14 19 |" >> out/generic-matmul.out
echo  "| 4 5 |   | 3 4 5 |   | 15 24 33 |" >> out/generic-matmul.out