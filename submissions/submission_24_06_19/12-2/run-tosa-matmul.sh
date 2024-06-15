../iree-build/tools/iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu-features=host \
    --iree-llvmcpu-target-triple=aarch64 \
    tosa-matmul.mlir \
    -o build/tosa-matmul.vmfb

../iree-build/tools/iree-run-module \
    --module=build/tosa-matmul.vmfb \
    --input="1x3x2xf32=[0 1][2 3][4 5]" \
    --input="1x2x3xf32=[0 1 2][3 4 5]" \
    > out/tosa-matmul.out 2>&1

echo ""                                    >> out/tosa-matmul.out 
echo "Correct result:"                     >> out/tosa-matmul.out              
echo  "| 0 1 |               |  3  4  5 |" >> out/tosa-matmul.out
echo  "| 2 3 | * | 0 1 2 | = |  9 14 19 |" >> out/tosa-matmul.out
echo  "| 4 5 |   | 3 4 5 |   | 15 24 33 |" >> out/tosa-matmul.out