iree-build/tools/iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu-features=host \
    --iree-llvmcpu-target-triple=aarch64 \
    --mlir-print-ir-after-all \
    add_tosa.mlir \
    -o add_tosa.vmfb
