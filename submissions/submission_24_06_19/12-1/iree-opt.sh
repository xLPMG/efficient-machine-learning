../iree-build/tools/iree-opt \
    --pass-pipeline="builtin.module(func.func(tosa-to-linalg))" \
    add_tosa.mlir \
    -o add_tosa_linalg.mlir 

../iree-build/tools/iree-opt \
    --pass-pipeline="builtin.module(func.func(iree-codegen-iree-comprehensive-bufferize))" \
    add_tosa_linalg.mlir \
    -o add_tosa_linalg_bufferized.mlir 

../iree-build/tools/iree-opt \
    --pass-pipeline="builtin.module(func.func(convert-linalg-to-parallel-loops))" \
    add_tosa_linalg_bufferized.mlir \
    -o add_tosa_parallel_loops.mlir 