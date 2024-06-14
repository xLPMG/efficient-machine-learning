module {
  func.func @add(%arg0: tensor<3x2xf32>, %arg1: tensor<1x1xf32>) -> tensor<3x2xf32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_memref %arg1 : memref<1x1xf32, strided<[?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<3x2xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() : memref<3x2xf32>
    scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c3, %c2) step (%c1, %c1) {
      %3 = memref.load %1[%arg2, %arg3] : memref<3x2xf32, strided<[?, ?], offset: ?>>
      %4 = memref.load %0[%c0, %c0] : memref<1x1xf32, strided<[?, ?], offset: ?>>
      %5 = arith.addf %3, %4 : f32
      memref.store %5, %alloc[%arg2, %arg3] : memref<3x2xf32>
      scf.reduce 
    }
    %2 = bufferization.to_tensor %alloc : memref<3x2xf32>
    return %2 : tensor<3x2xf32>
  }
}

