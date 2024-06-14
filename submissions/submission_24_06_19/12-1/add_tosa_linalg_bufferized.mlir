#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, 0)>
module {
  func.func @add(%arg0: tensor<3x2xf32>, %arg1: tensor<1x1xf32>) -> tensor<3x2xf32> {
    %0 = bufferization.to_memref %arg1 : memref<1x1xf32, strided<[?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<3x2xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() : memref<3x2xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %0 : memref<3x2xf32, strided<[?, ?], offset: ?>>, memref<1x1xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<3x2xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.addf %in, %in_0 : f32
      linalg.yield %3 : f32
    }
    %2 = bufferization.to_tensor %alloc : memref<3x2xf32>
    return %2 : tensor<3x2xf32>
  }
}

