#accesses = [
affine_map<(m, n, k) -> (m, k)>,
affine_map<(m, n, k) -> (k, n)>,
affine_map<(m, n, k) -> (m, n)>
]

#attrs = {
    indexing_maps = #accesses,
    iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @matmul(  %lhs  : tensor<?x?xf32>, 
                    %rhs  : tensor<?x?xf32>,
                    %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic #attrs
        ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%init : tensor<?x?xf32>) {
            ^bb(%a: f32, %b: f32, %c: f32):
                %0 = arith.mulf %a, %b : f32
                %1 = arith.addf %c, %0 : f32
                linalg.yield %1 : f32
        } -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
}