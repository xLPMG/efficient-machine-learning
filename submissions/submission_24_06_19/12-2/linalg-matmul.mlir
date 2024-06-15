func.func @matmul(  %lhs  : tensor<?x?xf32>, 
                    %rhs  : tensor<?x?xf32>,
                    %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.matmul
        ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
}