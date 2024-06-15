func.func @matmul(  %lhs: tensor<?x?x?xf32>,
                    %rhs: tensor<?x?x?xf32> ) -> tensor<?x?x?xf32> {
  %out = tosa.matmul %lhs, %rhs : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %out : tensor<?x?x?xf32>
}