## Forward method which computes 1/x.
# @param i_ctx context object.
# @param i_a node a.
# @return result 1/x.
def forward( io_ctx,
             i_a ):
  io_ctx.save_for_backward( i_a )
  l_result = 1 / i_a
  return l_result

## Backward method.
# @param i_ctx context object.
# @param i_grad gradient w.r.t. to output of forward method.
# @return gradient w.r.t. to input of forward method.
def backward( i_ctx,
              i_grad ):
  l_a = i_ctx.m_saved_data
  l_result = -1 / (l_a[0] * l_a[0]) * i_grad
  return l_result