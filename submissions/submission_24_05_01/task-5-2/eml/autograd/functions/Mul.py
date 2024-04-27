## Forward method which computes a*b.
# @param i_ctx context object.
# @param i_a node a.
# @param i_b node b.
# @return result a*b.
def forward( io_ctx,
             i_a,
             i_b ):
  io_ctx.save_for_backward( i_a,
                            i_b )
  l_result = i_a * i_b
  return l_result

## Backward method.
# @param i_ctx context object.
# @param i_grad gradient w.r.t. to output of forward method.
# @return gradient w.r.t. to input of forward method.
def backward( i_ctx,
              i_grad ):    
  l_a, l_b = i_ctx.m_saved_data
  l_grad_a = l_b * i_grad
  l_grad_b = l_a * i_grad
  return [ l_grad_a, l_grad_b ]