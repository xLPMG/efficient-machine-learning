## Forward method which compute a+b.
# @param i_ctx context object.
# @param i_a node a.
# @param i_b node b.
# @return result a+b.
def forward( io_ctx,
             i_a,
             i_b ):
  l_result = i_a + i_b
  return l_result

## Backward method.
# @param i_ctx context object.
# @param i_grad gradient w.r.t. to output of forward method.
# @return gradient w.r.t. to input of forward method.
def backward( i_ctx,
              i_grad ):
  l_grad_a = i_grad
  l_grad_b = i_grad
  return [ l_grad_a, l_grad_b ]