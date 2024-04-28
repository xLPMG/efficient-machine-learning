import math

## Forward method which computes e^a.
# @param i_ctx context object.
# @param i_a node a.
# @return result .
def forward( io_ctx,
             i_a ):
  io_ctx.save_for_backward( i_a )
  return math.exp( i_a )

## Backward method.
# @param i_ctx context object.
# @param i_grad gradient w.r.t. to output of forward method.
# @return gradient w.r.t. to input of forward method.
def backward( i_ctx,
              i_grad ):
  l_a = i_ctx.m_saved_data
  return math.exp( l_a[0] ) * i_grad