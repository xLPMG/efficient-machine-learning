import unittest
from . import Mul
from .. import context

class TestMul( unittest.TestCase ):
  def test_forward( self ):
    ctx = context.Context()
    l_result = Mul.forward( ctx,
                            3.0,
                            4.0 )
    
    self.assertAlmostEqual( l_result,
                            12.0 )
  
  def test_backward( self ):
    ctx = context.Context()
    Mul.forward( ctx,
                 3.0,
                 4.0 )
    l_grad_a, l_grad_b = Mul.backward( ctx,
                                       5.0 )
    
    self.assertEqual( l_grad_a, 4.0 * 5.0 )
    self.assertEqual( l_grad_b, 3.0 * 5.0 )