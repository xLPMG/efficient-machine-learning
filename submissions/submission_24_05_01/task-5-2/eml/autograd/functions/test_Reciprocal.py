import unittest
from . import Reciprocal
from .. import context

class TestReciprocal( unittest.TestCase ):
  def test_forward( self ):
    ctx = context.Context()
    l_result = Reciprocal.forward( ctx,
                                   4.0 )
    
    self.assertAlmostEqual( l_result,
                            0.25 )
  
  def test_backward( self ):
    ctx = context.Context()
    l_result = Reciprocal.forward( ctx,
                                   4.0 )
    l_grad_a = Reciprocal.backward( ctx,
                                    5.0 )
    l_result = -1/16 * 5.0
    self.assertAlmostEqual( l_grad_a, l_result )