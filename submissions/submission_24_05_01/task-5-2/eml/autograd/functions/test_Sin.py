import unittest
from . import Sin
from .. import context
import math

class TestExp( unittest.TestCase ):
  def test_forward( self ):
    ctx = context.Context()
    l_result = Sin.forward( ctx,
                            math.radians(90) )
    
    self.assertAlmostEqual( l_result,
                            1 )
  
  def test_backward( self ):
    ctx = context.Context()
    Sin.forward( ctx,
                 math.radians(180) )
    l_grad_a = Sin.backward( ctx,
                             5.0 )

    self.assertAlmostEqual( l_grad_a, -5.0 )