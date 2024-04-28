import unittest
from . import Cos
from .. import context
import math

class TestExp( unittest.TestCase ):
  def test_forward( self ):
    ctx = context.Context()
    l_result = Cos.forward( ctx,
                            math.radians(90) )
    
    self.assertAlmostEqual( l_result,
                            0 )
  
  def test_backward( self ):
    ctx = context.Context()
    Cos.forward( ctx, 
                 math.radians(90) )
    l_grad_a = Cos.backward( ctx,
                             5.0 )

    self.assertAlmostEqual( l_grad_a, -5.0 )