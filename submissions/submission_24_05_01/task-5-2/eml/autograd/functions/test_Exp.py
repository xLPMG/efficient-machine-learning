import unittest
from . import Exp
from .. import context

class TestExp( unittest.TestCase ):
  def test_forward( self ):
    ctx = context.Context()
    l_result = Exp.forward( ctx,
                            4.0 )
    
    self.assertAlmostEqual( l_result,
                            54.5981500331442 )
  
  def test_backward( self ):
    ctx = context.Context()
    Exp.forward( ctx,
                 4.0 )
    l_grad_a = Exp.backward( ctx,
                             5.0 )
    l_result = 272.9907501657212
    self.assertAlmostEqual( l_grad_a, l_result )