import unittest
from . import Add

class TestAdd( unittest.TestCase ):
  def test_forward( self ):
    l_result = Add.forward( None,
                            3.0,
                            4.0 )
    
    self.assertAlmostEqual( l_result,
                            7.0 )
  
  def test_backward( self ):
    l_grad_a, l_grad_b = Add.backward( None,
                                       5.0 )
    
    self.assertEqual( l_grad_a, 5.0 )
    self.assertEqual( l_grad_b, 5.0 )