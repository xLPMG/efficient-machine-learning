import unittest
from . import Nop

class TestNop( unittest.TestCase ):
  def test_backward( self ):
    l_grad = Nop.backward( None,
                           5.0 )

    self.assertEqual( l_grad,
                      5.0 )