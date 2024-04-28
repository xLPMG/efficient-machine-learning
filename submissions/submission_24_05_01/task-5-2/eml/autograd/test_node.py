import unittest
import math
from . import node

class TestNode( unittest.TestCase ):
    def test_forward_F( self ):
        l_x = node.Node( 1 )
        l_y = node.Node( 2 )
        l_z = node.Node( 3 )
        l_a = l_y + l_z
        l_b = l_x * l_a
        self.assertAlmostEqual(l_b.m_value, 5)
        
    def test_backward_F( self ):
        l_x = node.Node( 1 )
        l_y = node.Node( 2 )
        l_z = node.Node( 3 )
        l_a = l_y + l_z
        l_b = l_x * l_a
        
        l_b.backward( 1.0 )
        self.assertAlmostEqual(l_b.m_grad, 1.0)
        self.assertAlmostEqual(l_a.m_grad, 1.0)
        self.assertAlmostEqual(l_x.m_grad, 5.0)
        self.assertAlmostEqual(l_y.m_grad, 1.0)
        self.assertAlmostEqual(l_z.m_grad, 1.0)
        
        l_b.zero_grad()
        self.assertAlmostEqual(l_b.m_grad, 0.0)
        self.assertAlmostEqual(l_a.m_grad, 0.0)
        self.assertAlmostEqual(l_x.m_grad, 0.0)
        self.assertAlmostEqual(l_y.m_grad, 0.0)
        self.assertAlmostEqual(l_z.m_grad, 0.0)
        
    def test_forward_G( self ):
        l_w0 = node.Node( 1 )
        l_w1 = node.Node( 2 )
        l_w2 = node.Node( 3 )
        l_x0 = node.Node( 4 )
        l_x1 = node.Node( 5 )
        
        l_a = node.Node( -1.0 ) * (l_w0 * l_x0 + l_w1 * l_x1 + l_w2)
        l_b = node.Node( 1 ) + l_a.exp()
        l_c = l_b.reciprocal()
        
        result = 0.9999999586006
        self.assertAlmostEqual(l_c.m_value, result)
        
    def test_backward_G( self ):
        # TODO: implement
        self.assertAlmostEqual(2, 2)
        
    def test_forward_H( self ):
        l_x = node.Node( 1 )
        l_y = node.Node( 2 )
        l_a = ( l_x * l_y ).sin()
        l_b = ( l_x + l_y ).cos()
        l_c = ( l_x - l_y).exp()
        l_d = l_c.reciprocal()
        l_e = (l_a + l_b) * l_d
        
        self.assertAlmostEqual(l_e.m_value, -0.21935194181497492)
        
    def test_backward_H( self ):
        # TODO: implement
        self.assertAlmostEqual(2, 2)