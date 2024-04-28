import unittest
import math
from . import node

class TestNode( unittest.TestCase ):
    def test_forward( self ):
        l_x = node.Node( 1 )
        l_y = node.Node( 2 )
        l_z = node.Node( 3 )
        l_a = l_y + l_z
        l_b = l_x * l_a
        self.assertAlmostEqual(l_b.m_value, 5)
        
    def test_backward( self ):
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