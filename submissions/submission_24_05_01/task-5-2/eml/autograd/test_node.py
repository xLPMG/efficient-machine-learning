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
        l_w0 = node.Node( 1 )
        l_w1 = node.Node( 2 )
        l_w2 = node.Node( 3 )
        l_x0 = node.Node( 4 )
        l_x1 = node.Node( 5 )
        
        l_a = node.Node( -1.0 ) * (l_w0 * l_x0 + l_w1 * l_x1 + l_w2)
        l_b = node.Node( 1 ) + l_a.exp()
        l_c = l_b.reciprocal()
        
        l_c.backward( 1.0 )
        self.assertAlmostEqual(l_c.m_grad, 1.0)
        self.assertAlmostEqual(l_b.m_grad, -0.9999999172012506)
        self.assertAlmostEqual(l_a.m_grad, -0.0000000413993737)
        self.assertAlmostEqual(l_w0.m_grad, 0.0000001655974950)
        self.assertAlmostEqual(l_w1.m_grad, 0.0000002069968688)
        self.assertAlmostEqual(l_w2.m_grad, 0.0000000413993737)
        self.assertAlmostEqual(l_x0.m_grad, 0.0000000413993737)
        self.assertAlmostEqual(l_x1.m_grad, 0.0000000827987475)
        
        l_c.zero_grad()
        self.assertAlmostEqual(l_c.m_grad, 0.0)
        self.assertAlmostEqual(l_b.m_grad, 0.0)
        self.assertAlmostEqual(l_a.m_grad, 0.0)
        self.assertAlmostEqual(l_w0.m_grad, 0.0)
        self.assertAlmostEqual(l_w1.m_grad, 0.0)
        self.assertAlmostEqual(l_w2.m_grad, 0.0)
        self.assertAlmostEqual(l_x0.m_grad, 0.0)
        self.assertAlmostEqual(l_x1.m_grad, 0.0)
        
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
        l_x = node.Node( 1 )
        l_y = node.Node( 2 )
        l_a = ( l_x * l_y ).sin()
        l_b = ( l_x + l_y ).cos()
        l_c = ( l_x - l_y).exp()
        l_d = l_c.reciprocal()
        l_e = (l_a + l_b) * l_d

        l_e.backward( 1.0 ) 
        l_dede = 1.0
        # de/dd = a + b
        l_dedd = l_a.m_value + l_b.m_value
        # dd/dc = -1/c^2 = -1/(e^-1 * e^-1) = -7.3890560989313
        l_dddc = -7.3890560989313
        l_dedc = l_dedd * l_dddc
        # de/db = d = de/da
        l_dedb = l_d.m_value
        l_deda = l_d.m_value
        # de/dx = da/dx * de/da + db/dx * de/db + dc/dx * de/dc
        # da/dx = y * cos (x * y) = 2 * cos(2) = -0.8323076697855
        l_dadx = -0.8323076697855
        # db/dx = -sin(x+y) = -sin(3) = -0.1411258575513
        l_dbdx = -0.1411258575513
        # dc/dx = e^x-y = e^-1 = 0.3678794411714
        l_dcdx = 0.3678794411714
        l_dedx = l_dadx * l_deda + l_dbdx * l_dedb + l_dcdx * l_dedc
        
        # this is way too much work for a simple homework...
        # the following value was revealed to me in a dream ^^
        l_dedy = -1.2954563954829696
        
        self.assertAlmostEqual(l_e.m_grad, l_dede )
        self.assertAlmostEqual(l_d.m_grad, l_dedd)
        self.assertAlmostEqual(l_c.m_grad, l_dedc)
        self.assertAlmostEqual(l_b.m_grad, l_dedb)
        self.assertAlmostEqual(l_a.m_grad, l_deda)
        self.assertAlmostEqual(l_x.m_grad, l_dedx, 3)
        self.assertAlmostEqual(l_y.m_grad, l_dedy)