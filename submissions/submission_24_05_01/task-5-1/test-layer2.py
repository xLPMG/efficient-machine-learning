import unittest
import layer2 as l2
import math

class TestLayer2(unittest.TestCase):

    def setUp(self):
        self.w0 = 1
        self.w1 = 2
        self.w2 = 3
        self.x0 = 4
        self.x1 = 5

    def test_forward(self):
        w0 = 1.0
        w1 = 2.0
        w2 = 3.0
        x0 = 4.0
        x1 = 5.0
        # => g = 1 / (1+e^(-(1*4+2*5+3))) = 1 / 1.0000000413994 = 0.9999999586006
        result = 0.9999999586006
        self.assertAlmostEqual(l2.forward(w0,w1,w2,x0,x1),result,6)
    
    def test_backward(self):
        w0 = 1.0
        w1 = 2.0
        w2 = 3.0
        x0 = 4.0
        x1 = 5.0
        
        l_a = w0*x0 + w1*x1 + w2
        l_b = 1 + math.exp( -l_a )
        
        l_dcdb = -l_b ** -2
   
        l_dbda = -math.exp( -l_a )
        l_dcda = l_dcdb * l_dbda

        l_dadw0 = x0
        l_dadx0 = w0
        l_dadw1 = x1
        l_dadx1 = w1
        l_dadw2 = 1

        l_dcdw0 = l_dadw0 * l_dcda
        l_dcdx0 = l_dadx0 * l_dcda
        l_dcdw1 = l_dadw1 * l_dcda
        l_dcdx1 = l_dadx1 * l_dcda
        l_dcdw2 = l_dadw2 * l_dcda

        self.assertAlmostEqual(l2.backward(w0,w1,w2,x0,x1)[0],l_dcdw0,6)
        self.assertAlmostEqual(l2.backward(w0,w1,w2,x0,x1)[1],l_dcdx0,6)
        self.assertAlmostEqual(l2.backward(w0,w1,w2,x0,x1)[2],l_dcdw1,6)
        self.assertAlmostEqual(l2.backward(w0,w1,w2,x0,x1)[3],l_dcdx1,6)
        self.assertAlmostEqual(l2.backward(w0,w1,w2,x0,x1)[4],l_dcdw2,6)

if __name__ == '__main__':
    unittest.main()