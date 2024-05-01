import unittest
import layer2 as l2

class TestLayer2(unittest.TestCase):

    def setUp(self):
        self.w0 = 1
        self.w1 = 2
        self.w2 = 3
        self.x0 = 4
        self.x1 = 5

    def test_forward(self):
        # Test case:
        # w0 = 1
        # w1 = 2
        # w2 = 3
        # x0 = 4
        # x1 = 5
        # => g = 1 / (1+e^(-(1*4+2*5+3))) = 1 / 1.0000000413994 = 0.9999999586006
        result = 0.9999999586006
        self.assertAlmostEqual(l2.forward(1,2,3,4,5),result,6)
    
    def test_backward(self):
        # Test case:
        # w0 = 1
        # w1 = 2
        # w2 = 3
        # x0 = 4
        # x1 = 5
        self.assertAlmostEqual(l2.backward(1,2,3,4,5)[0],0.00000016559749504,6)
        self.assertAlmostEqual(l2.backward(1,2,3,4,5)[1],0.00000020699686880,6)
        self.assertAlmostEqual(l2.backward(1,2,3,4,5)[2],0.00000004139937376,6)
        self.assertAlmostEqual(l2.backward(1,2,3,4,5)[3],0.00000004139937376,6)
        self.assertAlmostEqual(l2.backward(1,2,3,4,5)[4],0.00000008279874752,6)

if __name__ == '__main__':
    unittest.main()