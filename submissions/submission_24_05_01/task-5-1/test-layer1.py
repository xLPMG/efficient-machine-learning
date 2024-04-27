import unittest
import layer1 as l1

class TestLayer1(unittest.TestCase):

    def setUp(self):
        self.x1 = 2
        self.y1 = 3
        self.z1 = 5

    def test_forward(self):
        # f = x * ( y + z )
        result = self.x1*(self.y1+self.z1)
        self.assertEqual(l1.forward(self.x1, 
                                    self.y1, 
                                    self.z1), 
                                    result)
        
    def test_backward(self):
        # given derivatives are y+z, x and x
        result = (self.y1+self.z1, self.x1, self.x1)
        self.assertEqual(l1.backward(self.x1, 
                                     self.y1, 
                                     self.z1), 
                                     result)

if __name__ == '__main__':
    unittest.main()