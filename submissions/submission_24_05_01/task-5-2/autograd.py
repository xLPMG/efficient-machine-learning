import numpy
import eml.autograd.node

#
# Example 1: b = a + b
#
print()
print( "*** Example 1: b = a + b ***" )
l_a = eml.autograd.node.Node(2.0)
l_b = eml.autograd.node.Node(3.0)
l_c = l_a + l_b

l_c.backward( 1.0 )
print( l_c )
print( l_b )
print( l_a )

#
# Example 2: c = a + 2.0 * b
#
print()
print( "*** Example 2: c = a + 2.0 * b ***" )
l_a = eml.autograd.node.Node(5.0)
l_b = eml.autograd.node.Node(4.0)
l_c = l_a + eml.autograd.node.Node(2.0) * l_b

l_c.backward( 1.0 )
print( l_c )
print( l_b )
print( l_a )

#
# Example 3: b = 2*a^a
#
print()
print( "*** Example 3: b = a^2 ***" )
l_a = eml.autograd.node.Node(3.0)
l_b =  eml.autograd.node.Node(2.0) * l_a * l_a

l_b.backward( 1.0 )
print( l_a )