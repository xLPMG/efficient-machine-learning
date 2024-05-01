from . import context
from . import functions

class Node:
  ## Initializes a node of the computation graph.
  # @param optional value used for leaf nodes.
  def __init__( self,
                i_value = None ):
    self.m_value = i_value
    self.m_grad = 0
    self.m_grad_fn = functions.Nop.backward
    self.m_children = []
    self.m_ctx = context.Context()

  ## String representation of a node.
  # @return newline separated string with value, grad and grad_fn.
  def __str__( self ):
    l_string  = "node:\n"
    l_string += "  value: "   + str( self.m_value ) + "\n"
    l_string += "  grad: "    + str( self.m_grad ) + "\n"
    l_string += "  grad_fn: " + str( self.m_grad_fn )
    return l_string

  ## Backward pass of the node.
  # @param i_grad  grad w.r.t. to output of forward pass.
  def backward( self,
                i_grad ):
    self.m_grad += i_grad
    l_grad_children = self.m_grad_fn( self.m_ctx,
                                      i_grad )

    if len(self.m_children) > 1:
      for l_ch in range( len(self.m_children) ):
        self.m_children[l_ch].backward( l_grad_children[l_ch] )
    elif len(self.m_children) > 0:
      self.m_children[0].backward( l_grad_children )

  ## Zeroes the grad of the node and all children.
  def zero_grad( self ):
    self.m_grad = 0
    for l_ch in self.m_children:
      l_ch.zero_grad()

  ## Returns a new node which represents the addition of the two input nodes.
  # @param self first input node.
  # @param i_other second input node.
  # @return node representing the addition.
  def __add__( self,
               i_other ):
    l_node = Node()
    l_node.m_grad_fn = functions.Add.backward
    l_node.m_children = [self, i_other]
    l_node.m_value = functions.Add.forward( l_node.m_ctx,
                                            self.m_value,
                                            i_other.m_value )
    return l_node
  
  ## Returns a new node which represents the subtraction of the two input nodes.
  # @param self first input node.
  # @param i_other second input node.
  # @return node representing the addition.
  def __sub__( self,
               i_other ):
    l_node = Node()
    l_node.m_grad_fn = functions.Add.backward
    l_node.m_children = [self, i_other]
    l_node.m_value = functions.Add.forward( l_node.m_ctx,
                                            self.m_value,
                                            - i_other.m_value )
    return l_node

  ## Returns a new node which represents the multiplication of the two input nodes.
  # @param self first input node.
  # @param i_other second input node.
  # @return node representing the multiplication.
  def __mul__( self,
               i_other ):
    l_node = Node()
    l_node.m_grad_fn = functions.Mul.backward
    l_node.m_children = [self, i_other]
    l_node.m_value = functions.Mul.forward( l_node.m_ctx,
                                            self.m_value,
                                            i_other.m_value )
    return l_node
    
  ## Returns a new node which represents the scalar reciprocal of an input node.
  # @param self input node.
  # @return node representing the reciprocal.
  def reciprocal( self ):
    l_node = Node()
    l_node.m_grad_fn = functions.Reciprocal.backward
    l_node.m_children = [self]
    l_node.m_value = functions.Reciprocal.forward( l_node.m_ctx,
                                                   self.m_value )
    return l_node
  
  ## Returns a new node which represents e^x on an input node.
  # @param self input node.
  # @return node representing e^x.
  def exp( self ):
    l_node = Node()
    l_node.m_grad_fn = functions.Exp.backward
    l_node.m_children = [self]
    l_node.m_value = functions.Exp.forward( l_node.m_ctx,
                                                   self.m_value )
    return l_node
  
  ## Returns a new node which represents sin() operation on an input node.
  # @param self input node.
  # @return node representing sin().
  def sin( self ):
    l_node = Node()
    l_node.m_grad_fn = functions.Sin.backward
    l_node.m_children = [self]
    l_node.m_value = functions.Sin.forward( l_node.m_ctx,
                                            self.m_value )
    return l_node
  
  ## Returns a new node which represents cos() operation on an input node.
  # @param self input node.
  # @return node representing cos().
  def cos( self ):
    l_node = Node()
    l_node.m_grad_fn = functions.Cos.backward
    l_node.m_children = [self]
    l_node.m_value = functions.Cos.forward( l_node.m_ctx,
                                            self.m_value )
    return l_node