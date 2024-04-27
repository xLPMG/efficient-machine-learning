import math

def forward( i_w0,
             i_w1,
             i_w2,
             i_x0,
             i_x1 ):
   l_a = i_w0*i_x0 + i_w1*i_x1 + i_w2
   l_b = 1 + math.exp( -l_a )
   l_c = 1 / l_b
   return l_c

def backward( i_w0,
              i_w1,
              i_w2,
              i_x0,
              i_x1 ):
   l_a = i_w0*i_x0 + i_w1*i_x1 + i_w2
   l_b = 1 + math.exp( -l_a )
   
   l_dcdb = -l_b ** -2
   
   l_dbda = -math.exp( -l_a )
   l_dcda = l_dcdb * l_dbda
   
   l_dadw0 = i_x0
   l_dadx0 = i_w0
   l_dadw1 = i_x1
   l_dadx1 = i_w1
   l_dadw2 = 1
   
   l_dcdw0 = l_dadw0 * l_dcda
   l_dcdx0 = l_dadx0 * l_dcda
   l_dcdw1 = l_dadw1 * l_dcda
   l_dcdx1 = l_dadx1 * l_dcda
   l_dcdw2 = l_dadw2 * l_dcda
   
   return l_dcdw0, l_dcdw1, l_dcdw2, l_dcdx0, l_dcdx1
   