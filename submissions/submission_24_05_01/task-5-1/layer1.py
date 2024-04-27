def forward( i_x,
             i_y,
             i_z ):
   l_a = i_y + i_z
   l_b = i_x * l_a
   return l_b

def backward( i_x,
              i_y,
              i_z ):
  l_a = i_y + i_z
  l_dbda = i_x
  l_dbdx = l_a
  l_dady = 1
  l_dadz = 1
  l_dbdy = l_dbda * l_dady
  l_dbdz = l_dbda * l_dadz
  return l_dbdx, l_dbdy, l_dbdz