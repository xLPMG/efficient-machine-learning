import torch

class SimpleDataSet( torch.utils.data.Dataset ):
  def __init__( self,
                i_length ):
    self.m_length = i_length

  def __len__( self ):
    return self.m_length

  def __getitem__( self,
                   i_idx ):
    return i_idx*10
