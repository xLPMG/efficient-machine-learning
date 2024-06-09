#include "MatmulAsmNeon.h"

extern "C"
{
  void gemm_asm_asimd_64_64_64(float const *i_a,
                               float const *i_b,
                               float *io_c);
}

at::Tensor mini_dnn::backend::MatmulAsmNeon::forward(at::Tensor i_x,
                                                     at::Tensor i_w)
{
  // get involved sizes
  Matmul::Sizes l_sizes = Matmul::getSizes(i_x,
                                           i_w);

  MINI_DNN_CHECK_EQ(l_sizes.bn, 64);
  MINI_DNN_CHECK_EQ(l_sizes.bk, 64);
  MINI_DNN_CHECK_EQ(l_sizes.bc, 64);

  // prepare data for GEMM calls
  at::Tensor l_y = at::zeros({l_sizes.kb, l_sizes.nb, l_sizes.bk, l_sizes.bn});

  c10::IntArrayRef l_strides_a = i_x.strides();
  c10::IntArrayRef l_strides_b = i_w.strides();
  c10::IntArrayRef l_strides_c = l_y.strides();

  float *l_ptr_a = (float *)i_x.data_ptr();
  float *l_ptr_b = (float *)i_w.data_ptr();
  float *l_ptr_c = (float *)l_y.data_ptr();

  // loop over outer dimensions
#pragma omp parallel for collapse(2)
  for (int64_t l_kb = 0; l_kb < l_sizes.kb; l_kb++)
  {
    for (int64_t l_nb = 0; l_nb < l_sizes.nb; l_nb++)
    {
      float *l_ptr_c_offset = l_ptr_c + l_kb * l_strides_c[0] + l_nb * l_strides_c[1];

      for (int64_t l_cb = 0; l_cb < l_sizes.cb; l_cb++)
      {
        float *l_ptr_a_offset = l_ptr_a + l_nb * l_strides_a[0] + l_cb * l_strides_a[1];

        float *l_ptr_b_offset = l_ptr_b + l_kb * l_strides_b[0] + l_cb * l_strides_b[1];

        gemm_asm_asimd_64_64_64(l_ptr_a_offset, l_ptr_b_offset, l_ptr_c_offset);
      }
    }
  }

  return l_y;
}