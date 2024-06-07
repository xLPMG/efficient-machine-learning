#include <catch2/catch.hpp>
#include "MatmulAsmNeon.h"

TEST_CASE( "Tests the Matmul forward operator through a Neon assembly kernel.",
           "[matmul][asm_neon][forward]" ) {
  // BLAS -> Deep Learning:
  // M: N (batch size)
  // N: K (out features)
  // K: C (in features)

  // sizes of the input
  int64_t l_size_n = 128;
  int64_t l_size_k = 256;
  int64_t l_size_c = 512;

  int64_t l_size_bn = 64;
  int64_t l_size_bk = 64;
  int64_t l_size_bc = 64;

  int64_t l_size_nb = l_size_n / l_size_bn;
  int64_t l_size_kb = l_size_k / l_size_bk;
  int64_t l_size_cb = l_size_c / l_size_bc;

// construct input tensors
  at::Tensor l_x = at::rand({l_size_n, l_size_c});
  at::Tensor l_w = at::rand({l_size_c, l_size_k});

  // blocking
  // X: nb x cb x bc x bn
  // W: kb x cb x bk x bc
  // Y: kb x nb x bk x bn

  // derive blocked X and W
  at::Tensor l_x_blocked = l_x.view({l_size_nb, l_size_bn, l_size_cb, l_size_bc}).permute({0, 2, 3, 1}).contiguous();
  at::Tensor l_w_blocked = l_w.view({l_size_cb, l_size_bc, l_size_kb, l_size_bk}).permute({2, 0, 3, 1}).contiguous();

  mini_dnn::backend::MatmulAsmNeon l_matmul_asm_neon;
  at::Tensor l_y_blocked = l_matmul_asm_neon.forward(l_x_blocked, l_w_blocked);

  // reverse blocking
  at::Tensor l_y = l_y_blocked.view({l_size_kb * l_size_nb, l_size_bk * l_size_bn}).contiguous();

  // compute reference
  at::Tensor l_reference = at::matmul(l_x, l_w);

  // check solution
  REQUIRE(at::allclose(l_y, l_reference));

}