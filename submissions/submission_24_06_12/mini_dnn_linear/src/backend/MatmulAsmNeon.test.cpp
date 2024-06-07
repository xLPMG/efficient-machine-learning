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
  at::Tensor l_x = at::rand( { l_size_n, l_size_c } );
  at::Tensor l_w = at::rand( { l_size_c, l_size_k } );

  // blocking
  // X: nb x cb x bc x bn
  // W: kb x cb x bk x bc
  // Y: kb x nb x bk x bn

  // TODO:
  //   1) derive blocked X and W
  //   2) compute blocked solution through MatmulAsmNeon.forward
  //   3) reverse blocking and verify

}