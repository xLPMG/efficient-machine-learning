#include <cstdlib>
#include <ATen/ATen.h>
#include <iostream>
#include <cstdint>

int main()
{
  std::cout << "running the ATen examples" << std::endl;

  float l_data[4 * 2 * 3] = {0.0f, 1.0f, 2.0f,
                             3.0f, 4.0f, 5.0f,

                             6.0f, 7.0f, 8.0f,
                             9.0f, 10.0f, 11.0f,

                             12.0f, 13.0f, 14.0f,
                             15.0f, 16.0f, 17.0f,

                             18.0f, 19.0f, 20.0f,
                             21.0f, 22.0f, 23.0f};

  std::cout << "l_data (ptr): " << l_data << std::endl;

  //////////////////////////////////////////////////////////////////
  std::cout << "############## TASK 2-2-storage-2 and 2-2-storage-3 ##############" << std::endl;
  //////////////////////////////////////////////////////////////////

  at::Tensor l_tensor = at::from_blob(l_data,
                                      {4, 2, 3});

  std::cout << "Tensor   :\n"
            << l_tensor << std::endl;
  std::cout << "Data ptr :" << l_tensor.data_ptr() << std::endl;
  std::cout << "Dtype    :" << l_tensor.dtype() << std::endl;
  std::cout << "Size     :" << l_tensor.sizes() << std::endl;
  std::cout << "Stride   :" << l_tensor.strides() << std::endl;
  std::cout << "Offset   :" << l_tensor.storage_offset() << std::endl;
  std::cout << "Device   :" << l_tensor.device() << std::endl;
  std::cout << "Layout   :" << l_tensor.layout() << std::endl;
  std::cout << "Contig.? :" << l_tensor.is_contiguous() << std::endl;

  //////////////////////////////////////////////////////////////////
  std::cout << "############## TASK 2-2-storage-4 ##############" << std::endl;
  //////////////////////////////////////////////////////////////////

  // RAW C POINTER
  // first row T1
  l_data[0] = 54;
  l_data[1] = 55;
  l_data[2] = 56;

  l_data[15] = 57;
  l_data[16] = 58;
  l_data[17] = 59;

  std::cout << "Tensor after editing using c pointer:\n"
            << l_tensor << std::endl;

  // TENSOR
  // first row T1
  l_tensor[0][0][0] = 94;
  l_tensor[0][0][1] = 95;
  l_tensor[0][0][2] = 96;

  // second row T3
  l_tensor[2][1][0] = 97;
  l_tensor[2][1][1] = 98;
  l_tensor[2][1][2] = 99;

  std::cout << "Tensor after editing using aten tensor:\n"
            << l_tensor << std::endl;

  //////////////////////////////////////////////////////////////////
  std::cout << "############## TASK 2-2-storage-5 ##############" << std::endl;
  //////////////////////////////////////////////////////////////////
  at::Tensor l_view = l_tensor.select(1, 1);
  std::cout << "l_view\n"
            << l_view << std::endl;

  /*
  Since l_view starts at index 1, it skips the first 3 elements.
  If we add 3 to the l_tensor pointer, it should point to the
  same location as the l_view pointer. Because the pointer is
  originally of type void, we should cast it to the correct type
  to allow for pointer arithmetic.
  */
  std::cout << "l_tensor.data_ptr() + 3 : " << (float *)l_tensor.data_ptr() + 3 << std::endl;
  std::cout << "l_view.data_ptr()       : " << l_view.data_ptr() << std::endl;

  //////////////////////////////////////////////////////////////////
  std::cout << "############## TASK 2-2-storage-6 ##############" << std::endl;
  //////////////////////////////////////////////////////////////////
  at::Tensor l_cont = l_view.contiguous();

  std::cout << "Both tensors should look the same:" << std::endl;
  std::cout << "l_view\n"
            << l_view << std::endl;
  std::cout << "l_cont\n"
            << l_cont << std::endl;

  std::cout << "However their difference lies in the memory representation. contiguous() allocated new and contiguous memory for the l_cont tensor:" << std::endl;

  std::cout << "l_view.data_ptr()\n"
            << l_view.data_ptr() << std::endl;
  std::cout << "l_cont.data_ptr()\n"
            << l_cont.data_ptr() << std::endl;

  std::cout << "Jumping 3 memory addresses yields:" << std::endl;

  float *l_viewPtr = (float *)l_view.data_ptr() + 3;
  float *l_contPtr = (float *)l_cont.data_ptr() + 3;

  std::cout << "value at l_view.data_ptr() + 3 = "
            << *l_viewPtr << std::endl;
  std::cout << "value at l_cont.data_ptr() + 3 = "
            << *l_contPtr << std::endl;

  std::cout << "That is because in l_view, the original tensor memory is still intact and after 3 4 5 still comes 6 7 8." << std::endl;

  //////////////////////////////////////////////////////////////////
  std::cout << "############## TASK 2-2-operations ##############" << std::endl;
  //////////////////////////////////////////////////////////////////

  at::Tensor l_A = at::rand({16, 4});
  at::Tensor l_B = at::rand({4, 16});

  std::cout << "===========================================================================" << std::endl;
  std::cout << "Matrix A\n"
            << l_A << std::endl;
  std::cout << "Matrix B\n"
            << l_B << std::endl;

  std::cout << "===========================================================================" << std::endl;
  at::Tensor l_productAB = at::matmul(l_A, l_B);
  std::cout << "Matrix multiplication of A and B\n"
            << l_productAB << std::endl;

  at::Tensor l_T_0 = at::rand({16, 4, 2});
  at::Tensor l_T_1 = at::rand({16, 2, 4});     

  std::cout << "===========================================================================" << std::endl;
  std::cout << "Tensor T0\n"
            << l_T_0 << std::endl;
  std::cout << "Tensor T1\n"
            << l_T_1 << std::endl;     

  std::cout << "===========================================================================" << std::endl;
  at::Tensor l_productT0T1 = at::bmm(l_T_0, l_T_1);
  std::cout << "Batched tensor multiplication of T0 and T1\n"
            << l_productT0T1 << std::endl;

  std::cout << "finished running ATen examples" << std::endl;

  return EXIT_SUCCESS;
}
