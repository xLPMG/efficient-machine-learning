#include <cstdlib>
#include <ATen/ATen.h>
#include <iostream>

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
  std::cout << "############## TASK 2-2-2 and 2-2-3 ##############" << std::endl;
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
  std::cout << "############## TASK 2-2-4 ##############" << std::endl;
  //////////////////////////////////////////////////////////////////

  std::cout << "Manipulating tensor to skip first 2 rows" << std::endl;
  l_tensor.set_(l_tensor, 6, {3, 2, 3});
  std::cout << "Tensor   :\n"
            << l_tensor << std::endl;

  // l_tensor = at::from_blob(l_data + 3,
  //                          {2,3});
  // std::cout << "Tensor   :" << l_tensor << std::endl;

  //TODO: finish

  //////////////////////////////////////////////////////////////////
  std::cout << "############## TASK 2-2-5 ##############" << std::endl;
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
  std::cout << "############## TASK 2-2-6 ##############" << std::endl;
  //////////////////////////////////////////////////////////////////
  at::Tensor l_cont = l_view.contiguous();

  std::cout << "l_view\n"
            << l_view << std::endl;
  std::cout << "l_cont\n"
            << l_cont << std::endl;

  std::cout << "finished running ATen examples" << std::endl;

  return EXIT_SUCCESS;
}
