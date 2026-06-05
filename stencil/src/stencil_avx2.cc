/*
 * stencil_avx2.cc — 使用 kernel_row_avx2 (逐行 AVX2 + stream + 2x 展开)
 */

#include "stencil.h"
#include "kernel_avx2.h"

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out) {
  kernel_row_avx2(img_in.pixel, img_out.pixel, img_in.width, img_in.height);
}

template void ApplyStencil<float>(ImageClass<float> & img_in, ImageClass<float> & img_out);
