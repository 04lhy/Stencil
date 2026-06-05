/*
 * stencil_block.cc — 使用 kernel_block_avx2 (读 6 行算 4 行 + stream store + mask tail)
 */

#include "stencil.h"
#include "kernel_avx2.h"

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out) {
  kernel_block_avx2(img_in.pixel, img_out.pixel, img_in.width, img_in.height);
}

template void ApplyStencil<float>(ImageClass<float> & img_in, ImageClass<float> & img_out);
