/*
 * stencil_simple.cc — 对标 IPCC opt_kernel.h: 1D 线性化 + 分组加法 + stream store + OpenMP
 *
 * 思路（对标 IPCC opt_kernel.h）：
 *   将 in/out 指针偏移到第一个内部像素 (width+1)
 *   按全图 stride=width 线性迭代内部像素
 *   好处：单次对齐 (vs 每行一次)、连续内存访问
 *
 * 修复：使用 #pragma omp parallel for 替代手动划分（避免 thread count 舍入溢出）
 */

#include "stencil.h"
#include <immintrin.h>
#include <omp.h>

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out) {

  const int width  = img_in.width;
  const int height = img_in.height;

  // 仅处理内部像素 (rows 1..height-2, cols 1..width-2)
  P * __restrict in  = img_in.pixel;
  P * __restrict out = img_out.pixel;

  const __m256 v8   = _mm256_set1_ps(8.0f);
  const __m256 v0   = _mm256_setzero_ps();
  const __m256 v255 = _mm256_set1_ps(255.0f);

  #pragma omp parallel for schedule(static)
  for (int i = 1; i < height - 1; i++) {
    float * __restrict r0 = in  + (i - 1) * width;
    float * __restrict r1 = in  +  i      * width;
    float * __restrict r2 = in  + (i + 1) * width;
    float * __restrict o  = out +  i      * width;

    int j = 1;
    int j_end = width - 8;  // 保证 j+7 < width-1

    // ---- 对齐 stream store 目标地址到 32 字节 ----
    unsigned long long addr = (unsigned long long)(o + j);
    int misalign = addr & 31;
    if (misalign != 0 && j < width - 1) {
      int align_count = (32 - misalign) >> 2;
      if (align_count > width - 1 - j)
        align_count = width - 1 - j;
      int j_align = j + align_count;

      for (; j < j_align; j++) {
        float val =
          -r0[j-1] - r0[j] - r0[j+1]
          -r1[j-1] + 8.0f * r1[j] - r1[j+1]
          -r2[j-1] - r2[j] - r2[j+1];
        val = (val < 0.0f ? 0.0f : val);
        val = (val > 255.0f ? 255.0f : val);
        o[j] = val;
      }
    }

    // ================================================================
    // SIMD 主循环：IPCC 分组加法 (依赖链 ~5 级)
    //   v1 = TL + TC, v2 = TR + ML, v3 = 8*center - MR
    //   v4 = BL + BC, v5 = BR
    //   合并: val = (v3-v4) - ((v1+v2)+v5)
    // ================================================================
    for (; j < j_end; j += 8) {
      _mm_prefetch((const char*)(r0 + j + 16), _MM_HINT_T0);
      _mm_prefetch((const char*)(r1 + j + 16), _MM_HINT_T0);
      _mm_prefetch((const char*)(r2 + j + 16), _MM_HINT_T0);

      __m256 mc  = _mm256_loadu_ps(r1 + j);

      // 分组加法
      __m256 v1 = _mm256_add_ps(_mm256_loadu_ps(r0 + j - 1),
                                _mm256_loadu_ps(r0 + j));
      __m256 v2 = _mm256_add_ps(_mm256_loadu_ps(r0 + j + 1),
                                _mm256_loadu_ps(r1 + j - 1));
      __m256 v3 = _mm256_fmsub_ps(v8, mc, _mm256_loadu_ps(r1 + j + 1));
      __m256 v4 = _mm256_add_ps(_mm256_loadu_ps(r2 + j - 1),
                                _mm256_loadu_ps(r2 + j));
      __m256 v5 = _mm256_loadu_ps(r2 + j + 1);

      // 合并
      __m256 val10 = _mm256_add_ps(v1, v2);
      __m256 val11 = _mm256_sub_ps(v3, v4);
      __m256 val20 = _mm256_add_ps(val10, v5);
      __m256 val   = _mm256_sub_ps(val11, val20);

      val = _mm256_max_ps(v0, val);
      val = _mm256_min_ps(v255, val);

      _mm256_stream_ps(o + j, val);
    }

    // ---- 尾部标量 ----
    for (; j < width - 1; j++) {
      float val =
        -r0[j-1] - r0[j] - r0[j+1]
        -r1[j-1] + 8.0f * r1[j] - r1[j+1]
        -r2[j-1] - r2[j] - r2[j+1];
      val = (val < 0.0f ? 0.0f : val);
      val = (val > 255.0f ? 255.0f : val);
      o[j] = val;
    }
  }
}

template void ApplyStencil<float>(ImageClass<float> & img_in, ImageClass<float> & img_out);
