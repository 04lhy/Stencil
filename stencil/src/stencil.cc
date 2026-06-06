/*
 * stencil.cc — 9-point stencil 优化实现
 *
 * 技术栈：
 *   - OpenMP 多线程并行 (row-based static scheduling)
 *   - AVX2 SIMD: 一次处理 8 个 float
 *   - IPCC 分组加法: 9 像素分入 4 独立子树, 依赖链 ~5 级 (vs 顺序累加 ~11 级)
 *   - FMA: _mm256_fmsub_ps 融合乘减
 *   - 非临时写入: _mm256_stream_ps 绕过 cache
 *   - 软件预取: _mm_prefetch 提前加载数据到 L1
 *
 * 性能: ~87ms on i9-13900H (P-core 12T close)
 */

#include "stencil.h"
#include <immintrin.h>

#define PREF_DIST  16   // 预取距离 (cache lines ahead)

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out) {

  const int width  = img_in.width;
  const int height = img_in.height;

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
    int j_end = width - 8;  // 最后一个 SIMD 起始位置 (保证 j+7 < width-1)

    // ---- 对齐 stream store 目标地址到 32 字节 ----
    unsigned long long addr = (unsigned long long)(o + j);
    int misalign = addr & 31;
    if (misalign != 0 && j < width - 1) {
      int align_count = (32 - misalign) >> 2;  // 补齐到 32 字节所需的 float 数
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
    //
    //   val1 = TL + TC           (组1: 上排左+中)
    //   val2 = TR + ML           (组2: 上排右+中排左)
    //   val3 = 8*center - MR     (组3: fmsub 中心-中排右)
    //   val4 = BL + BC           (组4: 下排左+中)
    //   val5 = BR                (组5: 下排右, 仅load)
    //
    //   合并:   val10 = val1 + val2     (TL+TC+TR+ML)
    //           val11 = val3 - val4     (8*center-MR-BL-BC)
    //           val20 = val10 + val5    (TL+TC+TR+ML+BR)
    //           val   = val11 - val20   (8*center - sum_of_8_neighbors)
    //
    // 对比顺序累加 (9 级加法依赖链 → 5 级合并树)
    // ================================================================
    for (; j < j_end; j += 8) {
      // ---- 软件预取 ----
      _mm_prefetch((const char*)(r0 + j + PREF_DIST), _MM_HINT_T0);
      _mm_prefetch((const char*)(r1 + j + PREF_DIST), _MM_HINT_T0);
      _mm_prefetch((const char*)(r2 + j + PREF_DIST), _MM_HINT_T0);

      __m256 mc  = _mm256_loadu_ps(r1 + j);

      // 分组加法 (4 条独立加法链并行)
      __m256 v1 = _mm256_add_ps(_mm256_loadu_ps(r0 + j - 1),
                                _mm256_loadu_ps(r0 + j));
      __m256 v2 = _mm256_add_ps(_mm256_loadu_ps(r0 + j + 1),
                                _mm256_loadu_ps(r1 + j - 1));
      __m256 v3 = _mm256_fmsub_ps(v8, mc, _mm256_loadu_ps(r1 + j + 1));
      __m256 v4 = _mm256_add_ps(_mm256_loadu_ps(r2 + j - 1),
                                _mm256_loadu_ps(r2 + j));
      __m256 v5 = _mm256_loadu_ps(r2 + j + 1);

      // 合并阶段 (3 级加法树)
      __m256 val10 = _mm256_add_ps(v1, v2);
      __m256 val11 = _mm256_sub_ps(v3, v4);
      __m256 val20 = _mm256_add_ps(val10, v5);
      __m256 val   = _mm256_sub_ps(val11, val20);

      // clamp [0, 255]
      val = _mm256_max_ps(v0, val);
      val = _mm256_min_ps(v255, val);

      // 非临时写入 (绕过 cache，地址已对齐)
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
