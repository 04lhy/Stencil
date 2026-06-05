/*
 * stencil.cc — 9-point stencil 优化实现
 *
 * 技术栈：
 *   - OpenMP 多线程并行 (row-based static scheduling)
 *   - AVX2 SIMD: 一次处理 8 个 float，2x 循环展开 (16 元素/迭代)
 *   - FMA: _mm256_fmsub_ps 融合乘减
 *   - 累加式加载: 边 load 边求和，降低寄存器压力，隐藏 load latency
 *   - 非临时写入: _mm256_stream_ps 绕过 cache
 *   - 软件预取: _mm_prefetch 提前加载数据到 L1
 *
 * 性能: ~67ms → 目标 ~30-40ms (2x 展开 + prefetch 预期 30-50% 提升)
 */

#include "stencil.h"
#include <immintrin.h>

#define PREF_DIST 16   // 预取距离 (cache lines ahead)

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out) {

  const int width  = img_in.width;
  const int height = img_in.height;

  P * __restrict in  = img_in.pixel;
  P * __restrict out = img_out.pixel;

  const __m256 v9   = _mm256_set1_ps(9.0f);
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

    // ---- 对齐处理：stream store 要求 32 字节对齐 ----
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

    // ====================================================================
    // SIMD 主循环：2x 展开，每次处理 16 个元素
    //
    // 累加式加载策略：
    //   不保存全部 9 个 load 结果，而是边 load 边累加到 sum 寄存器
    //   仅保留 center (mc) 用于最后 fmsub
    //   寄存器占用: sum0, mc0, sum1, mc1 + 临时 load + 3 常量 = 8 YMM
    //   远低于 16 个 YMM 的上限，不会 spill 到栈
    // ====================================================================
    int j_end_unroll = width - 16;  // 保证 j+15 < width-1 (两个 SIMD lane)

    // 先用 1x 循环处理到 j_end_unroll 对齐到 16
    // (如果 j 已经对齐到 8，则自然对齐到 16)
    for (; j < j_end_unroll; j += 16) {
      // ---- 软件预取：提前加载前方数据到 L1 cache ----
      _mm_prefetch((const char*)(r0 + j + PREF_DIST), _MM_HINT_T0);
      _mm_prefetch((const char*)(r1 + j + PREF_DIST), _MM_HINT_T0);
      _mm_prefetch((const char*)(r2 + j + PREF_DIST), _MM_HINT_T0);

      // ============================================================
      // Batch 0: 元素 j..j+7
      // ============================================================
      __m256 mc0  = _mm256_loadu_ps(r1 + j);          // center (保留给 fmsub)
      __m256 sum0 = _mm256_loadu_ps(r0 + j - 1);      // tl
      sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r0 + j));      // + tc
      sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r0 + j + 1));  // + tr
      sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r1 + j - 1));  // + ml
      sum0 = _mm256_add_ps(sum0, mc0);                          // + center
      sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r1 + j + 1));  // + mr
      sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r2 + j - 1));  // + bl
      sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r2 + j));      // + bc
      sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r2 + j + 1));  // + br

      // ============================================================
      // Batch 1: 元素 j+8..j+15
      //   在计算 batch 0 之前先 load batch 1 的 center，
      //   让 load pipeline 与 batch 0 的 add 链重叠
      // ============================================================
      __m256 mc1  = _mm256_loadu_ps(r1 + j + 8);               // center
      __m256 sum1 = _mm256_loadu_ps(r0 + j + 7);               // tl
      sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r0 + j + 8)); // + tc
      sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r0 + j + 9)); // + tr
      sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r1 + j + 7)); // + ml
      sum1 = _mm256_add_ps(sum1, mc1);                          // + center
      sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r1 + j + 9)); // + mr
      sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r2 + j + 7)); // + bl
      sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r2 + j + 8)); // + bc
      sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r2 + j + 9)); // + br

      // ---- 计算 batch 0: val = 9*center - sum ----
      __m256 val0 = _mm256_fmsub_ps(v9, mc0, sum0);
      val0 = _mm256_max_ps(v0, val0);
      val0 = _mm256_min_ps(v255, val0);

      // ---- 计算 batch 1 ----
      __m256 val1 = _mm256_fmsub_ps(v9, mc1, sum1);
      val1 = _mm256_max_ps(v0, val1);
      val1 = _mm256_min_ps(v255, val1);

      // ---- 非临时写入 (地址已对齐) ----
      _mm256_stream_ps(o + j,      val0);
      _mm256_stream_ps(o + j + 8,  val1);
    }

    // ---- 剩余 SIMD (1x 回退，处理不足 16 但 >= 8 的部分) ----
    for (; j < j_end; j += 8) {
      _mm_prefetch((const char*)(r0 + j + PREF_DIST), _MM_HINT_T0);
      _mm_prefetch((const char*)(r1 + j + PREF_DIST), _MM_HINT_T0);
      _mm_prefetch((const char*)(r2 + j + PREF_DIST), _MM_HINT_T0);

      __m256 mc  = _mm256_loadu_ps(r1 + j);
      __m256 sum = _mm256_loadu_ps(r0 + j - 1);
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(r0 + j));
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(r0 + j + 1));
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(r1 + j - 1));
      sum = _mm256_add_ps(sum, mc);
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(r1 + j + 1));
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(r2 + j - 1));
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(r2 + j));
      sum = _mm256_add_ps(sum, _mm256_loadu_ps(r2 + j + 1));

      __m256 val = _mm256_fmsub_ps(v9, mc, sum);
      val = _mm256_max_ps(v0, val);
      val = _mm256_min_ps(v255, val);

      _mm256_stream_ps(o + j, val);
    }

    // ---- 尾部标量 (j >= j_end 或 j_end 对齐剩余) ----
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
