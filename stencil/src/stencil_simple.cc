/*
 * stencil_simple.cc — 对标 IPCC opt_kernel.h: 1D 线性化 + stream store + OpenMP
 *
 * 思路（对标 IPCC opt_kernel.h）：
 *   将 in/out 指针偏移到第一个内部像素 (width+1)
 *   按全图 stride=width 线性迭代内部 + 部分边界像素
 *   好处：单次对齐、连续内存访问、stream store
 */

#include "stencil.h"
#include <immintrin.h>
#include <omp.h>

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out) {

  const int width  = img_in.width;
  const int height = img_in.height;

  // 总元素数 (从 width+1 开始，到 width*height-width-2 结束)
  // = 全图 - 首行 - 尾行 - 首元素 - 尾元素
  int counts = width * height - 2 * (width + 1);

  // 指针偏移到第一个内部像素 (1,1)
  P * __restrict in_ptr  = img_in.pixel  + width + 1;
  P * __restrict out_ptr = img_out.pixel + width + 1;

  // ---- 对齐 stream store 目标地址到 32 字节 ----
  int align_offset = 0;
  unsigned long long addr = (unsigned long long)(out_ptr);
  int misalign = addr & 31;
  if (misalign != 0) {
    align_offset = (32 - misalign) >> 2;  // 补齐所需的 float 数
    if (align_offset > counts) align_offset = counts;
  }

  // 对齐 prolog（标量）
  for (int k = 0; k < align_offset; k++) {
    float val =
      -in_ptr[k - width - 1] - in_ptr[k - width] - in_ptr[k - width + 1]
      -in_ptr[k - 1]         + 8.0f * in_ptr[k]   - in_ptr[k + 1]
      -in_ptr[k + width - 1] - in_ptr[k + width] - in_ptr[k + width + 1];
    val = (val < 0.0f ? 0.0f : val);
    val = (val > 255.0f ? 255.0f : val);
    out_ptr[k] = val;
  }

  int remaining = counts - align_offset;
  // 对齐到 8 (SIMD lane)
  int vec_count = remaining & ~7;

  const __m256 v8   = _mm256_set1_ps(8.0f);
  const __m256 v0   = _mm256_setzero_ps();
  const __m256 v255 = _mm256_set1_ps(255.0f);

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();

    // 每个线程处理的元素数（对齐到 8）
    int per_thread = (vec_count / nth) & ~7;
    int start = align_offset + tid * per_thread;
    int end = (tid == nth - 1)
                ? align_offset + vec_count
                : start + per_thread;

    if (start < end) {
      for (int k = start; k < end; k += 8) {
        // 累加式加载：边 load 边 sum（降低寄存器压力）
        __m256 mc  = _mm256_loadu_ps(&in_ptr[k]);
        __m256 sum = _mm256_loadu_ps(&in_ptr[k - width - 1]);
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(&in_ptr[k - width]));
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(&in_ptr[k - width + 1]));
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(&in_ptr[k - 1]));
        sum = _mm256_add_ps(sum, mc);
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(&in_ptr[k + 1]));
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(&in_ptr[k + width - 1]));
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(&in_ptr[k + width]));
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(&in_ptr[k + width + 1]));

        __m256 val = _mm256_fmsub_ps(v8, mc, sum);
        val = _mm256_max_ps(v0, val);
        val = _mm256_min_ps(v255, val);

        _mm256_stream_ps(&out_ptr[k], val);
      }
    }
  }

  // 尾部标量（不足 8 个元素）
  int tail_start = align_offset + vec_count;
  for (int k = tail_start; k < counts; k++) {
    float val =
      -in_ptr[k - width - 1] - in_ptr[k - width] - in_ptr[k - width + 1]
      -in_ptr[k - 1]         + 8.0f * in_ptr[k]   - in_ptr[k + 1]
      -in_ptr[k + width - 1] - in_ptr[k + width] - in_ptr[k + width + 1];
    val = (val < 0.0f ? 0.0f : val);
    val = (val > 255.0f ? 255.0f : val);
    out_ptr[k] = val;
  }
}

template void ApplyStencil<float>(ImageClass<float> & img_in, ImageClass<float> & img_out);
