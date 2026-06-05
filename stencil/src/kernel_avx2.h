/*
 * kernel_avx2.h — AVX2 + OpenMP 优化核心
 *
 * 技术栈：
 *   - AVX2 SIMD: _mm256 一次处理 8 个 float
 *   - FMA: _mm256_fmsub_ps 融合乘减
 *   - 累加式加载: 边 load 边 sum，降低寄存器压力，隐藏 load latency
 *   - 非临时写入: _mm256_stream_ps 绕过 cache (需 32 字节对齐)
 *   - 软件预取: _mm_prefetch
 *
 * 版本说明：
 *   - kernel_row_avx2:  逐行 AVX2 + stream store + 2x 展开 + prefetch
 *                        适用于超宽图像 (行数据在 L2 cache 内)
 *   - kernel_block_avx2: 读 6 行算 4 行，数据复用 (33% 更少 load)
 *                        适用于窄图像 (行数据超出 cache)
 */

#ifndef __KERNEL_AVX2_H__
#define __KERNEL_AVX2_H__

#include <immintrin.h>
#include <omp.h>

#define K_SIMD_LANE    8
#define K_BATCH_SIZE   4
#define K_BATCH_GAP    6     // 读 6 行算 4 行 (各重叠行共享)
#define K_PREF_DIST    16

// ============================================================================
// 版本 A：逐行 AVX2 + stream store + 2x 展开 + prefetch
//
// 适用场景：超宽图像 (行宽使得 3 行能放入 L2 cache)
// 每行独立处理，OpenMP 负载均衡最优
// ============================================================================
inline void kernel_row_avx2(float * __restrict in, float * __restrict out,
                            int width, int height)
{
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
        int j_end = width - K_SIMD_LANE;    // j + 7 < width - 1

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

        // ---- SIMD 主循环：2x 展开 (16 元素/迭代) ----
        int j_end_2x = width - (K_SIMD_LANE * 2);

        for (; j < j_end_2x; j += K_SIMD_LANE * 2) {
            _mm_prefetch((const char*)(r0 + j + K_PREF_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)(r1 + j + K_PREF_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)(r2 + j + K_PREF_DIST), _MM_HINT_T0);

            // Batch 0: j..j+7
            __m256 mc0  = _mm256_loadu_ps(r1 + j);
            __m256 sum0 = _mm256_loadu_ps(r0 + j - 1);
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r0 + j));
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r0 + j + 1));
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r1 + j - 1));
            sum0 = _mm256_add_ps(sum0, mc0);
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r1 + j + 1));
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r2 + j - 1));
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r2 + j));
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(r2 + j + 1));

            // Batch 1: j+8..j+15
            __m256 mc1  = _mm256_loadu_ps(r1 + j + 8);
            __m256 sum1 = _mm256_loadu_ps(r0 + j + 7);
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r0 + j + 8));
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r0 + j + 9));
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r1 + j + 7));
            sum1 = _mm256_add_ps(sum1, mc1);
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r1 + j + 9));
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r2 + j + 7));
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r2 + j + 8));
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(r2 + j + 9));

            __m256 val0 = _mm256_min_ps(v255, _mm256_max_ps(v0,
                           _mm256_fmsub_ps(v9, mc0, sum0)));
            __m256 val1 = _mm256_min_ps(v255, _mm256_max_ps(v0,
                           _mm256_fmsub_ps(v9, mc1, sum1)));

            _mm256_stream_ps(o + j,     val0);
            _mm256_stream_ps(o + j + 8, val1);
        }

        // 剩余 1x SIMD
        for (; j < j_end; j += K_SIMD_LANE) {
            _mm_prefetch((const char*)(r0 + j + K_PREF_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)(r1 + j + K_PREF_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)(r2 + j + K_PREF_DIST), _MM_HINT_T0);

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

            __m256 val = _mm256_min_ps(v255, _mm256_max_ps(v0,
                           _mm256_fmsub_ps(v9, mc, sum)));
            _mm256_stream_ps(o + j, val);
        }

        // 尾部标量
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


// ============================================================================
// 版本 B：分块 AVX2 — 读 6 行算 4 行 + 寄存器复用
//
// 原理：
//   输出 row_i 需要读 row_{i-1}, row_i, row_{i+1}
//   读 row_0..row_5 共 6 行，计算 row_1..row_4 共 4 行
//   每个输入行被 3 个相邻输出行共享 → 减少 33% 的 load
//
// 修复要点（对标 IPCC kernel_r6c4_avx2.h）：
//   - 行和预计算 + 寄存器复用流水线
//   - 正确处理 partial block (actual_bs < 4)
//   - mask load/store 处理尾部（避免标量回退开销）
//   - 注意：block kernel 使用 storeu 而非 stream store
//     (多行同时写无法满足统一的 32 字节对齐要求)
// ============================================================================
inline void kernel_block_avx2(float * __restrict in, float * __restrict out,
                               int width, int height)
{
    const int BS = K_BATCH_SIZE;   // 4
    const int GS = K_BATCH_GAP;    // 6
    const int SL = K_SIMD_LANE;    // 8

    const __m256 v9   = _mm256_set1_ps(9.0f);
    const __m256 v0   = _mm256_setzero_ps();
    const __m256 v255 = _mm256_set1_ps(255.0f);

    #pragma omp parallel for schedule(static)
    for (int bi = 1; bi < height - 1; bi += BS) {
        // actual_bs: 本块实际要计算的输出行数 (1..4)
        int actual_bs = (bi + BS <= height - 1) ? BS : (height - 1 - bi);
        if (actual_bs < 1) continue;

        // 输入行指针 (GS=6 行，从 bi-1 开始)
        float *r[6] = {nullptr};
        for (int k = 0; k < GS && (bi - 1 + k) < height; k++)
            r[k] = in + (bi - 1 + k) * width;

        // 输出行指针
        float *o[4] = {nullptr};
        for (int k = 0; k < actual_bs; k++)
            o[k] = out + (bi + k) * width;

        int j_end = width - SL;  // j + 7 < width - 1

        for (int j = 1; j < j_end; j += SL) {
            // ---- 软件预取 ----
            _mm_prefetch((const char*)(r[0] + j + K_PREF_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)(r[1] + j + K_PREF_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)(r[2] + j + K_PREF_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)(r[3] + j + K_PREF_DIST), _MM_HINT_T0);
            if (actual_bs >= 3) {
                _mm_prefetch((const char*)(r[4] + j + K_PREF_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(r[5] + j + K_PREF_DIST), _MM_HINT_T0);
            }

            // ============================================================
            // Phase 1: 加载 r0-r3，计算 o[0] 和 o[1]
            //   o[0] (row bi):   center=r1c, sum=s0+s1+s2 (rows bi-1,bi,bi+1)
            //   o[1] (row bi+1): center=r2c, sum=s1+s2+s3 (rows bi,bi+1,bi+2)
            // ============================================================

            // s0 = r0 行三列和
            __m256 s0 = _mm256_loadu_ps(r[0] + j - 1);
            s0 = _mm256_add_ps(s0, _mm256_loadu_ps(r[0] + j));
            s0 = _mm256_add_ps(s0, _mm256_loadu_ps(r[0] + j + 1));

            // s1 = r1 行三列和 (保留 center for o1)
            __m256 r1c = _mm256_loadu_ps(r[1] + j);
            __m256 s1  = _mm256_loadu_ps(r[1] + j - 1);
            s1 = _mm256_add_ps(s1, r1c);
            s1 = _mm256_add_ps(s1, _mm256_loadu_ps(r[1] + j + 1));

            // s2 = r2 行三列和 (保留 center for o2)
            __m256 r2c = _mm256_loadu_ps(r[2] + j);
            __m256 s2  = _mm256_loadu_ps(r[2] + j - 1);
            s2 = _mm256_add_ps(s2, r2c);
            s2 = _mm256_add_ps(s2, _mm256_loadu_ps(r[2] + j + 1));

            // s3 = r3 行三列和
            __m256 r3c = _mm256_loadu_ps(r[3] + j);
            __m256 s3  = _mm256_loadu_ps(r[3] + j - 1);
            s3 = _mm256_add_ps(s3, r3c);
            s3 = _mm256_add_ps(s3, _mm256_loadu_ps(r[3] + j + 1));

            // o[0]: center=r1c, sum=s0+s1+s2
            __m256 sum1 = _mm256_add_ps(_mm256_add_ps(s0, s1), s2);
            __m256 val1 = _mm256_min_ps(v255, _mm256_max_ps(v0,
                           _mm256_fmsub_ps(v9, r1c, sum1)));
            _mm256_storeu_ps(o[0] + j, val1);

            // o[1]: center=r2c, sum=s1+s2+s3
            __m256 sum2 = _mm256_add_ps(_mm256_add_ps(s1, s2), s3);
            __m256 val2 = _mm256_min_ps(v255, _mm256_max_ps(v0,
                           _mm256_fmsub_ps(v9, r2c, sum2)));
            _mm256_storeu_ps(o[1] + j, val2);

            // ============================================================
            // Phase 2: 加载 r4,r5，复用 s2,s3 计算 o[2] 和 o[3]
            //
            // actual_bs >= 3: 需要计算 o[2] (row bi+2)
            //   o[2]: center=r3c, sum=s2+s3+s4 (rows bi+1,bi+2,bi+3)
            //
            // actual_bs == 4: 还需要计算 o[3] (row bi+3)
            //   o[3]: center=r4c, sum=s3+s4+s5 (rows bi+2,bi+3,bi+4)
            // ============================================================
            if (actual_bs >= 3) {
                // s4 = r4 行三列和
                __m256 r4c = _mm256_loadu_ps(r[4] + j);
                __m256 s4  = _mm256_loadu_ps(r[4] + j - 1);
                s4 = _mm256_add_ps(s4, r4c);
                s4 = _mm256_add_ps(s4, _mm256_loadu_ps(r[4] + j + 1));

                // o[2]: center=r3c, sum=s2+s3+s4
                __m256 sum3 = _mm256_add_ps(_mm256_add_ps(s2, s3), s4);
                __m256 val3 = _mm256_min_ps(v255, _mm256_max_ps(v0,
                               _mm256_fmsub_ps(v9, r3c, sum3)));
                _mm256_storeu_ps(o[2] + j, val3);

                if (actual_bs == 4) {
                    // s5 = r5 行三列和
                    __m256 s5 = _mm256_loadu_ps(r[5] + j - 1);
                    s5 = _mm256_add_ps(s5, _mm256_loadu_ps(r[5] + j));
                    s5 = _mm256_add_ps(s5, _mm256_loadu_ps(r[5] + j + 1));

                    // o[3]: center=r4c, sum=s3+s4+s5
                    __m256 sum4 = _mm256_add_ps(_mm256_add_ps(s3, s4), s5);
                    __m256 val4 = _mm256_min_ps(v255, _mm256_max_ps(v0,
                                   _mm256_fmsub_ps(v9, r4c, sum4)));
                    _mm256_storeu_ps(o[3] + j, val4);
                }
            }
        } // j loop

        // ---- 尾部: mask load/store 处理剩余不足 8 个元素 ----
        int j_tail = j_end;
        if (j_tail < width - 1) {
            int rem = width - 1 - j_tail;
            if (rem > 0) {
                unsigned int mask_bits[8] = {0};
                for (int k = 0; k < rem; k++)
                    mask_bits[k] = 0xffffffff;
                __m256i r_mask = _mm256_loadu_si256((__m256i const*)mask_bits);

                // 加载 r0-r3
                __m256 t0_0 = _mm256_maskload_ps(r[0] + j_tail - 1, r_mask);
                __m256 t0_1 = _mm256_maskload_ps(r[0] + j_tail, r_mask);
                __m256 t0_2 = _mm256_maskload_ps(r[0] + j_tail + 1, r_mask);
                __m256 s0t = _mm256_add_ps(_mm256_add_ps(t0_0, t0_1), t0_2);

                __m256 t1_0 = _mm256_maskload_ps(r[1] + j_tail - 1, r_mask);
                __m256 t1_1 = _mm256_maskload_ps(r[1] + j_tail, r_mask);
                __m256 t1_2 = _mm256_maskload_ps(r[1] + j_tail + 1, r_mask);
                __m256 s1t = _mm256_add_ps(_mm256_add_ps(t1_0, t1_1), t1_2);

                __m256 t2_0 = _mm256_maskload_ps(r[2] + j_tail - 1, r_mask);
                __m256 t2_1 = _mm256_maskload_ps(r[2] + j_tail, r_mask);
                __m256 t2_2 = _mm256_maskload_ps(r[2] + j_tail + 1, r_mask);
                __m256 s2t = _mm256_add_ps(_mm256_add_ps(t2_0, t2_1), t2_2);

                __m256 t3_0 = _mm256_maskload_ps(r[3] + j_tail - 1, r_mask);
                __m256 t3_1 = _mm256_maskload_ps(r[3] + j_tail, r_mask);
                __m256 t3_2 = _mm256_maskload_ps(r[3] + j_tail + 1, r_mask);
                __m256 s3t = _mm256_add_ps(_mm256_add_ps(t3_0, t3_1), t3_2);

                // o[0]
                __m256 sum1t = _mm256_add_ps(_mm256_add_ps(s0t, s1t), s2t);
                __m256 val1t = _mm256_min_ps(v255, _mm256_max_ps(v0,
                                _mm256_fmsub_ps(v9, t1_1, sum1t)));
                _mm256_maskstore_ps(o[0] + j_tail, r_mask, val1t);

                // o[1]
                __m256 sum2t = _mm256_add_ps(_mm256_add_ps(s1t, s2t), s3t);
                __m256 val2t = _mm256_min_ps(v255, _mm256_max_ps(v0,
                                _mm256_fmsub_ps(v9, t2_1, sum2t)));
                _mm256_maskstore_ps(o[1] + j_tail, r_mask, val2t);

                if (actual_bs >= 3) {
                    __m256 t4_0 = _mm256_maskload_ps(r[4] + j_tail - 1, r_mask);
                    __m256 t4_1 = _mm256_maskload_ps(r[4] + j_tail, r_mask);
                    __m256 t4_2 = _mm256_maskload_ps(r[4] + j_tail + 1, r_mask);
                    __m256 s4t = _mm256_add_ps(_mm256_add_ps(t4_0, t4_1), t4_2);

                    // o[2]
                    __m256 sum3t = _mm256_add_ps(_mm256_add_ps(s2t, s3t), s4t);
                    __m256 val3t = _mm256_min_ps(v255, _mm256_max_ps(v0,
                                    _mm256_fmsub_ps(v9, t3_1, sum3t)));
                    _mm256_maskstore_ps(o[2] + j_tail, r_mask, val3t);

                    if (actual_bs == 4) {
                        __m256 t5_0 = _mm256_maskload_ps(r[5] + j_tail - 1, r_mask);
                        __m256 t5_1 = _mm256_maskload_ps(r[5] + j_tail, r_mask);
                        __m256 t5_2 = _mm256_maskload_ps(r[5] + j_tail + 1, r_mask);
                        __m256 s5t = _mm256_add_ps(_mm256_add_ps(t5_0, t5_1), t5_2);

                        // o[3]
                        __m256 sum4t = _mm256_add_ps(_mm256_add_ps(s3t, s4t), s5t);
                        __m256 val4t = _mm256_min_ps(v255, _mm256_max_ps(v0,
                                        _mm256_fmsub_ps(v9, t4_1, sum4t)));
                        _mm256_maskstore_ps(o[3] + j_tail, r_mask, val4t);
                    }
                }
            }
        }
    }
}

#endif // __KERNEL_AVX2_H__
