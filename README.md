# 9-Point Stencil 图像处理并行优化

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5-green.svg)](https://www.openmp.org/)
[![AVX2](https://img.shields.io/badge/SIMD-AVX2%20%2B%20FMA-orange.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)

> 2020 年首届 ACM 中国-国际并行计算挑战赛 (IPCC) 全国初赛题目 —— 基于 stencil 计算的图像边缘检测并行优化。

## 📊 性能概览

| 平台 | CPU | 最优时间 (10 次) | 加速比 | 实现 |
|------|-----|-----------------|--------|------|
| **本仓库** | Core Ultra 9 275HX (24C) | **62.5 ms** | **136.5x** | AVX2 + Stream + OpenMP |
| 原始 IPCC | AMD EPYC 7452×2 (64C) | 15.8 ms | 695.5x | AVX2 + Stream + NUMA |
| 基线 (O0) | Core Ultra 9 275HX | 8529.8 ms | 1.0x | 串行标量 |

> 在相同硬件 (Arrow Lake, 24 核, WSL2) 上，IPCC 参考代码与本地优化版几乎持平，差距 < 7%。

## 📁 目录结构

```
Stencil/
├── README.md                    # 项目总览 (本文件)
├── 优化记录.md                  # ★ 完整优化记录 (推荐阅读)
├── 优化报告.md                  # 优化路径与方法论总结
├── 实践计划.md                  # 分阶段优化计划
├── 实践题目.md                  # 原始赛题要求
│
├── stencil/                     # ★ 优化后项目 (可编译运行)
│   ├── Makefile                 #   多目标编译 (avx2/block/simple/debug)
│   ├── src/                     #   源代码
│   │   ├── main.cc              #     主程序入口 (计时部分不可修改)
│   │   ├── stencil.cc           #     ★ 默认优化: 2x unroll + stream + prefetch
│   │   ├── stencil_avx2.cc      #     AVX2 逐行 kernel
│   │   ├── stencil_block.cc     #     分块 kernel (读 6 行算 4 行)
│   │   ├── stencil_simple.cc    #     1D 线性化 (对标 IPCC opt_kernel)
│   │   ├── kernel_avx2.h        #     AVX2 kernel 集合
│   │   ├── image.h / image.cc   #     PNG 图像 I/O
│   │   └── stencil.h            #     ApplyStencil 模板声明
│   ├── include/                 #   PNG 库头文件
│   ├── lib/                     #   静态库 (libpng16.a, libz.a)
│   ├── IPCC.png                 #   输入图像 (10410×5905)
│   ├── check.png / check.txt    #   验证文件
│   └── bin/                     #   编译输出 (gitignored)
│
└── IPCC_stencil/                # IPCC 官方参考实现
    ├── Makefile*                 #   多种编译器 Makefile (gcc/clang/icc/debian)
    ├── src/
    │   ├── opt_kernel.h          #     ★ 1D 线性化 + AVX2 + stream (62.5ms)
    │   ├── my_kernel.h           #     分块 kernel + 线程亲和性
    │   ├── kernel_r6c4_avx2.h    #     读 6 算 4 核心 kernel
    │   └── kernel_r6c4_avx2_prefetch.h  #  带预取的 r6c4 kernel
    ├── include/
    ├── lib/
    └── README.md                 #   原始 README
```

## 🚀 快速开始

### 环境要求

- GCC 8+ (推荐 9.4+)
- GNU Make
- CPU 支持 AVX2 + FMA 指令集
- OpenMP 运行时库

### 编译 & 运行

```bash
# 编译优化版 stencil 项目
cd stencil
make all          # 编译默认优化版 (2x unroll + stream + prefetch)
make avx2         # 编译逐行 AVX2 kernel 版本
make block        # 编译分块 kernel 版本

# 运行 (自动使用最优线程配置)
make run          # 20 线程, spread 绑定

# 验证正确性
make check        # 对比 data.txt 与 check.txt

# 性能对比
make bench-all    # 运行所有版本并对比 Total 时间

# 线程数扫描
make scan-threads # 测试 4/8/12/16/20/24 线程性能
```

```bash
# 编译 IPCC 参考实现
cd IPCC_stencil
make -f Makefile.gcc-debian

# 运行 (手动指定线程)
OMP_NUM_THREADS=20 OMP_PROC_BIND=spread ./bin/stencil IPCC.png
```

### 一键运行对比

```bash
# 在 stencil/ 目录下
make bench-all
```

## 📈 优化技术栈

| 技术 | 效果 | 实现位置 |
|------|------|---------|
| `-Ofast -march=native` | ~1.5× | Makefile CXXFLAGS |
| OpenMP 多线程 (20T) | ~10.7× | `#pragma omp parallel for` |
| AVX2 手动向量化 (8 floats) | ~5.8× | `_mm256_loadu/add/fmsub/max/min` |
| Stream Store (非临时写入) | ~2.1× | `_mm256_stream_ps` |
| 2× 循环展开 + Prefetch | ~1.1× | 手动展开 + `_mm_prefetch` |
| 分组加法 (ILP 优化) | ~1.07× | IPCC opt_kernel.h 分组策略 |

**最终**: `1.5 × 10.7 × 5.8 × 2.1 × 1.1 ≈ 127×` (实测 127.2×，理论 215×，差距来自内存带宽饱和)

## 🔬 IPCC vs 本项目对比

### 关键差异

| 维度 | IPCC opt_kernel | 本项目 stencil.cc |
|------|----------------|-------------------|
| **加法依赖链** | ~5 级 (分组独立子树) | ~11 级 (顺序累加) |
| **对齐处理** | 1 次前置对齐 | 每行对齐检测 (5905 次) |
| **寄存器压力** | ~8 YMM | ~14 YMM |
| **循环展开** | 无 | 2× (每迭代 16 元素) |
| **软件预取** | 无 | _mm_prefetch T0 |
| **正确性** | ✅ | ✅ |
| **性能 (同硬件)** | **62.5 ms** | 67.1 ms |

**结论**: IPCC 的**分组加法策略**将 9 个邻域像素分入 4 个独立加法子树再合并，依赖链深度从 11 级降至 5 级，更好地利用了 Arrow Lake 的宽发射 ILP 能力。详见 [优化记录 §6.6](stencil/优化记录.md#66-ipcc-参考实现-vs-本项目实现对比)。

## 📊 性能演进

| 阶段 | 总时间 (10 次) | 加速比 | 技术 |
|------|---------------|--------|------|
| 0-baseline | 8529.8 ms | 1.0× | O0, serial, scalar |
| 1-O3 | 5855.6 ms | 1.5× | `-O3 -ffast-math` |
| 2-OpenMP | 796.4 ms | 10.7× | 4 线程并行 |
| 3-AVX2 | 138.2 ms | 61.7× | 手动 SIMD 向量化 |
| 4-Stream | ~67 ms | ~127× | 非临时写入 |
| 5-2× Unroll | 67.1 ms | 127.2× | 循环展开 + 预取 |
| **IPCC ref** | **62.5 ms** | **136.5×** | 分组加法 ILP |

## 📖 文档导航

| 文档 | 内容 |
|------|------|
| [优化记录.md](stencil/优化记录.md) | ★ 完整优化历程、性能对比、技术决策、IPCC 对比分析 |
| [优化报告.md](优化报告.md) | 优化方法论文档，七步优化路径 |
| [实践计划.md](实践计划.md) | 分阶段优化计划 (6 阶段) |
| [实践题目.md](实践题目.md) | 原始赛题要求 |
| [IPCC_stencil/README.md](IPCC_stencil/README.md) | IPCC 官方参考代码说明 |

## 🛠️ 已知限制

- **AVX-512**: Core Ultra 9 275HX (Arrow Lake) 不支持 AVX-512 指令集
- **P-core 绑定**: WSL2 不暴露 hybrid CPU 拓扑，无法区分 P-core / E-core
- **大页内存**: 需 root 权限配置，WSL2 支持不完整
- **编译器**: 当前用 GCC 9.4.0，Intel oneAPI (ICX) 可能生成更优代码

## 📄 License

MIT License — 详见各子目录。

---

*本仓库为 [IPCC 2020 初赛题目](https://github.com/04lhy/Stencil) 的完整实现与优化。*
*性能数据基于 Intel Core Ultra 9 275HX (Arrow Lake), 24C/24T, WSL2, GCC 9.4.0。*
