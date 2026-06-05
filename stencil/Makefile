# ============================================================
# 9-point Stencil 优化 Makefile
# 当前最佳: ~67ms (OMP_NUM_THREADS=20)
# 加速比: ~127x over baseline (8529.8ms)
# 目标: 200x+ (2x unroll + prefetch + block kernel with stream)
# ============================================================

CXX=g++
CXXFLAGS_OPT=-Iinclude/ -Ofast -fopenmp -mavx2 -mfma -march=native -no-pie -g
CXXFLAGS_AVX2=-Iinclude/ -Ofast -fopenmp -mavx2 -mfma -march=native -no-pie -g -DUSE_AVX2_KERNEL
CXXFLAGS_BLOCK=-Iinclude/ -Ofast -fopenmp -mavx2 -mfma -march=native -no-pie -g -DUSE_BLOCK_KERNEL
CXXFLAGS_DBG=-Iinclude/ -O0 -fopenmp -mavx2 -mfma -no-pie -g
LDFLAGS=lib/libpng16.a lib/libz.a -lm
SRCS=src/main.cc src/image.cc src/stencil.cc
RUN_ENV=OMP_NUM_THREADS=20 OMP_PROC_BIND=spread

# ===== 优化版本 (默认: 2x 展开 + stream store + prefetch) =====
all: bin/stencil
bin/stencil: $(SRCS) src/stencil.h src/image.h
	$(CXX) $(CXXFLAGS_OPT) -o bin/stencil $(SRCS) $(LDFLAGS)
	@echo "=== Optimized build (2x unroll + stream + prefetch) done ==="

# ===== AVX2 kernel 版本 (kernel_avx2.h: row kernel) =====
avx2: bin/stencil_avx2
bin/stencil_avx2: src/main.cc src/image.cc src/stencil_avx2.cc src/stencil.h src/image.h src/kernel_avx2.h
	$(CXX) $(CXXFLAGS_AVX2) -o bin/stencil_avx2 src/main.cc src/image.cc src/stencil_avx2.cc $(LDFLAGS)
	@echo "=== AVX2 row kernel build done ==="

# ===== Block kernel 版本 (kernel_avx2.h: block kernel) =====
block: bin/stencil_block
bin/stencil_block: src/main.cc src/image.cc src/stencil_block.cc src/stencil.h src/image.h src/kernel_avx2.h
	$(CXX) $(CXXFLAGS_BLOCK) -o bin/stencil_block src/main.cc src/image.cc src/stencil_block.cc $(LDFLAGS)
	@echo "=== Block kernel build done ==="

# ===== 简单版本 (对标 IPCC opt_kernel: 1D 线性化 + stream store) =====
simple: bin/stencil_simple
bin/stencil_simple: src/main.cc src/image.cc src/stencil_simple.cc src/stencil.h src/image.h
	$(CXX) $(CXXFLAGS_OPT) -o bin/stencil_simple src/main.cc src/image.cc src/stencil_simple.cc $(LDFLAGS)
	@echo "=== Simple 1D-linear kernel build done ==="

# ===== 调试版本 (O0) =====
debug: bin/stencil_dbg
bin/stencil_dbg: $(SRCS) src/stencil.h src/image.h src/kernel_avx2.h
	$(CXX) $(CXXFLAGS_DBG) -o bin/stencil_dbg $(SRCS) $(LDFLAGS)
	@echo "=== Debug build done ==="

# ===== 运行 =====
run: all
	$(RUN_ENV) ./bin/stencil IPCC.png

run-avx2: avx2
	$(RUN_ENV) ./bin/stencil_avx2 IPCC.png

run-block: block
	$(RUN_ENV) ./bin/stencil_block IPCC.png

run-simple: simple
	$(RUN_ENV) ./bin/stencil_simple IPCC.png

run-debug: debug
	OMP_NUM_THREADS=4 ./bin/stencil_dbg IPCC.png

# ===== 验证 =====
check: check.txt
	@diff data.txt check.txt > /dev/null 2>&1 && \
		echo "✓ data.txt == check.txt (完全正确)" || \
		(echo "✗ 有差异:" && diff data.txt check.txt | head -20)

# 从压缩文件解压验证数据
check.txt: check.txt.gz
	@echo "Decompressing check.txt.gz ..."
	@gunzip -kf check.txt.gz

# ===== 性能对比 =====
bench: all debug
	@echo "=== 基准测试 (O0, 4T) ===" && \
	OMP_NUM_THREADS=4 ./bin/stencil_dbg IPCC.png 2>&1 | grep -E "Total"
	@echo "=== 优化版本 ($(RUN_ENV)) ===" && \
	$(RUN_ENV) ./bin/stencil IPCC.png 2>&1 | grep -E "Total"

# ===== 全量对比 =====
bench-all: all avx2 block simple
	@echo "========================================="
	@echo "  全量性能对比 (nTrials=10)"
	@echo "========================================="
	@echo ""
	@echo "--- stencil (2x unroll + stream) ---"
	@$(RUN_ENV) ./bin/stencil IPCC.png 2>&1 | grep -E "Total"
	@echo ""
	@echo "--- stencil_avx2 (row kernel) ---"
	@$(RUN_ENV) ./bin/stencil_avx2 IPCC.png 2>&1 | grep -E "Total"
	@echo ""
	@echo "--- stencil_block (block kernel) ---"
	@$(RUN_ENV) ./bin/stencil_block IPCC.png 2>&1 | grep -E "Total"
	@echo ""
	@echo "--- stencil_simple (1D linear) ---"
	@$(RUN_ENV) ./bin/stencil_simple IPCC.png 2>&1 | grep -E "Total"

# ===== 线程数扫描 =====
scan-threads: all
	@for n in 4 8 12 16 20 24; do \
		echo -n "OMP_NUM_THREADS=$$n: "; \
		OMP_NUM_THREADS=$$n OMP_PROC_BIND=close ./bin/stencil IPCC.png 2>&1 | grep "Total" | awk '{print $$NF}'; \
	done

clean:
	rm -f bin/stencil* output.png data.txt check.txt src/*~
