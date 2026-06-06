# ============================================================
# 9-point Stencil 优化 Makefile
# 当前最佳: ~87.6ms (grouped addition + stream + P-core 12T)
# 平台: i9-13900H (6P+8E), 原生 Linux, GCC 11.4.0
# ============================================================

CXX=g++
CXXFLAGS_OPT=-Iinclude/ -Ofast -fopenmp -mavx2 -mfma -march=native -no-pie -g
CXXFLAGS_AVX2=-Iinclude/ -Ofast -fopenmp -mavx2 -mfma -march=native -no-pie -g -DUSE_AVX2_KERNEL
CXXFLAGS_BLOCK=-Iinclude/ -Ofast -fopenmp -mavx2 -mfma -march=native -no-pie -g -DUSE_BLOCK_KERNEL
CXXFLAGS_DBG=-Iinclude/ -O0 -fopenmp -mavx2 -mfma -no-pie -g
LDFLAGS=lib/libpng16.a lib/libz.a -lm
SRCS=src/main.cc src/image.cc src/stencil.cc
# 默认：16 线程 spread（全核心最优）
RUN_ENV=OMP_NUM_THREADS=16 OMP_PROC_BIND=spread
# P-core 最优配置 (i9-13900H: CPU 0-11 为 P-core)
P_CORE=OMP_NUM_THREADS=12 OMP_PROC_BIND=spread taskset -c 0-11

# ===== 优化版本 (默认: 分组加法 + stream store + prefetch) =====
all: bin/stencil
bin/stencil: $(SRCS) src/stencil.h src/image.h
	$(CXX) $(CXXFLAGS_OPT) -o bin/stencil $(SRCS) $(LDFLAGS)
	@echo "=== Optimized build (grouped addition + stream + prefetch) done ==="

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

# ===== 简单版本 (IPCC 分组加法 + row-by-row) =====
simple: bin/stencil_simple
bin/stencil_simple: src/main.cc src/image.cc src/stencil_simple.cc src/stencil.h src/image.h
	$(CXX) $(CXXFLAGS_OPT) -o bin/stencil_simple src/main.cc src/image.cc src/stencil_simple.cc $(LDFLAGS)
	@echo "=== Simple kernel build done ==="

# ===== 调试版本 (O0) =====
debug: bin/stencil_dbg
bin/stencil_dbg: $(SRCS) src/stencil.h src/image.h src/kernel_avx2.h
	$(CXX) $(CXXFLAGS_DBG) -o bin/stencil_dbg $(SRCS) $(LDFLAGS)
	@echo "=== Debug build done ==="

# ===== 运行 =====
run: all
	$(RUN_ENV) ./bin/stencil IPCC.png

run-pcore: all
	$(P_CORE) ./bin/stencil IPCC.png

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
		echo "✅ data.txt == check.txt (完全正确)" || \
		(echo "❌ 有差异:" && diff data.txt check.txt | head -20)

check-all: all avx2 block simple
	@echo "=== 验证所有版本 ==="
	@rm -f data.txt; ./bin/stencil IPCC.png 2>&1 > /dev/null; \
		diff data.txt check.txt > /dev/null 2>&1 && echo "✅ stencil" || echo "❌ stencil"
	@rm -f data.txt; ./bin/stencil_avx2 IPCC.png 2>&1 > /dev/null; \
		diff data.txt check.txt > /dev/null 2>&1 && echo "✅ stencil_avx2" || echo "❌ stencil_avx2"
	@rm -f data.txt; ./bin/stencil_block IPCC.png 2>&1 > /dev/null; \
		diff data.txt check.txt > /dev/null 2>&1 && echo "✅ stencil_block" || echo "❌ stencil_block"
	@rm -f data.txt; ./bin/stencil_simple IPCC.png 2>&1 > /dev/null; \
		diff data.txt check.txt > /dev/null 2>&1 && echo "✅ stencil_simple" || echo "❌ stencil_simple"

# 从压缩文件解压验证数据
check.txt: check.txt.gz
	@echo "Decompressing check.txt.gz ..."
	@gunzip -kf check.txt.gz

# ===== 性能对比 =====
bench: all debug
	@echo "=== Debug (O0, 4T) ===" && \
	OMP_NUM_THREADS=4 ./bin/stencil_dbg IPCC.png 2>&1 | grep -E "Total"
	@echo "=== Optimized (16T spread) ===" && \
	$(RUN_ENV) ./bin/stencil IPCC.png 2>&1 | grep -E "Total"
	@echo "=== Optimized (P-core 12T) ===" && \
	$(P_CORE) ./bin/stencil IPCC.png 2>&1 | grep -E "Total"

# ===== 全量对比 =====
bench-all: all avx2 block simple
	@echo "========================================="
	@echo "  全量性能对比 (nTrials=10)"
	@echo "========================================="
	@echo ""
	@echo "--- stencil (grouped addition + stream) ---"
	@$(RUN_ENV) ./bin/stencil IPCC.png 2>&1 | grep -E "Total"
	@echo ""
	@echo "--- stencil P-core 12T ---"
	@$(P_CORE) ./bin/stencil IPCC.png 2>&1 | grep -E "Total"
	@echo ""
	@echo "--- stencil_avx2 (sequential + 2x unroll) ---"
	@$(RUN_ENV) ./bin/stencil_avx2 IPCC.png 2>&1 | grep -E "Total"
	@echo ""
	@echo "--- stencil_block (读6算4) ---"
	@$(RUN_ENV) ./bin/stencil_block IPCC.png 2>&1 | grep -E "Total"
	@echo ""
	@echo "--- stencil_simple (grouped addition) ---"
	@$(RUN_ENV) ./bin/stencil_simple IPCC.png 2>&1 | grep -E "Total"

# ===== 线程数扫描 =====
scan-threads: all
	@echo "=== 线程扫描 (spread, 全核心) ==="
	@for n in 4 8 12 16 20; do \
		echo -n "T=$$n: "; \
		OMP_NUM_THREADS=$$n OMP_PROC_BIND=spread ./bin/stencil IPCC.png 2>&1 | grep "Total" | awk '{print $$NF}'; \
	done

scan-pcore: all
	@echo "=== P-core 绑定扫描 ==="
	@for n in 6 12; do \
		echo -n "P-core $$n""T spread: "; \
		OMP_NUM_THREADS=$$n OMP_PROC_BIND=spread taskset -c 0-11 ./bin/stencil IPCC.png 2>&1 | grep "Total" | awk '{print $$NF}'; \
	done
	@for n in 6 12; do \
		echo -n "P-core $$n""T close: "; \
		OMP_NUM_THREADS=$$n OMP_PROC_BIND=close taskset -c 0-11 ./bin/stencil IPCC.png 2>&1 | grep "Total" | awk '{print $$NF}'; \
	done

# ===== 大页内存测试 (需 root) =====
hugepages-test: all
	@echo "=== 大页内存性能测试 ==="
	@echo -n "默认 4KB 页: "
	@$(P_CORE) ./bin/stencil IPCC.png 2>&1 | grep "Total" | awk '{print $$NF}'
	@if [ -w /proc/sys/vm/nr_hugepages ]; then \
		echo "分配 512 个大页 (1GB)..."; \
		echo 512 | sudo tee /proc/sys/vm/nr_hugepages > /dev/null 2>&1; \
		echo -n "大页 (2MB): "; \
		$(P_CORE) ./bin/stencil IPCC.png 2>&1 | grep "Total" | awk '{print $$NF}'; \
		echo 0 | sudo tee /proc/sys/vm/nr_hugepages > /dev/null 2>&1; \
	else \
		echo "无 root 权限，跳过大页测试"; \
	fi

# ===== 多图像交叉验证 =====
validate-multi: all debug
	@echo "=== 多图像交叉验证 (opt vs debug 基准) ==="
	@for img in IPCC.png test_1024.png test_2048.png test_4096x2048.png test_800x600.png; do \
		echo -n "$$img: "; \
		if [ -f "$$img" ]; then \
			rm -f data.txt; \
			./bin/stencil $$img 2>&1 > /dev/null; \
			mv data.txt /tmp/data_opt.txt 2>/dev/null; \
			OMP_NUM_THREADS=4 ./bin/stencil_dbg $$img 2>&1 > /dev/null; \
			mv data.txt /tmp/data_dbg.txt 2>/dev/null; \
			if diff /tmp/data_opt.txt /tmp/data_dbg.txt > /dev/null 2>&1; then \
				echo "✅ opt == dbg"; \
			else \
				echo "❌ mismatch"; \
				diff /tmp/data_opt.txt /tmp/data_dbg.txt | head -3; \
			fi; \
			rm -f /tmp/data_opt.txt /tmp/data_dbg.txt output.png; \
		else \
			echo "⏭ file not found"; \
		fi; \
	done

clean:
	rm -f bin/stencil* output.png data.txt check.txt src/*~
