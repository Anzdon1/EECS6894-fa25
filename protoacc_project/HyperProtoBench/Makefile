LLVM 	 = ~/EECS6894-fa25/protoacc_project/install
PROTOC   = /usr/bin/protoc
CLANG    = $(LLVM)/bin/clang
MLIR_OPT = $(LLVM)/bin/mlir-opt
MLIR_TRANSLATE = $(LLVM)/bin/mlir-translate
PROTOACC_OPT = ~/EECS6894-fa25/protoacc_project/protoacc/build/protoacc-opt

OUT = benchmark_mlir_exec

CPPFLAGS = -std=c++17 -O2 -g -fno-exceptions \
            -DGOOGLE_PROTOBUF_NO_RTTI \
            -DGOOGLE_PROTOBUF_MIN_LOG_LEVEL=3

LDFLAGS = -lprotobuf -lpthread

SRC     = benchmark.cc
PROTO   = benchmark.proto
PB_CC   = benchmark.pb.cc
PB_H    = benchmark.pb.h
INC     = benchmark.inc

PB_LL        		= benchmark_pb.ll
PB_MLIR      		= benchmark_pb.mlir
PB_OPT_MLIR  		= benchmark_pb_opt.mlir
PB_FINAL_LL  		= benchmark_pb_final.ll

PB_MLIR_PROTOACC      = benchmark_pb_protoacc.mlir
PB_OPT_MLIR_PROTOACC  = benchmark_pb_opt_protoacc.mlir
PB_FINAL_LL_PROTOACC  = benchmark_pb_final_protoacc.ll
OUT_PROTOACC          = benchmark_mlir_exec_protoacc

all: $(OUT)

clean:
	find . -name "$(OUT_PROTOACC)" -delete
	find . -name "$(OUT)" -delete
	find . -name "*.ll" -delete
	find . -name "*.mlir" -delete
	find . -name "benchmark.pb.cc" -delete
	find . -name "benchmark.pb.h" -delete
	@echo "CLEAN. All intermediate files removed recursively."

protoacc: $(OUT_PROTOACC)

# 1. Generate protobuf C++ files
$(PB_CC) $(PB_H): $(PROTO)
	@echo " ------------- 1. Generate protobuf C++ files ------------- "
	$(PROTOC) $< --cpp_out=.

# 2. Emit LLVM IR from protobuf C++
$(PB_LL): $(PB_CC)
	@echo " ------------- 2. Emit LLVM IR from protobuf C++ ------------- "
	$(CLANG) $(CPPFLAGS) -g0 -emit-llvm -S $(PB_CC) -o $@

# 3. Import LLVM IR → MLIR
$(PB_MLIR): $(PB_LL)
	@echo " ------------- 3. Import LLVM IR → MLIR ------------- "
	$(MLIR_TRANSLATE) --import-llvm $< -o $@

# 4. Run MLIR optimization passes
$(PB_OPT_MLIR): $(PB_MLIR)
	@echo " ------------- 4. Run MLIR optimization passes ------------- "
	$(MLIR_OPT) $< \
		--canonicalize --cse --inline --symbol-dce \
		-o $@

# 5. Export MLIR → LLVM IR
$(PB_FINAL_LL): $(PB_MLIR)
	@echo " ------------- 5. Export MLIR → LLVM IR ------------- "
	$(MLIR_TRANSLATE) --mlir-to-llvmir $< -o $@

# 6. Link final executable
$(OUT): $(PB_FINAL_LL) $(SRC) $(INC)
	@echo " ------------- 6. Link final executable ------------- "
	$(CLANG) $(PB_FINAL_LL) $(SRC) $(CPPFLAGS) -O2 -std=c++17 \
		-o $@ -lprotobuf -lpthread -lstdc++ -lm
	@echo "DONE. Built $(OUT)"


# 4.5 Apply ProtoAcc BitPack optimization
$(PB_MLIR_PROTOACC): $(PB_MLIR)
	@echo " ------------- 4.5. Apply ProtoAcc BitPack optimization ------------- "
	$(PROTOACC_OPT) $< \
		--protoacc-bitpack -o $@

# 5. ProtoAcc export MLIR → LLVM IR
$(PB_FINAL_LL_PROTOACC): $(PB_MLIR_PROTOACC)
	@echo " ------------- 5. [PROTOACC] Export MLIR → LLVM IR ------------- "
	$(MLIR_TRANSLATE) --mlir-to-llvmir $< -o $@

# 6. ProtoAcc link final executable
$(OUT_PROTOACC): $(PB_FINAL_LL_PROTOACC) $(SRC) $(INC)
	@echo " ------------- 6. [PROTOACC] Link final executable ------------- "
	$(CLANG) $(PB_FINAL_LL_PROTOACC) $(SRC) $(CPPFLAGS) -O2 -std=c++17 \
		-o $@ -lprotobuf -lpthread -lstdc++ -lm
	@echo "DONE. Built $(OUT_PROTOACC)"


