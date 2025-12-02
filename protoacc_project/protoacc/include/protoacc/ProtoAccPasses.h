#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "protoacc/ProtoAccOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <memory>

namespace protoacc {

#define GEN_PASS_DECL
#include "protoacc/ProtoAccPasses.h.inc"

// std::unique_ptr<mlir::Pass> createConvertProtoAccToArithPass(ConvertProtoAccToArithOptions options={});
// std::unique_ptr<mlir::Pass> createDCEPass();
std::unique_ptr<mlir::Pass> createBitPackPass();
std::unique_ptr<mlir::Pass> createConvertToProtoAccPass();
std::unique_ptr<mlir::Pass> createLowerProtoAccToVectorPass();

#define GEN_PASS_REGISTRATION
#include "protoacc/ProtoAccPasses.h.inc"

}