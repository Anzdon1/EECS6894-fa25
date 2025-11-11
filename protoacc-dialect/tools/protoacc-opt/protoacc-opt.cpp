//===- protoacc-opt.cpp ----------------------------------------------------===//
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

std::unique_ptr<Pass> createConvertProtoAccToLLVMPass();

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<LLVM::LLVMDialect>();
  registerAllPasses();

  PassRegistration reg(
    "convert-protoacc-to-llvm",
    "Lower ProtoAcc ops to LLVM dialect",
    [] { return createConvertProtoAccToLLVMPass(); });

  return asMainReturnCode(
    MlirOptMain(argc, argv, "ProtoAcc optimizer\n", registry));
}
