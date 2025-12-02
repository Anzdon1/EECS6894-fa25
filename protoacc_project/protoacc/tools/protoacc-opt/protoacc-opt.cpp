#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
// #include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// 导入 Func Dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 导入 MLIR 自带 Pass
#include "mlir/Transforms/Passes.h"
// 导入我们新建的 Dialect
#include "protoacc/ProtoAccDialect.h"
#include "protoacc/ProtoAccPasses.h"
using namespace mlir;
using namespace llvm;

int main(int argc, char ** argv) {
  DialectRegistry registry;
  
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::DLTIDialect>();
  registry.insert<mlir::arith::ArithDialect>();

  registry.insert<mlir::vector::VectorDialect>();


  // 注册 Dialect
  registry.insert<protoacc::ProtoAccDialect, func::FuncDialect, arith::ArithDialect>();
  // registry.insert<LLVM::LLVMDialect>();
  // 注册两个 Pass
  registerCSEPass();
  registerCanonicalizerPass();
  protoacc::registerPasses();

  // registerAllDialects(registry);
  // registerLLVMDialectTranslation(registry);
  // registerAllTranslations();


  return asMainReturnCode(MlirOptMain(argc, argv, "protoacc-opt", registry));
}
