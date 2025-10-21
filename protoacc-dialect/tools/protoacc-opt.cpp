#include "ProtoAcc/ProtoAccDialect.h"
#include "ProtoAcc/Transforms/Passes.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::DLTIDialect, mlir::LLVM::LLVMDialect,
                  protoacc::ProtoAccDialect>();

  protoacc::registerProtoAccPasses();

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "ProtoAcc optimizer", registry));
}
