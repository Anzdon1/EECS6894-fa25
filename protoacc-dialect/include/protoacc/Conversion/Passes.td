//===- Passes.td ------------------------------------------------*- tablegen -*-===//
include "mlir/Pass/PassBase.td"

def ConvertProtoAccToLLVMPass
  : Pass<"convert-protoacc-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower ProtoAcc ops to LLVM dialect";
  let description = [{
    Lower protoacc.zigzag_encode (i32) to llvm.shl/llvm.ashr/llvm.xor.
  }];
}
