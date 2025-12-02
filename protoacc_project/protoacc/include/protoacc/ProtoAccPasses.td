#ifndef PROTOACC_PASSES_TD
#define PROTOACC_PASSES_TD

include "mlir/Pass/PassBase.td"

// def ConvertProtoAccToArith : Pass<"convert-protoacc-to-arith"> {
//   let summary = "Convert ProtoAcc To Arith";
//   let constructor = "protoacc::createConvertProtoAccToArithPass()";
//   let options = [
//     Option<"name", "name", "std::string", "", "help">
//   ];
// }

// def DCE : Pass<"protoacc-dce", "protoacc::FuncOp"> {
//   let summary = "dce";
//   let constructor = "protoacc::createDCEPass()";
// }

def BitPack : Pass<"protoacc-bitpack", "::mlir::LLVM::LLVMFuncOp"> {
  let summary = "Accelerate bitpack";
  let constructor = "protoacc::createBitPackPass()";
}

def ConvertToProtoAcc : Pass<"convert-to-protoacc"> {
  let summary = "Match protobuf varint patterns and replace with protoacc.decode_varint";

  let constructor = "createConvertToProtoAccPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "protoacc::ProtoAccDialect"];
}

def LowerProtoAccToVector : Pass<"lower-protoacc-to-vector"> {
  let summary = "Lower ProtoAcc ops to vectorized implementations";

  let constructor = "createLowerProtoAccToVectorPass()";
  let dependentDialects = ["mlir::vector::VectorDialect", "mlir::arith::ArithDialect", "mlir::LLVM::LLVMDialect", "protoacc::ProtoAccDialect"];
}


#endif
