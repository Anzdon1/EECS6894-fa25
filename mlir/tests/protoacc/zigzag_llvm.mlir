// RUN: protoacc-opt %s --convert-protoacc-to-llvm | mlir-opt | FileCheck %s
module {
  func.func @demo(%x: i32) -> i32 {
    %y = "protoacc.zigzag_encode"(%x : i32) : (i32) -> i32
    return %y : i32
  }
}
// CHECK: llvm.shl
// CHECK: llvm.ashr
// CHECK: llvm.xor
