//===- ProtoAccOps.td -------------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "protoacc/IR/ProtoAccDialect.td"

def ProtoAcc_ZigZagEncodeOp : Op<ProtoAcc_Dialect, "zigzag_encode", [Pure]> {
  let summary = "ZigZag encode (i32)";
  let description = [{ Computes (x << 1) ^ (x >> 31) on i32. }];
  let arguments = (ins I32:$in);
  let results   = (outs I32:$out);
  let assemblyFormat = "$in attr-dict";
}
