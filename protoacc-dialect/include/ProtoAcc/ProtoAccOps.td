include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ProtoAccDialect.td"

def ProtoAcc_ZigZagEncodeOp : Op<ProtoAcc_Dialect, "zigzag.encode", [Pure]> {
  let summary = "ZigZag encode (int32 -> int32)";
  let description = [{
    Computes (x << 1) ^ (x >> 31) for signed 32-bit integers.
  }];
  let arguments = (ins I32:$in);
  let results   = (outs I32:$out);
  let assemblyFormat = "$in attr-dict";
}
